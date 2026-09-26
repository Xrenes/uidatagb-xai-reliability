"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet-50.

Produces heatmaps showing WHICH regions of the ultrasound image
the model focused on when making its prediction.

This is run on:
  1. Local client model  â†’ site-specific explanations
  2. Global (aggregated) model â†’ global explanations
  Then we compare both.

Reference: Selvaraju et al. (2017) https://arxiv.org/abs/1610.02391
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

from model import build_model, get_target_layer
from dataset import IDX_TO_CLASS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "gradcam")


# â”€â”€ Grad-CAM Core â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class GradCAM:
    """
    Hooks into the target convolutional layer of a model
    and computes gradient-based class activation maps.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates Grad-CAM heatmap for input_tensor.

        Args:
            input_tensor: shape (1, 3, H, W)
            target_class: class index to explain. If None, uses predicted class.

        Returns:
            heatmap (np.ndarray): shape (H, W), values in [0, 1]
            predicted_class (int)
            confidence (float)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)

        if target_class is None:
            target_class = pred_class.item()

        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Pool gradients over spatial dimensions (global average pooling)
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # shape: (C,)

        # Weight activations by pooled gradients
        activation_map = self.activations[0]  # shape: (C, H, W)
        for i in range(activation_map.shape[0]):
            activation_map[i] *= pooled_grads[i]

        # Average across channels and apply ReLU
        heatmap = activation_map.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, pred_class.item(), confidence.item()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Grad-CAM++ ──────────────────────────────
class GradCAMPlusPlus:
    """
    Grad-CAM++ (Chattopadhay et al., 2018): pixel-wise second-order weighting
    of gradients, giving sharper and more localised activation maps than
    standard Grad-CAM, particularly for small or multiple lesion regions.

    Reference: https://arxiv.org/abs/1710.11063
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates Grad-CAM++ heatmap for input_tensor.

        Args:
            input_tensor: shape (1, 3, H, W)
            target_class: class index to explain. If None, uses predicted class.

        Returns:
            heatmap (np.ndarray): shape (H, W), values in [0, 1]
            predicted_class (int)
            confidence (float)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)

        if target_class is None:
            target_class = pred_class.item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        grads = self.gradients[0]          # (C, H, W)
        acts = self.activations[0]         # (C, H, W)

        grads_sq = grads.pow(2)
        grads_cube = grads.pow(3)
        sum_acts = acts.sum(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        eps = 1e-8
        alpha_denom = 2 * grads_sq + sum_acts * grads_cube
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps)
        )
        alphas = grads_sq / alpha_denom

        pos_grads = F.relu(grads)
        weights = (alphas * pos_grads).sum(dim=(1, 2))  # (C,)

        activation_map = acts.clone()
        for i in range(activation_map.shape[0]):
            activation_map[i] *= weights[i]

        heatmap = F.relu(activation_map.sum(dim=0)).cpu().numpy()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, pred_class.item(), confidence.item()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Vanilla Gradient Saliency Map ────────────────────────
class SaliencyMap:
    """
    Vanilla gradient saliency (Simonyan et al., 2014): the absolute value of
    the gradient of the target class score with respect to the input pixels,
    S^c = |d y^c / d x|. Operates directly in input space rather than on a
    convolutional feature map, giving pixel-resolution (not 7x7-upsampled)
    attribution at the cost of noisier maps.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates a saliency map for input_tensor.

        Args:
            input_tensor: shape (1, 3, H, W)
            target_class: class index to explain. If None, uses predicted class.

        Returns:
            heatmap (np.ndarray): shape (H, W), values in [0, 1]
            predicted_class (int)
            confidence (float)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).clone().detach().requires_grad_(True)

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)

        if target_class is None:
            target_class = pred_class.item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        grad = input_tensor.grad[0].abs()           # (3, H, W)
        heatmap = grad.max(dim=0)[0].cpu().numpy()  # max over channels

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, pred_class.item(), confidence.item()

    def remove_hooks(self):
        """No hooks are registered; present for interface parity with GradCAM."""
        pass


# ── Eigen-CAM ────────────────────────────────────────────────────────────────
class EigenCAM:
    """
    Eigen-CAM (Muhammad & Yeasin, 2020): takes the first principal component
    of the target layer's activations as the class activation map. Unlike
    Grad-CAM/Grad-CAM++, it requires no backward pass and is class-agnostic
    (the map does not depend on which class is being explained), which makes
    it the cheapest of the four methods to compute.

    Reference: https://arxiv.org/abs/2008.00299
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer
        self.activations = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates an Eigen-CAM heatmap for input_tensor.

        Args:
            input_tensor: shape (1, 3, H, W)
            target_class: ignored (Eigen-CAM is class-agnostic); accepted
                          for interface parity with the other explainers.

        Returns:
            heatmap (np.ndarray): shape (H, W), values in [0, 1]
            predicted_class (int)
            confidence (float)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)

        acts = self.activations[0]                     # (C, H, W)
        c, h, w = acts.shape
        flat = acts.reshape(c, h * w).cpu().numpy()    # (C, H*W)

        flat = flat - flat.mean(axis=1, keepdims=True)
        u, s, vt = np.linalg.svd(flat, full_matrices=False)
        first_pc = vt[0].reshape(h, w)                 # (H, W)

        heatmap = np.maximum(first_pc, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        elif first_pc.min() < 0:
            # All-negative first component: flip sign so the map isn't all-zero.
            heatmap = np.maximum(-first_pc, 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

        return heatmap, pred_class.item(), confidence.item()

    def remove_hooks(self):
        self._fwd_hook.remove()


# ── Score-CAM ────────────────────────────────────────────────────────────────
class ScoreCAM:
    """
    Score-CAM (Wang et al., 2020): gradient-free CAM that scores each
    activation-map channel by its effect on the target class's softmax
    score when used as an input mask, then takes a weighted sum of
    channels. More faithful than gradient-based methods (avoids gradient
    saturation/noise) but far more expensive: one extra forward pass per
    channel (up to 2048 for ResNet-50's layer4).

    To keep runtime tractable, only the `max_channels` channels with the
    largest activation energy are scored (a standard approximation used
    in the original paper's "fast" variant); the rest are treated as
    contributing zero weight. Forward passes are batched.

    Reference: https://arxiv.org/abs/1910.01279
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module,
                 max_channels: int = 64, batch_size: int = 16):
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer
        self.max_channels = max_channels
        self.batch_size = batch_size
        self.activations = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates a Score-CAM heatmap for input_tensor.

        Args:
            input_tensor: shape (1, 3, H, W)
            target_class: class index to explain. If None, uses predicted class.

        Returns:
            heatmap (np.ndarray): shape (H, W), values in [0, 1]
            predicted_class (int)
            confidence (float)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        _, _, H, W = input_tensor.shape

        with torch.no_grad():
            output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)
        if target_class is None:
            target_class = pred_class.item()

        acts = self.activations[0]  # (C, h, w)
        c = acts.shape[0]

        # Restrict to the top-energy channels to keep the number of extra
        # forward passes tractable (see class docstring).
        energy = acts.abs().sum(dim=(1, 2))
        k = min(self.max_channels, c)
        top_idx = torch.topk(energy, k).indices

        # Upsample each selected channel to input resolution and normalize
        # to [0, 1] so it can be used as a multiplicative mask.
        selected = acts[top_idx].unsqueeze(1)  # (k, 1, h, w)
        masks = F.interpolate(selected, size=(H, W), mode="bilinear", align_corners=False)
        masks = masks.squeeze(1)  # (k, H, W)
        m_min = masks.amin(dim=(1, 2), keepdim=True)
        m_max = masks.amax(dim=(1, 2), keepdim=True)
        denom = (m_max - m_min).clamp_min(1e-8)
        masks = (masks - m_min) / denom

        weights = torch.zeros(k, device=self.device)
        with torch.no_grad():
            for start in range(0, k, self.batch_size):
                batch_masks = masks[start:start + self.batch_size].unsqueeze(1)  # (b,1,H,W)
                masked_inputs = input_tensor * batch_masks  # broadcast over channels
                batch_out = self.model(masked_inputs)
                batch_probs = F.softmax(batch_out, dim=1)[:, target_class]
                weights[start:start + self.batch_size] = batch_probs

        weights = F.softmax(weights, dim=0)  # normalize channel importance

        activation_map = acts[top_idx] * weights.view(-1, 1, 1)
        heatmap = F.relu(activation_map.sum(dim=0)).cpu().numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, pred_class.item(), confidence.item()

    def remove_hooks(self):
        self._fwd_hook.remove()


# ── Cosine-Overlap Metric ──────────────────────────────
def cosine_overlap(heatmap_a, heatmap_b):
    """
    Cosine-overlap score O_k^c between two flattened heatmaps (Section II-D):
        O = <a, b> / (||a|| * ||b||)

    Used to quantify agreement between a client's local-model heatmap and
    the global federated model's heatmap for the same image/class.
    """
    a = heatmap_a.flatten().astype(np.float64)
    b = heatmap_b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# â”€â”€ Visualization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on the original image.

    Returns:
        overlaid (np.ndarray): RGB image with heatmap overlay
    """
    img_array = np.array(original_image.resize((224, 224)))

    # Resize heatmap (7x7 feature map) â†’ image size (224x224)
    heatmap_up = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (img_array.shape[1], img_array.shape[0]), resample=Image.BILINEAR
    )
    heatmap_up = np.array(heatmap_up) / 255.0
    heatmap_colored = cm.jet(heatmap_up)[:, :, :3]  # RGBA â†’ RGB, shape (224,224,3)
    heatmap_colored = np.uint8(255 * heatmap_colored)

    overlaid = (alpha * img_array + (1 - alpha) * heatmap_colored).astype(np.uint8)
    return overlaid


def save_gradcam_figure(
    original_image: Image.Image,
    heatmap: np.ndarray,
    pred_class: int,
    confidence: float,
    true_class: int,
    save_path: str,
    title_prefix: str = "",
    heatmap_label: str = "Grad-CAM Heatmap",
):
    """Saves a 3-panel figure: original | heatmap | overlay."""
    overlay = overlay_heatmap(original_image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_image.resize((224, 224)))
    axes[0].set_title("Original Ultrasound")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(heatmap_label)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    pred_name = IDX_TO_CLASS.get(pred_class, str(pred_class))
    true_name = IDX_TO_CLASS.get(true_class, str(true_class))
    axes[2].set_title(
        f"{title_prefix}Pred: {pred_name} ({confidence:.2%})\nTrue: {true_name}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GradCAM] Saved â†’ {save_path}")


# â”€â”€ Run Grad-CAM on a batch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_gradcam_on_samples(
    model: torch.nn.Module,
    dataset,
    num_samples: int = 5,
    model_tag: str = "global",
    output_subdir: str = "",
):
    """
    Runs Grad-CAM on `num_samples` images from `dataset` and saves figures.

    Args:
        model:       Trained PyTorch model (local or global)
        dataset:     PyTorch dataset (each item: (tensor, label))
        num_samples: How many images to explain
        model_tag:   Label for the output filenames ("global", "client_0", etc.)
        output_subdir: Subfolder inside OUTPUT_DIR
    """
    model = model.to(DEVICE)
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    # Pre-process transform for original image display
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
    ])

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    out_dir = os.path.join(OUTPUT_DIR, output_subdir)

    for i, idx in enumerate(indices):
        tensor, true_label = dataset[idx]
        input_tensor = tensor.unsqueeze(0)  # add batch dim

        heatmap, pred_class, confidence = gradcam.generate(input_tensor)

        # Reconstruct original image for display
        original_tensor = inv_normalize(tensor).permute(1, 2, 0).numpy()
        original_tensor = np.clip(original_tensor, 0, 1)
        original_image = Image.fromarray((original_tensor * 255).astype(np.uint8))

        save_path = os.path.join(out_dir, f"{model_tag}_sample_{i:03d}.png")
        save_gradcam_figure(
            original_image=original_image,
            heatmap=heatmap,
            pred_class=pred_class,
            confidence=confidence,
            true_class=true_label,
            save_path=save_path,
            title_prefix=f"[{model_tag}] ",
        )

    gradcam.remove_hooks()
    print(f"[GradCAM] Done. {len(indices)} figures saved to {out_dir}")


# â”€â”€ Compare local vs global explanations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compare_local_vs_global(
    local_model: torch.nn.Module,
    global_model: torch.nn.Module,
    dataset,
    client_id: int,
    num_samples: int = 3,
):
    """
    Side-by-side Grad-CAM comparison: local model vs global federated model.
    This is the key figure for your paper â€” shows what each model attends to.
    """
    local_model = local_model.to(DEVICE)
    global_model = global_model.to(DEVICE)

    local_target = get_target_layer(local_model)
    global_target = get_target_layer(global_model)

    gcam_local = GradCAM(local_model, local_target)
    gcam_global = GradCAM(global_model, global_target)

    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
    ])

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    out_dir = os.path.join(OUTPUT_DIR, f"comparison_client_{client_id}")
    os.makedirs(out_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        tensor, true_label = dataset[idx]
        input_tensor = tensor.unsqueeze(0)

        heatmap_local, pred_local, conf_local = gcam_local.generate(input_tensor)
        heatmap_global, pred_global, conf_global = gcam_global.generate(input_tensor)

        original_tensor = inv_normalize(tensor).permute(1, 2, 0).numpy()
        original_tensor = np.clip(original_tensor, 0, 1)
        original_image = Image.fromarray((original_tensor * 255).astype(np.uint8))

        overlay_local = overlay_heatmap(original_image, heatmap_local)
        overlay_global = overlay_heatmap(original_image, heatmap_global)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(original_image.resize((224, 224)))
        axes[0].set_title(f"Original | True: {IDX_TO_CLASS[true_label]}")
        axes[0].axis("off")

        axes[1].imshow(overlay_local)
        axes[1].set_title(
            f"Local Model (Client {client_id})\n"
            f"Pred: {IDX_TO_CLASS[pred_local]} ({conf_local:.2%})"
        )
        axes[1].axis("off")

        axes[2].imshow(overlay_global)
        axes[2].set_title(
            f"Global Federated Model\n"
            f"Pred: {IDX_TO_CLASS[pred_global]} ({conf_global:.2%})"
        )
        axes[2].axis("off")

        plt.suptitle(
            f"Grad-CAM: Local vs Global â€” Client {client_id}, Sample {i}",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"compare_sample_{i:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[GradCAM Compare] Saved â†’ {save_path}")

    gcam_local.remove_hooks()
    gcam_global.remove_hooks()


# â”€â”€ Per-Class Grad-CAM (guarantees coverage across all 5 disease classes) â”€â”€â”€â”€â”€
def run_gradcam_per_class(
    model: torch.nn.Module,
    dataset,
    samples_per_class: int = 6,
    model_tag: str = "global",
    output_subdir: str = "global_per_class",
) -> int:
    """
    Generates Grad-CAM figures ensuring at least `samples_per_class`
    examples per disease class. For 5 classes Ã— 6 samples = 30 figures.

    Filenames: <model_tag>_<classname>_<n>.png
    """
    from collections import defaultdict

    CLASS_LONG = {
        0: "normal",
        1: "benign_mural_thickening",
        2: "stones",
        3: "abnormal_benign",
        4: "malignant",
    }

    model = model.to(DEVICE)
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
    ])

    # Group dataset indices by class label
    class_indices: dict = defaultdict(list)
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        class_indices[int(lbl)].append(idx)

    out_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    total = 0

    for cls_idx in sorted(class_indices.keys()):
        indices = class_indices[cls_idx]
        n = min(samples_per_class, len(indices))
        chosen = np.random.choice(indices, n, replace=False)
        cls_name = CLASS_LONG.get(cls_idx, f"class_{cls_idx}")

        for sample_num, idx in enumerate(chosen, start=1):
            tensor, true_label = dataset[idx]
            input_tensor = tensor.unsqueeze(0)

            heatmap, pred_class, confidence = gradcam.generate(input_tensor)

            orig = inv_normalize(tensor).permute(1, 2, 0).numpy()
            orig = np.clip(orig, 0, 1)
            original_image = Image.fromarray((orig * 255).astype(np.uint8))

            save_path = os.path.join(
                out_dir, f"{model_tag}_{cls_name}_{sample_num:02d}.png"
            )
            save_gradcam_figure(
                original_image=original_image,
                heatmap=heatmap,
                pred_class=pred_class,
                confidence=confidence,
                true_class=true_label,
                save_path=save_path,
                title_prefix=f"[{model_tag}] ",
            )
            total += 1

    gradcam.remove_hooks()
    print(f"[GradCAM] Per-class complete: {total} figures saved â†’ {out_dir}")
    return total


# â”€â”€ Client-level Grad-CAM (runs per-class on each client's local model) â”€â”€â”€â”€â”€â”€â”€
def run_gradcam_client_local(
    client_ckpt_path: str,
    dataset,
    client_id: int,
    samples_per_class: int = 2,
) -> int:
    """
    Loads a client checkpoint and runs per-class Grad-CAM.
    Saves results under outputs/gradcam/client_<id>_per_class/
    Returns number of figures generated.
    """
    sys_modules_backup = None
    try:
        import importlib
        import sys as _sys
        # build_model is available via import from same package
        from model import build_model
    except ImportError:
        print(f"[GradCAM Client] Cannot import model. Skipping client {client_id}.")
        return 0

    if not os.path.exists(client_ckpt_path):
        print(f"[GradCAM Client] Checkpoint not found: {client_ckpt_path}. Skipping.")
        return 0

    model = build_model(pretrained=False).to(DEVICE)
    state = torch.load(client_ckpt_path, map_location=DEVICE)
    # Handle both raw state_dict and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    print(f"[GradCAM Client {client_id}] Loaded checkpoint: {os.path.basename(client_ckpt_path)}")
    n = run_gradcam_per_class(
        model=model,
        dataset=dataset,
        samples_per_class=samples_per_class,
        model_tag=f"client_{client_id}",
        output_subdir=f"client_{client_id}_per_class",
    )
    return n

