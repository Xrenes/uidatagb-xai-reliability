"""
model.py
--------
ResNet-50 based CNN for 9-class gallbladder disease classification
(UIdataGB). See dataset.py for the class list and labels.

Uses transfer learning from ImageNet pretrained weights.
"""

import torch
import torch.nn as nn
from torchvision import models

from dataset import NUM_CLASSES  # 9 for UIdataGB; kept in sync with dataset.py


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Builds a ResNet-50 model with a custom classifier head.

    Args:
        pretrained:       Load ImageNet weights (recommended).
        freeze_backbone:  If True, only the final FC layer is trained.
                          Useful for very small datasets.

    Returns:
        model (nn.Module)
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer for 5-class output
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES),
    )

    return model


def build_model_densenet121(pretrained: bool = True) -> nn.Module:
    """
    Second-backbone comparison model (Phase 3.3 of the revision plan):
    DenseNet-121 with the same classifier-head shape as build_model, so the
    Stage-2 leakage/generalization conclusion can be checked for survival
    under a different architecture rather than resting on ResNet-50 alone.
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def get_model_parameters(model: nn.Module):
    """Returns model parameters as a list of numpy arrays (for Flower FL)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters):
    """Loads a list of numpy arrays into model state dict (for Flower FL)."""
    import numpy as np
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def get_target_layer(model: nn.Module):
    """
    Returns the last convolutional layer for Grad-CAM.
    For ResNet-50, this is layer4[-1].
    """
    return model.layer4[-1]
