# GPU run instructions — Eigen-CAM + Score-CAM

This file is self-contained: read it top to bottom and run the commands
in order. It adds two new XAI methods (Eigen-CAM, Score-CAM) to the
XAI-FedGB paper's cosine-overlap analysis, using this machine's GPU
(NVIDIA GeForce GTX 1050, 4GB).

Context: Grad-CAM, Grad-CAM++, and Saliency Map overlap results already
exist in `fedgb/outputs/overlap_all_methods.json` (computed on CPU).
Eigen-CAM and Score-CAM classes now exist in `fedgb/gradcam.py` but have
never been run. Score-CAM is expensive (one extra forward pass per
sampled activation channel per image) — on CPU it was measured at ~9s/image
even restricted to 64 channels, which would take ~7 hours for the full
689-image x 4-model sweep. On a GPU this should drop to roughly 1-2
hours total. Eigen-CAM is cheap either way (no backward pass, no extra
forward passes) — under 10 minutes even on CPU.

## Step 0 — Confirm the GPU is actually usable

```
nvidia-smi
```

If this fails with "command not found", NVIDIA drivers are either not
installed or not on PATH. Install/repair the driver first (GeForce
Experience or the driver installer from nvidia.com) before continuing —
do not proceed to Step 1 until this prints GPU info.

## Step 1 — Install a CUDA-enabled PyTorch

Check the existing environment first:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `+cpu` or `False`, the installed PyTorch build has no CUDA
support and must be replaced. Do NOT `pip install torch` plain — that
resolves to a CPU wheel on many systems. Use the CUDA-specific index
matching this GPU's driver version (check https://pytorch.org/get-started/locally/
if unsure; cu121 works for most current drivers on a 1050):

```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then re-verify:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output should show `True` and `NVIDIA GeForce GTX 1050`.

## Step 2 — Run Eigen-CAM (fast, do this first regardless of Score-CAM timing)

From `fedgb/`:

```
python compute_overlap_all_methods.py --methods eigencam --device cuda --merge
```

`--merge` preserves the existing Grad-CAM/Grad-CAM++/Saliency results in
`outputs/overlap_all_methods.json` instead of overwriting them with only
the new method.

## Step 3 — Run Score-CAM

```
python compute_overlap_all_methods.py --methods scorecam --device cuda --merge
```

This is the long-running one. Watch progress live at any time from a
second terminal:

```
python overlap_progress_server.py
```

Then open http://localhost:8766 in a browser, or poll directly:

```
curl -s http://localhost:8766/status
```

If it's taking far longer than ~2 hours, something is wrong (e.g. it
silently fell back to CPU) — check `torch.cuda.is_available()` again and
confirm `nvidia-smi` shows GPU utilization while the script runs.

## Step 4 — Verify and report back

Once both finish, `fedgb/outputs/overlap_all_methods.json` should contain
five top-level method keys: `gradcam`, `gradcam_pp`, `saliency`,
`eigencam`, `scorecam`. Sanity check:

```
python -c "import json; d = json.load(open('outputs/overlap_all_methods.json')); print(sorted(k for k in d if k != 'meta'))"
```

Expected: `['eigencam', 'gradcam', 'gradcam_pp', 'saliency', 'scorecam']`

At that point, tell Claude the run is done — it will extend Table VI in
`FedGB_Paper.html` from a 3-method to a 5-method comparison, update the
Conclusion/Abstract/Related-Work text (which currently lists Score-CAM
and Eigen-CAM as "future directions" — that framing needs to change to
reflect they're now implemented), and remove this instructions file.

## If something goes wrong

- **"CUDA out of memory"**: the 1050 only has 4GB. Lower Score-CAM's
  batch size: edit `compute_overlap_all_methods.py`'s `make_explainer()`
  function, the line `ScoreCAM(model, get_target_layer(model), max_channels=64, batch_size=32)`
  — reduce `batch_size` to 8 or 16.
- **Still too slow even on GPU**: reduce `max_channels` from 64 to 32 in
  the same line — halves Score-CAM's runtime at some cost to fidelity
  (fewer channels sampled). Note this in the paper's methodology if done.
- **Results look inconsistent with the CPU-computed Grad-CAM numbers**:
  they shouldn't be — Grad-CAM/Grad-CAM++/Saliency are deterministic
  given the same checkpoint and images, so re-running those (not just
  the new methods) on GPU should reproduce the exact same numbers
  already in the paper. If you want to double check, run
  `python compute_overlap_all_methods.py --methods gradcam --device cuda --out outputs/gradcam_gpu_check.json`
  and diff against the existing per-class means in Table VI.
