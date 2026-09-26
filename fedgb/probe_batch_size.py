"""
probe_batch_size.py
--------------------
Finds the largest batch size that fits in this GPU's VRAM (with a safety
margin), by actually running a forward+backward step at increasing sizes
and watching torch's reserved-memory counter. Run from fedgb/.
"""
import torch
import torch.nn as nn

from model import build_model

DEVICE = torch.device("cuda")
BUDGET_MIB = 3600  # stay under 4096 MiB total, leaving headroom for the
                    # OS/driver baseline (~500MB) and eval-time overhead

model = build_model(pretrained=True).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
torch.backends.cudnn.benchmark = True

best = None
for bs in [16, 24, 32, 40, 48, 56, 64, 80, 96]:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, 3, 224, 224, device=DEVICE)
        y = torch.randint(0, 5, (bs,), device=DEVICE)
        for _ in range(3):  # a few steps so cudnn.benchmark settles on its algo
            optim.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            optim.step()
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_reserved() / (1024 * 1024)
        print(f"bs={bs:3d}  peak_reserved={peak_mib:7.1f} MiB")
        if peak_mib <= BUDGET_MIB:
            best = bs
        else:
            print(f"  -> exceeds {BUDGET_MIB} MiB budget, stopping.")
            break
    except torch.cuda.OutOfMemoryError:
        print(f"bs={bs:3d}  OOM")
        break

print(f"\n[probe] largest safe batch size under {BUDGET_MIB} MiB: {best}")
