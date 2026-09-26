"""
compute_checksums.py
----------------------
Phase 11: SHA-256 checksums for every checkpoint and every frozen results
file, so reviewers/readers can verify a downloaded artifact matches what
the manuscript reports.

Run from fedgb/:  python compute_checksums.py
Outputs: outputs/frozen_results/checksums.txt
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [
    "outputs/seeds",
    "outputs/session_corrected",
    "outputs/session_corrected_densenet121",
    "outputs/frozen_results/results_master.csv",
    "outputs/phase0",
    "outputs/leakage",
]
EXTENSIONS = (".pt", ".json", ".csv")


def sha256_file(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for target in TARGETS:
        full = os.path.join(HERE, target)
        if os.path.isfile(full):
            paths = [full]
        elif os.path.isdir(full):
            paths = []
            for root, _, files in os.walk(full):
                for fn in files:
                    if fn.endswith(EXTENSIONS):
                        paths.append(os.path.join(root, fn))
        else:
            print(f"[checksums] skipping missing path: {target}")
            continue

        for p in sorted(paths):
            rel = os.path.relpath(p, HERE).replace(os.sep, "/")
            digest = sha256_file(p)
            rows.append((digest, rel))
            print(f"[checksums] {digest}  {rel}")

    out_dir = os.path.join(HERE, "outputs", "frozen_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "checksums.txt")
    with open(out_path, "w") as f:
        for digest, rel in rows:
            f.write(f"{digest}  {rel}\n")
    print(f"\n[checksums] wrote {out_path} ({len(rows)} files)")


if __name__ == "__main__":
    main()
