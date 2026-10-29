#!/usr/bin/env bash
#SBATCH --job-name=diffuscene_fid_kid
#SBATCH --output=fid_kid_%j.out
#SBATCH --error=fid_kid_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

# Activate your DiffuScene Python environment before submitting this job.
# Override these with exported environment variables if needed.
export DIFFUSCENE_ROOT="${DIFFUSCENE_ROOT:-$HOME/DiffuScene}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

cd "$DIFFUSCENE_ROOT"

"$PYTHON_BIN" - <<'PY'
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid


root = Path(os.environ["DIFFUSCENE_ROOT"]).expanduser().resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workers = max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1)
rooms = (
    ("diningroom", "diningrooms_bert"),
    ("bedroom", "bedrooms_bert"),
    ("livingroom", "livingrooms_bert"),
)

# Validate every input before starting expensive feature extraction.
inputs = []
for room, model in rooms:
    real = root / "datasets" / "3d_front_processed" / f"{room}_image"
    fake = root / "models" / "pretrained_diffusion" / model / "final" / "rendered_resolved"
    counts = []
    for folder in (real, fake):
        if not folder.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {folder}")
        # CleanFID reads images directly inside each directory, not recursively.
        count = sum(
            path.is_file() and path.suffix in {".png", ".jpg", ".jpeg"}
            for path in folder.iterdir()
        )
        if count < 2:
            raise ValueError(f"Need at least 2 images directly inside {folder}; found {count}")
        counts.append(count)
    inputs.append((room, real, fake, *counts))

job_id = os.environ.get("SLURM_JOB_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
output = root / "evaluation_results" / f"fid_kid_{job_id}.json"
output.parent.mkdir(parents=True, exist_ok=True)
report = {
    "device": str(device),
    "mode": "clean",
    "reference_selection": "all images directly inside each supplied GT folder; no split filtering",
    "results": [],
}

print(f"Root: {root}\nDevice: {device}\nResults: {output}", flush=True)
for room, real, fake, n_real, n_fake in inputs:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # Adapt KID to reference folders containing fewer than 1000 images.
    subset_size = min(1000, n_real, n_fake)
    print(f"\n{room}: {n_real} real / {n_fake} generated images", flush=True)
    print(f"Real: {real}\nFake: {fake}\nKID subset size: {subset_size}", flush=True)
    options = dict(device=device, mode="clean", num_workers=workers)
    fid_score = float(fid.compute_fid(str(real), str(fake), **options))
    kid_score = float(fid.compute_kid(
        str(real), str(fake), num_subsets=100, max_subset_size=subset_size, **options
    ))
    report["results"].append({
        "room": room,
        "real_dir": str(real),
        "fake_dir": str(fake),
        "num_real": n_real,
        "num_fake": n_fake,
        "kid_subset_size": subset_size,
        "FID": fid_score,
        "KID": kid_score,
        "KID_x1000": kid_score * 1000,
    })
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{room}: FID={fid_score:.6f}  KID={kid_score:.8f}  KID*1000={kid_score * 1000:.6f}", flush=True)

print("\nRoom          Real    Fake          FID           KID      KID*1000", flush=True)
for result in report["results"]:
    print(
        f"{result['room']:<12} {result['num_real']:>5} {result['num_fake']:>7} "
        f"{result['FID']:>12.6f} {result['KID']:>13.8f} {result['KID_x1000']:>13.6f}",
        flush=True,
    )
print(f"\nSaved: {output}", flush=True)
PY
