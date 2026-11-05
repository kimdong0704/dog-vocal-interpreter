#!/bin/bash
# Fine-tune YOLO11m-pose on StanfordExtra dog keypoints (team train/val splits; test split untouched).
# Submit:  sbatch /work/cssema416/202710/05/jobs/train_pose_encoder.sbatch
# Watch:   squeue -u $USER      tail -f /work/cssema416/202710/05/jobs/logs/pose-encoder-<jobid>.out
# Cancel:  scancel <jobid>   (rerun with RESUME=1 sbatch ... to continue from last.pt)
#SBATCH --job-name=pose-encoder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/work/cssema416/202710/05/jobs/logs/%x-%j.out

set -euo pipefail
REPO=/work/cssema416/202710/05/dog-vocal-interpreter
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python - <<'EOF'
import os
from ultralytics import YOLO

REPO = "/work/cssema416/202710/05/dog-vocal-interpreter"
RUN_DIR = f"{REPO}/models/pose"
NAME = "yolo11m-pose_full"

if os.environ.get("RESUME"):
    YOLO(f"{RUN_DIR}/{NAME}/weights/last.pt").train(resume=True)
else:
    YOLO("yolo11m-pose.pt").train(
        data=f"{REPO}/data/processed/yolo_dog_pose/dog_pose_team.yaml",
        epochs=150,
        patience=30,  # stop once val fitness hasn't improved for 30 epochs
        time=5.6,  # hours; ends training cleanly inside the 6 h job limit
        imgsz=640,
        batch=32,
        cos_lr=True,
        close_mosaic=15,
        workers=10,
        seed=0,
        device=0,
        project=RUN_DIR,
        name=NAME,
        exist_ok=False,
        plots=True,
    )

best = YOLO(f"{RUN_DIR}/{NAME}/weights/best.pt")
m = best.val(data=f"{REPO}/data/processed/yolo_dog_pose/dog_pose_team.yaml", split="val",
             project=RUN_DIR, name=f"{NAME}_val", exist_ok=True)
print(f"best.pt  pose mAP50-95 {m.pose.map:.4f}  pose mAP50 {m.pose.map50:.4f}  box mAP50-95 {m.box.map:.4f}")
EOF
