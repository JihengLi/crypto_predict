# nohup environs/python3.10/bin/python3.10 tensor_metric_infonce_training/train.py > tensor_metric_infonce_training/out.log 2>&1 &

import os, math, re
import torch

from datasets import *
from models import *
from utils import *
from losses import *
from configs import *

EPOCH_NUM = 100

if __name__ == "__main__":
    features_path = "data/features/processed_features.npy"
    labels_path = "data/features/labels.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Mamba2Config(
        features_path=features_path,
        labels_path=labels_path,
        epochs=EPOCH_NUM,
        device=device,
    )
    train_loader = cfg.train_loader
    val_loader = cfg.val_loader
    model = cfg.model
    optimizer = cfg.optimizer
    scheduler = cfg.scheduler
    scaler = cfg.scaler

    start_epoch = 1
    best_val_loss = math.inf
    ckpt_dir = "outputs/checkpoints"

    latest_epoch = -1
    pat = re.compile(r"epoch(\d+)\.pth$")
    resume_path = None

    for fname in os.listdir(ckpt_dir):
        m = pat.match(fname)
        if m:
            ep = int(m.group(1))
            if ep > latest_epoch:
                latest_epoch = ep
                resume_path = os.path.join(ckpt_dir, fname)

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]

        print(
            f"Auto-resumed from {resume_path} — "
            f"epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.4f})"
        )
    else:
        print("No checkpoint found — starting fresh training.")

    train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        epochs=EPOCH_NUM,
        ckpt_dir=ckpt_dir,
    )
