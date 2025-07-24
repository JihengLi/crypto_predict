import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from models import *
from datasets import *


class Mamba2Config:
    def __init__(
        self,
        features_path,
        labels_path,
        epochs,
        device,
    ):
        self.features_path = features_path
        self.labels_path = labels_path
        self.device = device
        self.epochs = epochs

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler_cosine_decay()
        self.scaler = self._build_scaler()

    def _build_dataloaders(self):
        train_dataset = WindowDataset(
            features_path=self.features_path,
            labels_path=self.labels_path,
            split="train",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
        )

        val_dataset = WindowDataset(
            features_path=self.features_path,
            labels_path=self.labels_path,
            split="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _build_model(self):
        model = Mamba2Multitask(dropout=0.3, drop_path_prob=0.3, enable_mhsa=False).to(
            self.device
        )
        return model

    def _build_optimizer(self):
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(
                keyword in name.lower()
                for keyword in ("bias", "bn", "ln", "layernorm", "log_var")
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 3e-2},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _build_scheduler_cosine_decay(self):
        warmup_pct = 0.10
        start_factor = 1e-2
        min_lr_ratio = 1e-4

        steps_per_epoch = len(self.train_loader)
        total_steps = self.epochs * steps_per_epoch
        warm_steps = max(1, int(total_steps * warmup_pct))

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warm_steps,
        )
        base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        eta_min = min(lr * min_lr_ratio for lr in base_lrs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warm_steps,
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warm_steps],
            last_epoch=-1,
        )

    def _build_scaler(self):
        return GradScaler(
            init_scale=2**14,
            growth_interval=2000,
        )
