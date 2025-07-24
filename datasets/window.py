import numpy as np
import torch

from pathlib import Path
from typing import Union
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(
        self,
        features_path: Union[str, Path],
        labels_path: Union[str, Path],
        split: str = "train",
        val_ratio: float = 0.2,
        history_steps: int = 8640,
        stride: int = 96,
    ):
        assert split in ("train", "val", "test"), f"invalid split={split}"
        self.split = split
        self.history = history_steps
        self.stride = stride

        X = np.load(features_path)
        y = np.load(labels_path)
        Nf, Ny = X.shape[0], y.shape[0]

        ends_all = np.arange(self.history - 1, Nf, self.stride, dtype=np.int64)
        ends_ov = ends_all[ends_all < Ny]
        ends_test = ends_all[ends_all >= Ny]

        split_idx = int(len(ends_ov) * (1 - val_ratio))
        if split == "train":
            self.ends = ends_ov[:split_idx]
        elif split == "val":
            self.ends = ends_ov[split_idx:]
        else:
            self.ends = ends_test

        self.X = X
        self.y = y
        self.has_label = split in ("train", "val")

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, idx):
        end = int(self.ends[idx])
        start = end - self.history + 1
        x_window = self.X[start : end + 1]
        inputs = torch.from_numpy(x_window).float()

        if not self.has_label:
            return {"inputs": inputs}

        cls, p90, p10, sigma = self.y[end]
        return {
            "inputs": inputs,
            "cls": torch.tensor(int(cls), dtype=torch.long),
            "p90": torch.tensor(float(p90), dtype=torch.float32),
            "p10": torch.tensor(float(p10), dtype=torch.float32),
            "sigma": torch.tensor(float(sigma), dtype=torch.float32),
        }
