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
        history_steps: int = 8640,
        stride: int = 96,
    ):
        self.X = np.load(features_path)
        self.y = np.load(labels_path)
        if self.X.shape[0] > self.y.shape[0]:
            self.X = self.X[: self.y.shape[0]]
        assert (
            self.X.shape[0] == self.y.shape[0]
        ), f"features length {self.X.shape[0]} != labels length {self.y.shape[0]}"

        self.history = history_steps
        self.stride = stride

        T = self.X.shape[0]
        first_end = self.history - 1
        last_end = T - 1
        self.ends = np.arange(first_end, last_end + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, idx):
        end = self.ends[idx]
        start = end - self.history + 1
        x = self.X[start : end + 1]
        cls, p90, p10, sigma = self.y[end]

        return {
            "inputs": torch.from_numpy(x),
            "cls": torch.tensor(int(cls), dtype=torch.long),
            "p90": torch.tensor(float(p90), dtype=torch.float32),
            "p10": torch.tensor(float(p10), dtype=torch.float32),
            "sigma": torch.tensor(float(sigma), dtype=torch.float32),
        }
