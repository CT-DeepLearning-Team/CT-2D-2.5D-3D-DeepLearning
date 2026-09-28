"""Manifest loading, leakage checks and the 2D slice datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common import config as C
from src.common import cache as cache_mod
from src.common.transforms import augment_moco, augment_supervised, window_hu


# ------------------------------------------------------------------ manifests
def load_splits() -> dict[str, pd.DataFrame]:
    out = {}
    for name, path in C.SPLIT_CSV.items():
        df = pd.read_csv(path, low_memory=False)
        df[C.LABEL_COL] = df[C.LABEL_COL].astype(int)
        out[name] = df
    return out


def load_ssl_pool() -> pd.DataFrame:
    return pd.read_csv(C.SSL_CSV, low_memory=False)


def assert_no_leakage(splits: dict[str, pd.DataFrame], ssl: pd.DataFrame | None = None):
    """Hard guard: the protocol's central promise, re-checked on every run.

    Raises rather than warns -- a silent leak invalidates every number we report.
    """
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = set(splits[a][C.PATIENT_COL]), set(splits[b][C.PATIENT_COL])
            if pa & pb:
                raise AssertionError(f"patient leakage {a}/{b}: {len(pa & pb)}")
            na, nb = set(splits[a][C.ID_COL]), set(splits[b][C.ID_COL])
            if na & nb:
                raise AssertionError(f"nodule leakage {a}/{b}: {len(na & nb)}")
    if ssl is not None:
        sp = set(ssl[C.PATIENT_COL])
        for held in ("val", "test"):
            bad = sp & set(splits[held][C.PATIENT_COL])
            if bad:
                raise AssertionError(
                    f"SSL pool contains {len(bad)} {held} patients -- "
                    "pretraining would contaminate the held-out evaluation"
                )
    return True


def class_weights(train: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights, N / (K * n_k). Counters the 46% majority
    'indeterminate' class so the model cannot coast by predicting it."""
    counts = train[C.LABEL_COL].value_counts().reindex(range(C.NUM_CLASSES)).values
    w = counts.sum() / (C.NUM_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32)


# ------------------------------------------------------------------- datasets
class CentralSliceDataset(Dataset):
    """Single central CT slice -> 3-class label. This is the 2D input.

    Returns x of shape (1, 72, 80), windowed to [0, 1] then standardised with
    train-set statistics.
    """

    def __init__(self, df: pd.DataFrame, mean: float, std: float,
                 train: bool = False, seed: int = 0, aug: str = "mild"):
        arr, index = cache_mod.load()
        self.arr = arr
        self.rows = np.array([index[i] for i in df[C.ID_COL]], dtype=np.int64)
        self.labels = df[C.LABEL_COL].to_numpy(dtype=np.int64)
        self.ids = df[C.ID_COL].tolist()
        self.mean, self.std, self.train = mean, std, train
        self.aug = aug
        self._seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Augmentation must differ every epoch; the RNG seed includes the epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x = window_hu(np.asarray(self.arr[self.rows[i]]))
        x = torch.from_numpy(x).unsqueeze(0)             # (1, H, W)
        if self.train:
            rng = np.random.default_rng((self._seed, self.epoch, i))
            x = augment_supervised(x, rng, strength=self.aug)
        x = (x - self.mean) / self.std
        return x, int(self.labels[i])


class MoCoPairDataset(Dataset):
    """Two independently augmented views of the same central slice, no labels."""

    def __init__(self, df: pd.DataFrame, mean: float, std: float, seed: int = 0):
        arr, index = cache_mod.load()
        self.arr = arr
        self.rows = np.array(
            [index[i] for i in df[C.ID_COL] if i in index], dtype=np.int64
        )
        self.mean, self.std = mean, std
        self._seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        base = window_hu(np.asarray(self.arr[self.rows[i]]))
        base = torch.from_numpy(base).unsqueeze(0)
        rng = np.random.default_rng((self._seed, self.epoch, i))
        q = (augment_moco(base, rng) - self.mean) / self.std
        k = (augment_moco(base, rng) - self.mean) / self.std
        return q, k


def train_statistics(train: pd.DataFrame) -> tuple[float, float]:
    """Mean/std of windowed intensities over the TRAINING split only."""
    arr, index = cache_mod.load()
    rows = np.array([index[i] for i in train[C.ID_COL]], dtype=np.int64)
    vals = window_hu(np.asarray(arr[np.sort(rows)]))
    return float(vals.mean()), float(vals.std())
