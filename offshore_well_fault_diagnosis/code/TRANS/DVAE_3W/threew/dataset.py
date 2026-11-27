from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore
        pass


FEATURE_COLUMNS = [
    "ABER-CKGL",
    "ABER-CKP",
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "P-ANULAR",
    "P-JUS-BS",
    "P-JUS-CKGL",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-MON-CKP",
    "P-MON-SDV-P",
    "P-PDG",
    "PT-P",
    "P-TPT",
    "QBS",
    "QGL",
    "T-JUS-CKP",
    "T-MON-CKP",
    "T-PDG",
    "T-TPT",
]

CLASS_NAMES = {
    0: "Normal",
    1: "Abrupt increase of BSW",
    2: "Spurious closure of DHSV",
    3: "Severe slugging",
    4: "Flow instability",
    5: "Rapid productivity loss",
    6: "Quick restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in production line",
    9: "Hydrate in service line",
}


def normalize_3w_class(values: np.ndarray) -> np.ndarray:
    labels = pd.Series(values).astype("float").to_numpy()
    labels = np.where(np.isnan(labels), -1, labels).astype(np.int64)
    transient = (labels >= 101) & (labels <= 109)
    labels[transient] -= 100
    return labels


def list_parquet_files(root: str | Path, classes: Iterable[int] | None = None) -> list[Path]:
    root = Path(root)
    selected = set(classes) if classes is not None else None
    paths: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.isdigit():
            continue
        cls = int(child.name)
        if selected is not None and cls not in selected:
            continue
        paths.extend(sorted(child.glob("*.parquet")))
    return paths


def stratified_split_by_instance(
    paths: list[Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_files_per_class: int | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    by_class: dict[int, list[Path]] = {}
    for path in paths:
        by_class.setdefault(int(path.parent.name), []).append(path)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []
    for cls, cls_paths in sorted(by_class.items()):
        cls_paths = list(cls_paths)
        rng.shuffle(cls_paths)
        if max_files_per_class is not None:
            cls_paths = cls_paths[:max_files_per_class]

        n = len(cls_paths)
        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        test.extend(cls_paths[:n_test])
        val.extend(cls_paths[n_test : n_test + n_val])
        train.extend(cls_paths[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def compute_normalization_stats(
    paths: list[Path],
    feature_columns: list[str] = FEATURE_COLUMNS,
    max_files: int | None = 200,
    max_rows_per_file: int = 5000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = random.Random(seed)
    selected = list(paths)
    rng.shuffle(selected)
    if max_files is not None:
        selected = selected[:max_files]

    sums = np.zeros(len(feature_columns), dtype=np.float64)
    sq_sums = np.zeros(len(feature_columns), dtype=np.float64)
    counts = np.zeros(len(feature_columns), dtype=np.float64)

    for path in selected:
        df = pd.read_parquet(path, columns=feature_columns)
        if len(df) > max_rows_per_file:
            df = df.sample(n=max_rows_per_file, random_state=seed)
        arr = df.to_numpy(dtype=np.float64)
        mask = np.isfinite(arr)
        safe = np.where(mask, arr, 0.0)
        sums += safe.sum(axis=0)
        sq_sums += (safe * safe).sum(axis=0)
        counts += mask.sum(axis=0)

    counts = np.maximum(counts, 1.0)
    mean = sums / counts
    var = np.maximum(sq_sums / counts - mean * mean, 1e-8)
    std = np.sqrt(var)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class ThreeWWindowDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        window_size: int,
        stride: int,
        stats: dict[str, np.ndarray],
        task: str = "multiclass",
        feature_columns: list[str] = FEATURE_COLUMNS,
        max_windows_per_file: int | None = 16,
        seed: int = 42,
        cache_size: int = 8,
        clip_norm: float | None = None,
        return_index: bool = False,
    ):
        if task not in {"binary", "multiclass"}:
            raise ValueError("task must be 'binary' or 'multiclass'")
        self.paths = list(paths)
        self.window_size = window_size
        self.stride = stride
        self.stats = stats
        self.task = task
        self.feature_columns = feature_columns
        self.cache_size = cache_size
        self.clip_norm = clip_norm
        self.return_index = return_index
        self._cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self.index = self._build_index(max_windows_per_file, seed)

    def _build_index(self, max_windows_per_file: int | None, seed: int) -> list[tuple[Path, int, int]]:
        rng = random.Random(seed)
        items: list[tuple[Path, int, int]] = []
        for path in self.paths:
            labels_raw = pd.read_parquet(path, columns=["class"])["class"].to_numpy()
            labels = normalize_3w_class(labels_raw)
            if len(labels) < self.window_size:
                continue

            starts = list(range(0, len(labels) - self.window_size + 1, self.stride))
            if max_windows_per_file is not None and len(starts) > max_windows_per_file:
                starts = sorted(rng.sample(starts, max_windows_per_file))

            for start in starts:
                window_labels = labels[start : start + self.window_size]
                valid = window_labels[window_labels >= 0]
                if len(valid) == 0:
                    continue
                counts = np.bincount(valid, minlength=10)
                label = int(np.argmax(counts[:10]))
                if self.task == "binary":
                    label = int(label != 0)
                items.append((path, start, label))
        return items

    def _read_frame(self, path: Path) -> pd.DataFrame:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        df = pd.read_parquet(path, columns=self.feature_columns)
        df = df.ffill().bfill()
        self._cache[path] = df
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return df

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required to create training tensors.")
        path, start, label = self.index[idx]
        df = self._read_frame(path)
        arr = df.iloc[start : start + self.window_size].to_numpy(dtype=np.float64)
        arr = (arr - self.stats["mean"]) / self.stats["std"]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_norm is not None and self.clip_norm > 0:
            arr = np.clip(arr, -self.clip_norm, self.clip_norm)
        arr = arr.astype(np.float32, copy=False)
        x = torch.from_numpy(arr)
        y = torch.tensor(label, dtype=torch.long)
        if self.return_index:
            return x, y, torch.tensor(idx, dtype=torch.long)
        return x, y
