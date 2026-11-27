"""Same candidate trainer/objective with deterministic GPU-resident batching.

The data cache is small enough for this 8 GB GPU. This changes data delivery,
not features, labels, masks, losses or model selection. Falls back to CPU
DataLoader when insufficient GPU memory is available.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tuning_common import ROOT, BASE, read_json, save_json
import argparse
import traceback
import numpy as np
import torch
import tune_train

OriginalLoader = tune_train.DataLoader
CACHE = {}


class DeviceLoader:
    def __init__(self, ds, batch_size, shuffle=False, sampler=None, generator=None, **kwargs):
        self.ds, self.batch_size = ds, batch_size
        self.shuffle, self.sampler, self.generator = shuffle, sampler, generator
        self.key = str(ds.x.filename)
        self.fallback = None
        if not torch.cuda.is_available():
            self.fallback = OriginalLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, generator=generator, **kwargs)
            return
        if self.key not in CACHE:
            need = ds.x.nbytes + ds.mask.nbytes
            free, _ = torch.cuda.mem_get_info()
            if free < need + 1200*1024**2:
                self.fallback = OriginalLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, generator=generator, **kwargs)
                return
            # Explicit copies avoid writable-array warnings and leave disk arrays intact.
            x = torch.from_numpy(np.array(ds.x)).to("cuda")
            mask = torch.from_numpy(np.array(ds.mask, dtype=bool)).to("cuda")
            CACHE[self.key] = (x, mask)
            print(f"GPU cache loaded: {self.key} {need/1024**2:.1f} MiB", flush=True)

    def __len__(self):
        return (len(self.ds)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        if self.fallback is not None:
            yield from self.fallback
            return
        if self.sampler is not None:
            order = np.fromiter(iter(self.sampler), dtype=np.int64)
        elif self.shuffle:
            order = torch.randperm(len(self.ds), generator=self.generator).numpy()
        else:
            order = np.arange(len(self.ds))
        x, mask = CACHE[self.key]
        for offset in range(0, len(order), self.batch_size):
            ids = self.ds.ids[order[offset:offset+self.batch_size]]
            dev_ids = torch.as_tensor(ids, device="cuda")
            yield x.index_select(0, dev_ids), mask.index_select(0, dev_ids), torch.as_tensor(self.ds.index[ids, 2].astype(np.int64)), torch.as_tensor(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    tune_train.DataLoader = DeviceLoader
    try:
        tune_train.train_candidate(args.name, read_json(ROOT/"configs"/args.config))
    except BaseException:
        save_json(ROOT/"runs"/args.name/"failure.json", {"traceback": traceback.format_exc()})
        raise
