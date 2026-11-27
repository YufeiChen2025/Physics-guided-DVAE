"""One validation-selected candidate. No test loading is implemented here."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import time
import traceback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tuning_common import BASE, CODE, ROOT, read_json, save_json, sha256
from tuned_model import TunedDiagnoser
from train import seed_everything, classification_metrics, atomic_torch_save, rng_state, restore_rng


class TuningDataset(Dataset):
    def __init__(self, part, normalization="standard", limit=0):
        if part not in ["train", "val"]:
            raise ValueError("Tuning runner is prohibited from loading test data")
        folder = BASE / "cache" / part
        self.index = np.load(folder / "index.npy")
        self.mask = np.load(folder / "mask.npy", mmap_mode="r")
        xdir = ROOT / "cache" / "robust" / part if normalization == "robust" else folder
        self.x = np.load(xdir / "x.npy", mmap_mode="r")
        self.ids = np.arange(len(self.index))
        if limit and limit < len(self.ids):
            rng = np.random.default_rng(20261028)
            ids = []
            for c in range(9):
                rows = np.flatnonzero(self.index[:, 2] == c)
                n = max(1, round(len(rows)*limit/len(self.ids)))
                ids.extend(rng.choice(rows, min(n, len(rows)), replace=False))
            self.ids = np.sort(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = self.ids[i]
        return torch.from_numpy(self.x[j].copy()), torch.from_numpy(self.mask[j].copy()).bool(), torch.tensor(int(self.index[j, 2]), dtype=torch.long), torch.tensor(j)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, labels, logits_all, ids_all = [], [], [], []
    for x, mask, y, ids in loader:
        logits = model.predict(x.to(device, non_blocking=True), mask.to(device, non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Invalid validation predictions")
        preds.append(logits.argmax(1).cpu().numpy()); labels.append(y.numpy())
        logits_all.append(logits.cpu().numpy()); ids_all.append(ids.numpy())
    y, p = np.concatenate(labels), np.concatenate(preds)
    return classification_metrics(y, p), np.concatenate(logits_all), y, np.concatenate(ids_all)


def load_checkpoint(spec):
    if spec.startswith("archive:"):
        return torch.load(BASE / "runs" / spec.split(":", 1)[1] / "best.pt", map_location="cpu")["model"]
    if spec.startswith("tuning:"):
        return torch.load(ROOT / "runs" / spec.split(":", 1)[1] / "best.pt", map_location="cpu")["model"]
    raise ValueError(spec)


def train_candidate(name, cfg):
    dest = ROOT / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(cfg))
    cfg["split_sha256"] = read_json(BASE / "configs" / "normalization.json")["split_sha256"]
    cfg["code_sha256"] = {p.name: sha256(p) for p in sorted((CODE / "src").glob("*.py"))}
    if (dest / "config.json").exists() and read_json(dest / "config.json") != cfg:
        raise ValueError("Frozen candidate configuration differs")
    save_json(dest / "config.json", cfg)
    if (dest / "result.json").exists():
        return read_json(dest / "result.json")
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    features = read_json(BASE / "configs" / "normalization.json")["features"]
    model = TunedDiagnoser(features, cfg)
    if cfg.get("init"):
        model.load_initial(load_checkpoint(cfg["init"]))
    ds = TuningDataset("train", cfg.get("normalization", "standard"), cfg.get("train_limit", 0))
    valds = TuningDataset("val", cfg.get("normalization", "standard"))
    counts = np.bincount(ds.index[:, 2], minlength=9).astype(float)
    weights = (counts.sum()/9/counts)**cfg.get("weight_power", 0.)
    weights = np.clip(weights, .1, 5)
    weights /= weights.mean()
    model.class_weights.copy_(torch.tensor(weights, dtype=torch.float32))
    if cfg.get("freeze_backbone"):
        for p in model.dvae.parameters():
            p.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    backbone, heads = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (backbone if n.startswith("dvae.") else heads).append(p)
    lr = cfg.get("lr", .0003)
    parameter_groups = []
    if backbone:
        parameter_groups.append({"params": backbone, "lr": lr*cfg.get("backbone_lr_ratio", 1.)})
    if heads:
        parameter_groups.append({"params": heads, "lr": lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=cfg.get("weight_decay", 1e-5))
    val_loader = DataLoader(valds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    history, best, best_epoch, stale, begin = [], -1., -1, 0, 0
    resumed_seconds = 0.
    if (dest / "last.pt").exists():
        last = torch.load(dest / "last.pt", map_location=device)
        model.load_state_dict(last["model"]); optimizer.load_state_dict(last["optimizer"])
        history, best, best_epoch, stale, begin, resumed_seconds = [last[k] for k in ["history", "best", "best_epoch", "stale", "next_epoch", "seconds"]]
        restore_rng(last["rng"])
    else:
        initial_metrics, initial_logits, y, ids = validate(model, val_loader, device)
        save_json(dest / "initial_validation.json", initial_metrics)
        # Include epoch 0 for fine-tuning, so continuation cannot silently replace
        # a better initialization with a worse epoch-1 model.
        if cfg.get("init"):
            best, best_epoch = initial_metrics["macro_f1"], 0
            atomic_torch_save(dest / "best.pt", {"model": model.state_dict(), "config": cfg, "epoch": 0, "metrics": initial_metrics})
            np.savez_compressed(dest / "validation_predictions.npz", logits=initial_logits, labels=y, window_ids=ids)
    start = time.monotonic()
    epochs = cfg.get("epochs", 8)
    for epoch in range(begin, epochs):
        stamp = time.monotonic()
        gen = torch.Generator().manual_seed(seed*10000+epoch)
        sampler = None
        if cfg.get("balanced_sampler"):
            yy = ds.index[ds.ids, 2]
            sampler = WeightedRandomSampler(torch.tensor(1/counts[yy], dtype=torch.double), len(ds), replacement=True, generator=gen)
        loader = DataLoader(ds, batch_size=cfg.get("batch_size", 128), shuffle=sampler is None, sampler=sampler, generator=gen, num_workers=0, pin_memory=True)
        model.train()
        if cfg.get("freeze_backbone"):
            model.dvae.eval()
        totals, seen, max_grad, clipped = {}, 0, 0., 0
        for step, (x, mask, y, _) in enumerate(loader):
            x, mask, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(x, mask, y, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {name} epoch{epoch} step{step}")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.), error_if_nonfinite=True))
            max_grad = max(max_grad, norm); clipped += norm > cfg.get("grad_clip", 5.)
            optimizer.step()
            seen += len(y)
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.)+float(v)*len(y)
            if step % 400 == 0:
                save_json(dest / "progress.json", {"status": "training", "name": name, "epoch": epoch+1, "max_epochs": epochs, "step": step, "steps": len(loader), "best_val_macro_f1": best})
        metrics, logits, y, ids = validate(model, val_loader, device)
        score = metrics["macro_f1"]
        if score > best:
            best, best_epoch, stale = score, epoch+1, 0
            atomic_torch_save(dest / "best.pt", {"model": model.state_dict(), "config": cfg, "epoch": epoch+1, "metrics": metrics})
            np.savez_compressed(dest / "validation_predictions.npz", logits=logits, labels=y, window_ids=ids)
        else:
            stale += 1
        row = {"epoch": epoch+1, "seconds": time.monotonic()-stamp, "train": {k: v/seen for k, v in totals.items()}, "maximum_gradient": max_grad, "clipped_step_fraction": clipped/len(loader), "validation": metrics}
        history.append(row)
        save_json(dest / "history.json", history)
        seconds = resumed_seconds+time.monotonic()-start
        atomic_torch_save(dest / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best, "best_epoch": best_epoch, "stale": stale, "next_epoch": epoch+1, "seconds": seconds, "rng": rng_state()})
        save_json(dest / "progress.json", {"status": "epoch_complete", "name": name, "epoch": epoch+1, "best_val_macro_f1": best, "latest_val_macro_f1": score})
        print(f"{name} epoch={epoch+1} seconds={row['seconds']:.1f} valF1={score:.6f} best={best:.6f} recall2={metrics['per_class']['2']['recall']:.3f} recall3={metrics['per_class']['3']['recall']:.3f} recall7={metrics['per_class']['7']['recall']:.3f}", flush=True)
        if cfg.get("patience", 4) and stale >= cfg.get("patience", 4):
            break
    checkpoint = torch.load(dest / "best.pt", map_location="cpu")
    result = {"name": name, "best_epoch": best_epoch, "best_val_macro_f1": best, "validation": checkpoint["metrics"], "epochs_completed": len(history), "training_seconds": resumed_seconds+time.monotonic()-start, "train_windows": len(ds), "validation_windows": len(valds), "test_evaluated": False, "config": cfg}
    save_json(dest / "result.json", result)
    save_json(dest / "progress.json", {"status": "complete", "name": name, "best_epoch": best_epoch, "best_val_macro_f1": best})
    print("CANDIDATE COMPLETE "+json.dumps({k: result[k] for k in ["name", "best_epoch", "best_val_macro_f1", "training_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    try:
        train_candidate(args.name, read_json(ROOT / "configs" / args.config))
    except BaseException:
        save_json(ROOT / "runs" / args.name / "failure.json", {"traceback": traceback.format_exc()})
        raise
