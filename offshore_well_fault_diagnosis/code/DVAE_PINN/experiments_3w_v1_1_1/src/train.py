"""One reproducible run. Test data are not evaluated during model selection."""
from __future__ import annotations

import argparse
import csv
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import CLASS_NAMES, CODE, ROOT, SOURCES, read_json, save_json, sha256
from models import Diagnoser


class WindowDataset(Dataset):
    def __init__(self, part, limit=0):
        root = ROOT / "cache" / part
        self.x = np.load(root / "x.npy", mmap_mode="r")
        self.mask = np.load(root / "mask.npy", mmap_mode="r")
        self.index = np.load(root / "index.npy")
        # Diagnostic limits are evenly spread, never the first class only.
        self.ids = np.linspace(0, len(self.index)-1, min(limit, len(self.index)), dtype=int) if limit else np.arange(len(self.index))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = self.ids[i]
        return torch.from_numpy(self.x[j].copy()), torch.from_numpy(self.mask[j].copy()).bool(), torch.tensor(int(self.index[j, 2]), dtype=torch.long), torch.tensor(j, dtype=torch.long)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def classification_metrics(y, pred, classes=None):
    classes = list(range(9)) if classes is None else list(classes)
    cm = np.bincount(np.asarray(y, int)*9+np.asarray(pred, int), minlength=81).reshape(9, 9)
    details = {}
    for c in range(9):
        support = int(cm[c].sum())
        tp = int(cm[c, c])
        precision = tp / max(1, int(cm[:, c].sum()))
        recall = tp / max(1, support)
        details[str(c)] = {"name": CLASS_NAMES[c], "support": support, "precision": precision, "recall": recall, "f1": 2*precision*recall / max(1e-12, precision+recall)}
    return {"accuracy": float(np.trace(cm) / max(1, cm.sum())), "macro_f1": float(np.mean([details[str(c)]["f1"] for c in classes])) if classes else None, "balanced_accuracy": float(np.mean([details[str(c)]["recall"] for c in classes])) if classes else None, "macro_classes": classes, "per_class": details, "confusion_matrix": cm.tolist(), "samples": int(cm.sum())}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_y, all_ids = [], [], []
    for x, _, y, ids in loader:
        logits = model.predict(x.to(device, non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.numpy()); all_ids.append(ids.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y), np.concatenate(all_ids)


def atomic_torch_save(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def run(args):
    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    stats = read_json(ROOT / "configs" / "normalization.json")
    settings = vars(args).copy()
    settings.update({"features": stats["features"], "split_sha256": stats["split_sha256"], "normalization_sha256": sha256(ROOT / "configs" / "normalization.json"), "implementation_sha256": {p.name: sha256(p) for p in sorted((CODE / "src").glob("*.py"))}, "framework": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}, "loss": {"beta": .001, "classification": 2., "rule": .0005, "chain": .001, "chain_flow": .005, "chain_temperature": 1.}, "latent_sampling": "rsample (reparameterized), consistent for every group", "reconstruction": "raw-observation-masked per-window mean NLL multiplied by T", "auxiliary_latent": "encoded from first 64 input samples only", "class_weighting": "none", "rule_logit_fusion": False})
    config_path = run_dir / "config.json"
    if config_path.exists():
        old = read_json(config_path)
        # Source changes invalidate resumability, including future reporting edits.
        if old != settings:
            raise ValueError(f"Existing run config differs; use a new run name: {args.name}")
    else:
        save_json(config_path, settings)
    if (run_dir / "validation_result.json").exists():
        print(f"ALREADY COMPLETE: {args.name}", flush=True)
        return
    model = Diagnoser(stats["features"], args.evidence, args.chain, args.rule)
    warm_start = None
    if args.init:
        init = torch.load(ROOT / "runs" / args.init / "best.pt", map_location="cpu")
        if init["settings"]["split_sha256"] != settings["split_sha256"] or init["settings"]["seed"] != args.seed:
            raise ValueError("Warm-start split or training seed mismatch")
        warm_start = model.warm_start(init["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    datasets = {p: WindowDataset(p, args.limit_train if p == "train" else args.limit_val) for p in ["train", "val"]}
    # Epoch-local sampler seeds give every ablation exactly the same window order.
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size*2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    begin, best, best_epoch, stale, history, total_seconds = 0, -1., -1, 0, [], 0.
    last = run_dir / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location=device)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        begin, best, best_epoch, stale, history, total_seconds = (checkpoint[k] for k in ["next_epoch", "best", "best_epoch", "stale", "history", "total_seconds"])
        restore_rng(checkpoint["rng"])
        print(f"RESUME {args.name}: epoch {begin+1}", flush=True)
    else:
        save_json(run_dir / "initialization.json", {"warm_start": warm_start, "parameters": sum(p.numel() for p in model.parameters()), "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
    start = time.monotonic()
    for epoch in range(begin, args.epochs):
        stamp = time.monotonic()
        generator = torch.Generator().manual_seed(args.seed*10000+epoch+1000)
        train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
        model.train()
        totals, seen, maximum_grad = {}, 0, 0.
        for step, (x, mask, y, _) in enumerate(train_loader):
            x, mask, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(x, mask, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch+1}, step={step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 20., error_if_nonfinite=True)
            maximum_grad = max(maximum_grad, float(grad))
            optimizer.step()
            batch = len(y); seen += batch
            for k, v in parts.items():
                if not k.endswith("_targets"):
                    totals[k] = totals.get(k, 0.) + float(v) * batch
            if step % 200 == 0:
                status = {"status": "training", "run": args.name, "epoch": epoch+1, "max_epochs": args.epochs, "step": step, "steps_per_epoch": len(train_loader), "best_val_macro_f1": best, "seconds_this_epoch": time.monotonic()-stamp}
                save_json(run_dir / "progress.json", status)
                print(json.dumps(status), flush=True)
            if args.max_steps and step+1 >= args.max_steps:
                break
        logits, y, _ = predict(model, val_loader, device)
        val_metrics = classification_metrics(y, logits.argmax(1))
        score = val_metrics["macro_f1"]
        if score > best:
            best, best_epoch, stale = score, epoch+1, 0
            atomic_torch_save(run_dir / "best.pt", {"model": model.state_dict(), "settings": settings, "epoch": epoch+1, "val_metrics": val_metrics})
        else:
            stale += 1
        row = {"epoch": epoch+1, "seconds": time.monotonic()-stamp, "train": {k: v/seen for k, v in totals.items()}, "max_unclipped_gradient_norm": maximum_grad, "validation": val_metrics}
        history.append(row)
        elapsed = total_seconds + time.monotonic()-start
        atomic_torch_save(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "settings": settings, "next_epoch": epoch+1, "best": best, "best_epoch": best_epoch, "stale": stale, "history": history, "rng": rng_state(), "total_seconds": elapsed})
        save_json(run_dir / "history.json", history)
        save_json(run_dir / "progress.json", {"status": "epoch_complete", "run": args.name, "epoch": epoch+1, "best_val_macro_f1": best, "best_epoch": best_epoch, "elapsed_seconds": elapsed})
        print(f"{args.name} epoch={epoch+1} seconds={row['seconds']:.1f} val_macro_f1={score:.6f} best={best:.6f}", flush=True)
        if args.patience and stale >= args.patience:
            break
    result = {"run": args.name, "seed": args.seed, "group": args.group, "epochs_completed": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best, "training_seconds": total_seconds+time.monotonic()-start, "train_windows": len(datasets["train"]), "validation_windows": len(datasets["val"]), "split_sha256": settings["split_sha256"], "diagnostic_only": bool(args.max_steps or args.limit_train or args.limit_val), "test_evaluated": False}
    save_json(run_dir / "validation_result.json", result)
    save_json(run_dir / "progress.json", {"status": "complete", **result})
    print(f"COMPLETE {args.name}: {json.dumps(result)}", flush=True)


def evaluate(name, batch_size=256):
    run_dir = ROOT / "runs" / name
    result_path = run_dir / "test_result.json"
    if result_path.exists():
        return read_json(result_path)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    cfg = checkpoint["settings"]
    if cfg["max_steps"] or cfg["limit_train"] or cfg["limit_val"]:
        raise ValueError("Diagnostic runs cannot be included in formal test reports")
    seed_everything(cfg["seed"])
    model = Diagnoser(cfg["features"], cfg["evidence"], cfg["chain"], cfg["rule"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = WindowDataset("test")
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=device.type == "cuda")
    logits, y, ids = predict(model, loader, device)
    sources = ds.index[ids, 3]
    pred = logits.argmax(1)
    result = {"run": name, "seed": cfg["seed"], "group": cfg["group"], "best_epoch": checkpoint["epoch"], "split_sha256": cfg["split_sha256"], "mixed_9class": classification_metrics(y, pred), "by_source": {}}
    for i, s in enumerate(SOURCES):
        take = sources == i
        present = sorted(np.unique(y[take]).tolist())
        result["by_source"][s] = classification_metrics(y[take], pred[take], present)
        result["by_source"][s]["unsupported_classes"] = sorted(set(range(9))-set(present))
    np.savez_compressed(run_dir / "test_predictions.npz", logits=logits, labels=y, predictions=pred, window_ids=ids, sources=sources)
    save_json(result_path, result)
    return result


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", default="stage1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=.001)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--init", default="")
    ap.add_argument("--evidence", action="store_true")
    ap.add_argument("--rule", action="store_true")
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--evaluate", action="store_true")
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        if args.evaluate:
            print(json.dumps(evaluate(args.name), ensure_ascii=False))
        else:
            run(args)
    except BaseException:
        save_json(ROOT / "runs" / args.name / "failure.json", {"traceback": traceback.format_exc(), "time": time.time()})
        raise
