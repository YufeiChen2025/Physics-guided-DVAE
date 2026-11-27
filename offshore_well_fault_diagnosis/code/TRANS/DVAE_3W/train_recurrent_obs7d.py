from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from threew.dataset import FEATURE_COLUMNS, ThreeWWindowDataset, compute_normalization_stats
from train_3w_dvae_classifier import load_split, metrics_from_logits, paths_sha256


OBS7D_NAMES = [
    "wc_proxy_rise",
    "wc_proxy_smooth",
    "dhsv_open_mean",
    "pck_drop",
    "pck_open_mean",
    "dhsv_low_evidence",
    "pck_low_evidence",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quarter_means(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = max(values.shape[1] // 4, 1)
    return values[:, :q].mean(dim=1), values[:, -q:].mean(dim=1)


def sequence_roughness(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] < 2:
        return values.mean(dim=1) * 0.0
    return (values[:, 1:] - values[:, :-1]).square().mean(dim=1).sqrt()


def observed_7d_evidence(x_norm: torch.Tensor, stats: dict[str, np.ndarray]) -> torch.Tensor:
    """Pure observation-derived 7D proxy evidence for non-DVAE backbones.

    This intentionally avoids DVAE decoder/latent heads. The water dimensions
    use a pressure-flow-temperature proxy because 3W lacks measured water cut.
    """
    mean = torch.as_tensor(stats["mean"], dtype=x_norm.dtype, device=x_norm.device).view(1, 1, -1)
    std = torch.as_tensor(stats["std"], dtype=x_norm.dtype, device=x_norm.device).view(1, 1, -1).clamp_min(1e-6)
    x_units = x_norm * std + mean
    col = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

    def bounded(name: str, percent_scale: bool = False) -> torch.Tensor:
        values = torch.nan_to_num(x_units[..., col[name]], nan=0.0, posinf=0.0, neginf=0.0)
        if percent_scale:
            values = values / 100.0
        return values.clamp(0.0, 1.0)

    def normalized_col(name: str) -> torch.Tensor:
        return torch.nan_to_num(x_norm[..., col[name]], nan=0.0, posinf=0.0, neginf=0.0)

    pck_open = bounded("ABER-CKP", percent_scale=True)
    dhsv_open = bounded("ESTADO-DHSV", percent_scale=False)
    pck_first, pck_last = quarter_means(pck_open)

    qbs = normalized_col("QBS")
    qgl = normalized_col("QGL")
    p_tpt = normalized_col("P-TPT")
    p_pdg = normalized_col("P-PDG")
    t_tpt = normalized_col("T-TPT")
    wc_proxy = torch.sigmoid(0.35 * qbs - 0.20 * qgl + 0.15 * (p_pdg - p_tpt) - 0.10 * t_tpt)
    wc_first, wc_last = quarter_means(wc_proxy)

    features = torch.stack(
        [
            wc_last - wc_first,
            sequence_roughness(wc_proxy),
            dhsv_open.mean(dim=1),
            pck_first - pck_last,
            pck_open.mean(dim=1),
            F.relu(0.85 - dhsv_open).mean(dim=1),
            F.relu(0.35 - pck_open).mean(dim=1),
        ],
        dim=1,
    )
    return torch.tanh(torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))


class RecurrentFaultClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        model_type: str,
        evidence_dim: int = 0,
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if model_type not in {"gru", "lstm"}:
            raise ValueError("model_type must be gru or lstm")
        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM
        self.pooling = pooling
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, evidence: torch.Tensor | None = None) -> torch.Tensor:
        out, _hidden = self.rnn(x)
        feat = out[:, -1, :] if self.pooling == "last" else out.mean(dim=1)
        if evidence is not None:
            feat = torch.cat([feat, evidence], dim=1)
        return self.classifier(feat)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_obs7d: bool,
    stats: dict[str, np.ndarray],
) -> dict:
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device).float()
            labels = labels.to(device)
            evidence = observed_7d_evidence(x, stats) if use_obs7d else None
            logits_all.append(model(x, evidence).cpu())
            labels_all.append(labels.cpu())
    return metrics_from_logits(torch.cat(logits_all), torch.cat(labels_all), num_classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=TRAIN_DIR.parent / "3w_dataset_2.0.0")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR.parent / "runs")
    parser.add_argument("--split_file", type=Path, default=TRAIN_DIR / "splits" / "3w_seed42_full_val15_test15.json")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model_type", choices=["gru", "lstm"], required=True)
    parser.add_argument("--use_obs7d", action="store_true")
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max_windows_per_file", type=int, default=100000)
    parser.add_argument("--clip_norm", type=float, default=20.0)
    parser.add_argument("--evaluate_checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L2", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()
    split_file = args.split_file.resolve()

    train_paths, val_paths, test_paths = load_split(split_file, data_root)
    evaluation_checkpoint = None
    if args.evaluate_checkpoint is not None:
        evaluation_checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        stats = evaluation_checkpoint["stats"]
        args.epochs = 0
    else:
        stats = compute_normalization_stats(train_paths, seed=args.seed)
    ds_kwargs = dict(
        window_size=args.T,
        stride=args.stride,
        stats=stats,
        task="multiclass",
        max_windows_per_file=args.max_windows_per_file,
        seed=args.seed,
        cache_size=32,
        clip_norm=args.clip_norm,
    )
    train_ds = ThreeWWindowDataset(train_paths, **ds_kwargs)
    val_ds = ThreeWWindowDataset(val_paths, **ds_kwargs)
    test_ds = ThreeWWindowDataset(test_paths, **ds_kwargs)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentFaultClassifier(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=args.hidden_dim,
        num_classes=10,
        model_type=args.model_type,
        evidence_dim=len(OBS7D_NAMES) if args.use_obs7d else 0,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
    model_path = args.evaluate_checkpoint or (args.output_dir / f"{args.run_name}.pth")
    result_path = args.output_dir / f"{args.run_name}_results.json"
    best_val = None if evaluation_checkpoint is not None else -1.0
    epochs_since_improvement = 0
    started = time.time()

    print(f"device: {device}")
    print(f"run: {args.run_name}, model={args.model_type}, use_obs7d={args.use_obs7d}")
    print(f"split: {split_file}")
    print(f"test split sha256: {paths_sha256(test_paths, data_root)}")
    print(f"windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, labels in train_loader:
            x = x.to(device).float()
            labels = labels.to(device)
            evidence = observed_7d_evidence(x, stats) if args.use_obs7d else None
            optimizer.zero_grad()
            logits = model(x, evidence)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device, 10, args.use_obs7d, stats)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            epochs_since_improvement = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "stats": stats}, model_path)
            saved = " saved"
        else:
            epochs_since_improvement += 1
            saved = ""
        print(
            f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}{saved}"
        )
        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print(f"early stopping after {epoch}; best val macro-F1={best_val:.4f}")
            break

    checkpoint = evaluation_checkpoint or torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, 10, args.use_obs7d, stats)
    result = {
        "task": "multiclass",
        "model_type": args.model_type,
        "use_obs7d": args.use_obs7d,
        "obs7d_names": OBS7D_NAMES if args.use_obs7d else [],
        "runtime_minutes": (time.time() - started) / 60.0,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "best_val_macro_f1": best_val,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "split_file": str(split_file),
        "test_split_sha256": paths_sha256(test_paths, data_root),
        "evidence_note": (
            "obs7d is a pure observation-derived proxy evidence set; water dimensions use a "
            "pressure-flow-temperature proxy because measured water cut is unavailable."
            if args.use_obs7d
            else "No rule evidence used."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
