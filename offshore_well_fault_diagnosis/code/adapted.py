"""Run the packaged 3W 1.1.1 model branches against a prepared cache."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
CHAIN = CODE / "DVAE_PINN" / "chain_adaptation_20261111" / "experiment.py"
PROTOCOL = CODE.parent / "protocols" / "v111"


def load_chain():
    spec = importlib.util.spec_from_file_location("chain_experiment", CHAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_evidence(worker, stats):
    config = worker.OUT / "configs"
    cache = worker.OUT / "cache"
    for path in (config, cache):
        path.mkdir(parents=True, exist_ok=True)
    mode = worker.OUT / "evidence_mode.json"
    if mode.exists() and worker.read(mode)["stats"] != stats:
        raise ValueError("Use a separate --work-dir to switch between fitted and frozen evidence")
    worker.save(mode, {"stats": stats})
    if stats == "frozen":
        for name in ("transform.json", "new7_empirical_prior.json"):
            shutil.copyfile(PROTOCOL / name, config / name)
    elif not (worker.OUT / "prepared.json").exists():
        worker.prepare()


def train(experiment, group, seed, epochs):
    worker = experiment.W
    (worker.OUT / "runs").mkdir(parents=True, exist_ok=True)
    if group in ("B", "EVIDENCE"):
        worker.ARGS.seed = seed
        worker.ARGS.role = {"B": "rule", "EVIDENCE": "evidence"}[group]
        worker.train(max_epochs=epochs)
        return
    if group == "C":
        original_make_model = experiment.make_model

        def make_c_only(*args, **kwargs):
            model = original_make_model(*args, **kwargs)
            model.with_r = False
            return model

        experiment.make_model = make_c_only
    for role in ("full", "rule"):
        path = worker.OUT / "runs" / f"seed{seed}_{role}" / "initial.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            model = worker.initialise_stage2(seed, role)
            worker.atomic_torch_save(path, {"model": model.state_dict()})
    experiment.ROOT = worker.OUT.parent / f"chain_{group}"
    experiment.train(seed, "decay0", experiment.load_data(), max_epochs=epochs)


@torch.inference_mode()
def evaluate(experiment, group, seed, checkpoint_path, output):
    worker = experiment.W
    ckpt = torch.load(checkpoint_path, map_location=experiment.DEVICE, weights_only=False)
    if group == "A":
        sys.path.insert(0, str(worker.BASE_CODE / "tuning_20261028" / "src"))
        from tuned_model import TunedDiagnoser
        features = worker.read(worker.BASE / "configs" / "normalization.json")["features"]
        model = TunedDiagnoser(features, ckpt["config"])
    else:
        role = {"B": "rule", "C": "chain", "D": "full", "EVIDENCE": "evidence"}[group]
        model = worker.Model(2, role, seed)
    model.load_state_dict(ckpt["model"])
    model.to(experiment.DEVICE).eval()

    cache = worker.BASE / "cache" / "test"
    x = np.load(cache / "x.npy", mmap_mode="r")
    mask = np.load(cache / "mask.npy", mmap_mode="r")
    index = np.load(cache / "index.npy", mmap_mode="r")
    if group != "A":
        evidence_path = worker.OUT / "cache" / "test_e.npy"
        if not evidence_path.exists():
            raw, valid = worker.raw_features("test")
            evidence = worker.transform(raw, valid, worker.read(worker.OUT / "configs" / "transform.json"))
            np.save(evidence_path, evidence)
        evidence = np.load(evidence_path, mmap_mode="r")
    logits = []
    for start in range(0, len(index), 256):
        part = slice(start, start + 256)
        values = torch.tensor(np.array(x[part]), device=experiment.DEVICE)
        observed = torch.tensor(np.array(mask[part]), device=experiment.DEVICE, dtype=torch.bool)
        if group == "A":
            pred = model.predict(values, observed)
        else:
            selected = torch.tensor(np.array(evidence[part]), device=experiment.DEVICE)
            pred = model.predict(values, observed, selected)
        logits.append(pred.cpu().numpy())
    scores = np.concatenate(logits)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite diagnostic scores")
    predictions = scores.argmax(1)
    result = worker.classification_metrics(index[:, 2], predictions)
    result.update(group=group, seed=seed, checkpoint_epoch=ckpt["epoch"],
                  macro_p=float(np.mean([data["precision"] for data in result["per_class"].values()])),
                  macro_r=result["balanced_accuracy"])
    output.mkdir(parents=True, exist_ok=True)
    worker.save(output / "result.json", result)
    temporary = output / "test_predictions.tmp.npz"
    np.savez_compressed(temporary, logits=scores, predictions=predictions,
                        labels=index[:, 2], sources=index[:, 3])
    temporary.replace(output / "test_predictions.npz")
    print(f"{group}/{seed}: Macro-F1 = {result['macro_f1']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evidence", "train", "evaluate"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--group", choices=("A", "B", "C", "D", "EVIDENCE"))
    parser.add_argument("--stats", choices=("frozen", "fitted"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment = load_chain()
    if args.stage == "evidence":
        prepare_evidence(experiment.W, args.stats)
    elif args.stage == "train":
        train(experiment, args.group, args.seed, args.epochs)
    else:
        evaluate(experiment, args.group, args.seed, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
