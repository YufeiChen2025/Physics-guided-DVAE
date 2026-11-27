"""Portable entry points for the archived 3W experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
V2 = CODE / "TRANS" / "DVAE_3W"
PIN = CODE / "DVAE_PINN"
V111 = PIN / "experiments_3w_v1_1_1"
TUNING = V111 / "tuning_20261028"
PROTOCOLS = ROOT / "protocols"
REFERENCE = ROOT / "reference"
V111_RUNNER = CODE / "adapted.py"
GROUPS = {"2.0.0": ("A0", "B", "C", "D", "E"), "1.1.1": ("A", "B", "C", "D", "EVIDENCE")}
V2_EXTRA = ("GRU", "GRU-OBS7D", "LSTM", "LSTM-OBS7D", "E-ZERO-DECAY")
PARTS = ("train", "val", "test")
SEEDS = (42, 2025)
WINDOWS = {"2.0.0": (392484, 87405, 85717), "1.1.1": (276108, 61744, 57401)}
KEEP = "water_rise,water_smooth,dhsv_open_mean,pck_drop,pck_open_mean,dhsv_low_evidence,pck_low_evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def execute(script, args, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{script.name}  [log: {log}]", flush=True)
    with log.open("a", encoding="utf-8") as output:
        print(json.dumps([str(script), *map(str, args)], ensure_ascii=False), file=output, flush=True)
        run = subprocess.run([sys.executable, "-u", str(script), *map(str, args)],
                             cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    if run.returncode:
        with log.open(encoding="utf-8", errors="replace") as output:
            from collections import deque
            print("".join(deque(output, maxlen=25)), file=sys.stderr)
        raise RuntimeError(f"{script.name} exited with {run.returncode}. See {log}")


def verify_data(version, data):
    if not data:
        raise ValueError("--data-root must point to the extracted dataset directory")
    data = data.resolve()
    key = "v2" if version == "2.0.0" else "v111"
    frozen = read(PROTOCOLS / key / "fixed_split.json")
    missing = [name for part in PARTS for name in frozen[part] if not (data / name).is_file()]
    if missing:
        raise ValueError(f"{data} is missing {len(missing)} split files; examples: {missing[:3]}")
    return data


def prepare_v111(data, work, env):
    marker = work / "data" / "audit" / "READY.json"
    source = work / "data_source.json"
    if source.is_file() and read(source)["data_root"] != str(data):
        raise ValueError(f"Use a new --work-dir for a different 3W 1.1.1 dataset: {source}")
    write(source, {"data_root": str(data)})
    if not marker.is_file():
        execute(V111 / "src" / "prepare.py",
                ["--data-root", data, "--frozen-split", PROTOCOLS / "v111" / "fixed_split.json"],
                work / "logs" / "prepare.log", env)
    protocol = PROTOCOLS / "v111"
    if file_hash(work / "data" / "splits" / "fixed_seed42.json") != file_hash(protocol / "fixed_split.json"):
        raise ValueError("Prepared 3W 1.1.1 cache has a different split")
    stats = read(work / "data" / "configs" / "normalization.json")
    frozen_stats = read(protocol / "normalization.json")
    for field in ("features", "train_valid_counts", "split_sha256"):
        if stats[field] != frozen_stats[field]:
            raise ValueError(f"Dataset check failed for normalization.{field}")
    for field in ("mean", "std"):
        if any(abs(v - ref) > max(1e-8, 1e-10 * abs(ref)) for v, ref in zip(stats[field], frozen_stats[field], strict=True)):
            raise ValueError(f"Dataset check failed for normalization.{field}")
    counts = tuple(read(work / "data" / "cache" / part / "metadata.json")["windows"] for part in PARTS)
    if counts != WINDOWS["1.1.1"]:
        raise ValueError(f"3W 1.1.1 window counts differ from the original: {counts}")


def command_v2(args, data, group):
    protocol = PROTOCOLS / "v2"
    config = read(protocol / f"{group}.json")
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    env = os.environ.copy()
    env.pop("RULE_EVIDENCE_KEEP", None)
    env.pop("RULE_EVIDENCE_DROP", None)
    env.pop("RULE_EVIDENCE_ABLATION_NAME", None)
    baseline_model = config.get("model_type") in ("gru", "lstm")
    if group in ("E", "E-ZERO-DECAY"):
        env["RULE_EVIDENCE_DROP"] = "dp_trend,pck_smooth,dhsv_drop"
    elif not baseline_model and group != "A0":
        env["RULE_EVIDENCE_KEEP"] = KEEP
    config.update(data_root=data, split_file=protocol / "fixed_split.json", run_name=group)
    if not baseline_model:
        config["pinn_root"] = PIN / "10d_dimension_ablation"
    work = args.work_dir / "v2" / args.action
    config["output_dir"] = work
    if args.action == "evaluate":
        config.pop("save_resume_state", None)
        if args.checkpoint_source == "trained":
            config["evaluate_checkpoint"] = args.work_dir / "v2" / "train" / f"{group}.pth"
        else:
            config["evaluate_checkpoint"] = REFERENCE / "v2" / group / "best.pt"
    elif not baseline_model and group != "A0":
        start = (REFERENCE / "v2" / "A0" / "best.pt" if args.baseline == "reference"
                 else work / "A0.pth")
        if not start.is_file():
            raise ValueError("Train A0 first or use --baseline reference")
        config["init_checkpoint"] = start
        config["resume_state"] = work / f"{group}.resume.pth"
    elif group == "A0":
        config["resume_state"] = work / "A0.resume.pth"
        config["save_resume_state"] = True
    if "evaluate_checkpoint" in config and not config["evaluate_checkpoint"].is_file():
        raise FileNotFoundError(config["evaluate_checkpoint"])
    result = work / f"{group}_results.json"
    if args.action == "train":
        run_record = dict(config)
        run_record["evidence_keep"] = env.get("RULE_EVIDENCE_KEEP")
        run_record["evidence_drop"] = env.get("RULE_EVIDENCE_DROP")
        run_record = {key: str(value) if isinstance(value, Path) else value for key, value in run_record.items()}
        manifest = work / f"{group}_requested_config.json"
        if manifest.is_file() and read(manifest) != run_record:
            raise ValueError(f"Training configuration changed; use a new --work-dir ({manifest})")
        write(manifest, run_record)
        if result.exists():
            if not (work / f"{group}.pth").is_file():
                raise FileNotFoundError(work / f"{group}.pth")
            return result
    argv = []
    for key, value in config.items():
        if value is True:
            argv.append(f"--{key}")
        elif value is not False and value is not None:
            argv.extend((f"--{key}", value))
    script = V2 / ("train_recurrent_obs7d.py" if baseline_model else "train_3w_dvae_classifier.py")
    execute(script, argv, work / "logs" / f"{group}.log", env)
    out = read(result)
    counts = tuple(out[f"{part}_windows"] for part in PARTS)
    expected_sha = read(protocol / "fixed_split.json")["test_sha256"]
    if counts != WINDOWS["2.0.0"] or out["test_split_sha256"] != expected_sha:
        raise ValueError(f"3W 2.0.0 dataset check failed: {counts}, {out['test_split_sha256']}")
    return result


def v111_checkpoint(work, group, seed):
    if group == "A":
        return work / "adapted" / "runs" / f"seed{seed}_stage1" / "best.pt"
    if group in ("B", "EVIDENCE"):
        role = "rule" if group == "B" else "evidence"
        return work / "adapted" / "runs" / f"seed{seed}_{role}" / "best.pt"
    return work / f"chain_{group}" / "runs" / f"seed{seed}_decay0" / "best.pt"


def baseline_v111(mode, seed, work, env):
    record = work / "baseline_mode.json"
    if record.exists() and read(record)["mode"] != mode:
        raise ValueError("Baseline mode has changed; use a new --work-dir")
    write(record, {"mode": mode})
    reference = REFERENCE / "v111" / f"seed{seed}"
    dst = v111_checkpoint(work, "A", seed)
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "reference":
        shutil.copyfile(reference / "A" / "best.pt", dst)
        return
    stage1 = work / "data" / "runs" / f"seed{seed}_stage1" / "best.pt"
    stage1.parent.mkdir(parents=True, exist_ok=True)
    if mode == "scratch":
        execute(V111 / "src" / "train.py", ["--name", f"seed{seed}_stage1", "--seed", seed],
                work / "logs" / f"seed{seed}_stage1.log", env)
    elif not stage1.exists():
        shutil.copyfile(reference / "stage1.pt", stage1)
    run_name = f"F_seed{seed}_baseline"
    config = PROTOCOLS / "v111" / f"P3_seed{seed}.json"
    execute(TUNING / "confirmation" / "fast_train.py", ["--name", run_name, "--config", config],
            work / "logs" / f"{run_name}.log", env)
    trained = work / "data" / "tuning_20261028" / "runs" / run_name / "best.pt"
    shutil.copyfile(trained, dst)


def command_v111(args, data, groups, seeds):
    work = args.work_dir / "v111"
    env = os.environ.copy() | {"DVAE_V111_WORK": str(work), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    prepare_v111(data, work, env)
    if args.action == "prepare":
        return
    is_reference = args.action == "evaluate" and args.checkpoint_source == "reference"
    block = "frozen" if is_reference else "fitted"
    if is_reference:
        env["DVAE_V111_ADAPTED"] = str(work / "reference_adapted")
    execute(V111_RUNNER, ["evidence", "--stats", block],
            work / "logs" / f"evidence_{block}.log", env)
    if args.action == "train":
        for seed in seeds:
            baseline_v111(args.baseline, seed, work, env)
            for group in groups:
                if group != "A":
                    execute(V111_RUNNER, ["train", "--seed", seed, "--group", group],
                            work / "logs" / f"train_{group}_seed{seed}.log", env)
    for seed in seeds:
        for group in groups:
            checkpoint = (REFERENCE / "v111" / f"seed{seed}" / group / "best.pt"
                          if is_reference else v111_checkpoint(work, group, seed))
            destination = work / args.action / f"seed{seed}" / group
            execute(V111_RUNNER, ["evaluate", "--seed", seed, "--group", group,
                                  "--checkpoint", checkpoint, "--output", destination],
                    work / "logs" / f"{args.action}_{group}_seed{seed}.log", env)


def metric_record(result):
    return result.get("test_metrics", result.get("mixed", result))


def prf(result):
    metrics = metric_record(result)
    matrix = metrics["confusion_matrix"]
    n = len(matrix)
    columns = [sum(row[i] for row in matrix) for i in range(n)]
    rows = list(map(sum, matrix))
    tp = [matrix[i][i] for i in range(n)]
    p = statistics.mean(tp[i] / max(1, columns[i]) for i in range(n))
    r = statistics.mean(tp[i] / max(1, rows[i]) for i in range(n))
    f = statistics.mean(2 * tp[i] / max(1, rows[i] + columns[i]) for i in range(n))
    if abs(f - metrics["macro_f1"]) > 1e-10:
        raise ValueError(f"Stored F1 does not match confusion matrix: {f} vs {metrics['macro_f1']}")
    return p, r, f


def table(version, source, work, extras=False):
    lines = [f"## 3W {version} ({source})", "",
             "| Group | Seeds | Macro-P | Macro-R | Macro-F1 |", "|---|---|---:|---:|---:|"]
    groups = GROUPS[version] + (V2_EXTRA if version == "2.0.0" and extras else ())
    for group in groups:
        if version == "2.0.0":
            paths = {42: (REFERENCE / "v2" / group / "result.json" if source == "reference"
                          else work / "v2" / source / f"{group}_results.json")}
        else:
            base = REFERENCE / "v111" if source == "reference" else work / "v111" / source
            paths = {seed: base / f"seed{seed}" / group / "result.json" for seed in SEEDS}
        rows = {seed: prf(read(path)) for seed, path in paths.items() if path.is_file()}
        if not rows:
            continue
        summary = [f"{statistics.mean(values):.6f}" + (f" +/- {statistics.stdev(values):.6f}" if len(values) > 1 else "")
                   for values in zip(*rows.values())]
        lines.append("| " + " | ".join([group, ", ".join(map(str, rows)), *summary]) + " |")
    if len(lines) == 4:
        raise ValueError(f"No {source} results for 3W {version}")
    return "\n".join(lines)


def verify_bundle():
    manifest = ROOT / "SHA256SUMS.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    for version in GROUPS:
        print(table(version, "reference", ROOT, extras=True))
    print("Packaged files and stored confusion matrices verified.")


def compare_evaluations(work, version):
    versions = (version,) if version else tuple(GROUPS)
    changed = []
    checked = 0
    for v in versions:
        if v == "2.0.0":
            files = work.glob("v2/evaluate/*_results.json")
            pairs = ((p, REFERENCE / "v2" / p.stem.removesuffix("_results") / "result.json") for p in files)
        else:
            base = work / "v111" / "evaluate"
            pairs = ((p, REFERENCE / "v111" / p.relative_to(base)) for p in base.glob("*/*/result.json"))
        for actual, reference in sorted(pairs):
            if not reference.is_file():
                continue
            evaluated = metric_record(read(actual))
            original = metric_record(read(reference))
            delta = evaluated["macro_f1"] - original["macro_f1"]
            identical = evaluated["confusion_matrix"] == original["confusion_matrix"]
            checked += 1
            label = actual.relative_to(work).as_posix()
            print(f"{label}: confusion matrix {'identical' if identical else 'differs'}, Macro-F1 delta {delta:+.8f}")
            if not identical:
                changed.append(label)
    if not checked:
        raise ValueError("No checkpoint evaluations found in --work-dir")
    if changed:
        raise ValueError(f"{len(changed)} of {checked} confusion matrices differ")
    print(f"All {checked} checked confusion matrices match.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "prepare", "train", "evaluate", "report", "compare"))
    parser.add_argument("--version", choices=tuple(GROUPS))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work")
    parser.add_argument("--groups", nargs="+", help="Defaults to all packaged groups for the selected version")
    parser.add_argument("--include-extras", action="store_true", help="Include supplemental 2.0.0 groups")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, help="1.1.1 only; defaults to both")
    parser.add_argument("--num-workers", type=int, help="2.0.0 DataLoader workers, default 4")
    parser.add_argument("--baseline", choices=("reference", "p3", "scratch"),
                        help="Default: scratch for 2.0.0, P3 retraining for 1.1.1")
    parser.add_argument("--checkpoint-source", choices=("reference", "trained"), default="reference")
    parser.add_argument("--source", choices=("reference", "evaluate", "train"), default="reference")
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    if args.action == "verify":
        verify_bundle()
        return
    if args.action == "compare":
        compare_evaluations(args.work_dir, args.version)
        return
    if args.action == "report":
        versions = (args.version,) if args.version else tuple(GROUPS)
        print("\n\n".join(table(v, args.source, args.work_dir, extras=args.include_extras) for v in versions))
        return
    if not args.version:
        parser.error("--version is required")
    version = args.version
    seeds = tuple(dict.fromkeys(args.seeds or SEEDS))
    extras = V2_EXTRA if version == "2.0.0" else ()
    defaults = GROUPS[version] + (extras if args.include_extras else ())
    groups = tuple(dict.fromkeys(args.groups or defaults))
    unknown = set(groups) - set(GROUPS[version] + extras)
    if unknown:
        parser.error(f"Unknown groups for {version}: {', '.join(sorted(unknown))}")
    args.baseline = args.baseline or ("scratch" if version == "2.0.0" else "p3")
    if version == "2.0.0" and (args.seeds or args.baseline == "p3"):
        parser.error("3W 2.0.0 uses only seed 42 and does not use P3")
    data = verify_data(version, args.data_root)
    if version == "1.1.1":
        command_v111(args, data, groups, seeds)
    elif args.action != "prepare":
        if args.action == "train" and "A0" in groups:
            groups = ("A0", *(group for group in groups if group != "A0"))
        for group in groups:
            command_v2(args, data, group)
    if args.action == "prepare":
        print(f"Validated 3W {version} data in {data}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        sys.exit(str(error))
