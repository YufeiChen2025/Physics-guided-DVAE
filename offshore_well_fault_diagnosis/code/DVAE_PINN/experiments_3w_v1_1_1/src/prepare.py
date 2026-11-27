"""Audit every CSV, group overlapping events, fit train-only stats, cache windows.

The raw files are never modified. Missing targets remain excluded from every NLL.
Only causal forward filling is used for model inputs; raw masks are retained.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASS_NAMES, DATA, FEATURES, ROOT, SOURCES, read_json, save_json, sha256, source_of


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def make_split(records, seed=42):
    """Stratify groups by their class set and source; tiny strata stay in train."""
    uf = UnionFind(len(records))
    duplicates = {}
    wells = collections.defaultdict(list)
    for i, r in enumerate(records):
        if r["content_sha256"] in duplicates:
            uf.union(i, duplicates[r["content_sha256"]])
        else:
            duplicates[r["content_sha256"]] = i
        if r["source"] == "real" and r["start_ns"] is not None:
            well = r["relative_path"].split("/")[-1].split("_")[0]
            wells[well].append(i)
    overlap_pairs = 0
    for ids in wells.values():
        ids.sort(key=lambda i: records[i]["start_ns"])
        if not ids:
            continue
        anchor, end = ids[0], records[ids[0]]["end_ns"]
        for i in ids[1:]:
            if records[i]["start_ns"] <= end:
                uf.union(anchor, i)
                overlap_pairs += 1
                end = max(end, records[i]["end_ns"])
            else:
                anchor, end = i, records[i]["end_ns"]
    groups = collections.defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)
    strata = collections.defaultdict(list)
    for ids in groups.values():
        key = (tuple(sorted({records[i]["event_class"] for i in ids})), tuple(sorted({records[i]["source"] for i in ids})))
        strata[key].append(ids)
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    membership = {}
    for key, units in sorted(strata.items()):
        rng.shuffle(units)
        n = len(units)
        n_test = max(1, round(n * .15)) if n >= 3 else 0
        n_val = max(1, round(n * .15)) if n >= 3 else 0
        for j, ids in enumerate(units):
            part = "test" if j < n_test else "val" if j < n_test + n_val else "train"
            for i in ids:
                rel = records[i]["relative_path"]
                split[part].append(rel)
                membership[rel] = {"split": part, "group_id": min(ids)}
    for part in split:
        split[part].sort()
        present = {int(p.split("/")[0]) for p in split[part]}
        if present != set(range(9)):
            raise ValueError(f"Missing event classes in {part}: {set(range(9)) - present}")
    assert len({p for paths in split.values() for p in paths}) == len(records)
    for ids in groups.values():
        assert len({membership[records[i]["relative_path"]]["split"] for i in ids}) == 1
    return split, membership, {"groups": len(groups), "overlap_connections": overlap_pairs, "duplicate_instances": len(records) - len(duplicates), "max_group_size": max(map(len, groups.values()))}


def audit(data_root, output):
    records = []
    raw = output / "cache" / "instances"
    raw.mkdir(parents=True, exist_ok=True)
    paths = [p for c in range(9) for p in sorted((data_root / str(c)).glob("*.csv"))]
    if not paths:
        raise FileNotFoundError(data_root)
    stamp = time.monotonic()
    for i, path in enumerate(paths):
        rel = path.relative_to(data_root).as_posix()
        df = pd.read_csv(path, usecols=["timestamp", *FEATURES, "class"])
        values = df[FEATURES].to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        values[~mask] = np.nan
        label_raw = pd.to_numeric(df["class"], errors="coerce").to_numpy(dtype=float)
        known = np.isfinite(label_raw)
        if not np.equal(label_raw[known], np.floor(label_raw[known])).all():
            raise ValueError(f"Non-integer class values: {path}")
        labels = np.where(known, label_raw, -1).astype(np.int16)
        labels[(labels >= 101) & (labels <= 108)] -= 100
        if not np.isin(labels, [-1, *range(9)]).all():
            raise ValueError(f"Unexpected labels: {path}: {np.unique(labels)}")
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
            raise ValueError(f"Invalid or non-increasing timestamps: {path}")
        steps = ts.diff().dropna().dt.total_seconds().to_numpy()
        if len(steps) and not np.all(steps == 1):
            raise ValueError(f"Sampling is not exactly 1Hz; needs segmentation: {path}")
        sums = np.where(mask, values, 0).astype(np.float64).sum(0)
        squares = np.where(mask, values, 0).astype(np.float64)
        squares = np.square(squares).sum(0)
        frozen = [bool(len(v := values[mask[:, k], k]) > 0 and np.min(v) == np.max(v)) for k in range(len(FEATURES))]
        digest = hashlib.sha256(np.nan_to_num(values, nan=0).tobytes() + mask.tobytes() + labels.tobytes()).hexdigest()
        stem = f"{i:04d}"
        np.save(raw / f"{stem}_values.npy", values)
        np.save(raw / f"{stem}_labels.npy", labels)
        records.append({"relative_path": rel, "cache_id": stem, "source": source_of(path.name), "event_class": int(path.parent.name), "rows": len(df), "start_ns": int(ts.iloc[0].value), "end_ns": int(ts.iloc[-1].value), "valid_counts": mask.sum(0).tolist(), "frozen": frozen, "sums": sums.tolist(), "squares": squares.tolist(), "raw_label_counts": {str(k): int(v) for k, v in zip(*np.unique(np.where(known, label_raw, -1).astype(int), return_counts=True))}, "content_sha256": digest, "csv_sha256": sha256(path)})
        if (i+1) % 50 == 0 or i == len(paths)-1:
            print(f"audit {i+1}/{len(paths)} files, elapsed={time.monotonic()-stamp:.1f}s", flush=True)
    save_json(output / "audit" / "instances.json", records)
    return records


def prepare(data_root=DATA, output=ROOT, frozen_split=None):
    recfile = output / "audit" / "instances.json"
    records = read_json(recfile) if recfile.exists() else audit(data_root, output)
    split_path = output / "splits" / "fixed_seed42.json"
    if frozen_split is None:
        split, membership, group_info = make_split(records)
        splitdoc = {"seed": 42, "protocol": "class/source-stratified instance groups; exact content duplicates and overlapping intervals in the same well are grouped", "nominal_ratios": [.70, .15, .15], **group_info, **split, "membership": membership}
        save_json(split_path, splitdoc)
    else:
        splitdoc = read_json(frozen_split)
        split = {part: splitdoc[part] for part in ("train", "val", "test")}
        membership = splitdoc["membership"]
        group_info = {key: splitdoc[key] for key in
                      ("groups", "overlap_connections", "duplicate_instances", "max_group_size")}
        listed = [name for paths in split.values() for name in paths]
        actual = {record["relative_path"] for record in records}
        if len(listed) != len(set(listed)) or set(listed) != actual:
            raise ValueError("CSV files do not match the published 3W 1.1.1 split")
        if any(membership[name]["split"] != part for part, paths in split.items() for name in paths):
            raise ValueError("Inconsistent membership in the frozen split")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen_split, split_path)
    train = [r for r in records if membership[r["relative_path"]]["split"] == "train"]
    count = np.array([r["valid_counts"] for r in train], dtype=np.float64).sum(0)
    sums = np.array([r["sums"] for r in train]).sum(0)
    squares = np.array([r["squares"] for r in train]).sum(0)
    selected = np.flatnonzero(count > 0)
    mean = sums[selected] / count[selected]
    var = np.maximum(squares[selected] / count[selected] - mean**2, 1e-8)
    std = np.sqrt(var)
    stats = {"features": [FEATURES[k] for k in selected], "excluded_no_training_observations": [FEATURES[k] for k in range(8) if k not in selected], "mean": mean.tolist(), "std": std.tolist(), "train_valid_counts": count[selected].astype(int).tolist(), "fit_on": "all finite raw observations in train instances only", "split_sha256": sha256(output / "splits" / "fixed_seed42.json")}
    save_json(output / "configs" / "normalization.json", stats)
    by_rel = {r["relative_path"]: r for r in records}
    T, stride = 128, 128
    summary = {}
    for part in ["train", "val", "test"]:
        destination = output / "cache" / part
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "metadata.json").exists():
            cached = read_json(destination / "metadata.json")
            if cached["split_sha256"] != stats["split_sha256"] or cached["features"] != stats["features"]:
                raise ValueError("Cache protocol changed; use a new output root")
            summary[part] = cached
            continue
        entries = []
        for rel in split[part]:
            r = by_rel[rel]
            values = np.load(output / "cache" / "instances" / f"{r['cache_id']}_values.npy", mmap_mode="r")
            labels = np.load(output / "cache" / "instances" / f"{r['cache_id']}_labels.npy", mmap_mode="r")
            for start in range(0, len(labels)-T+1, stride):
                lab = labels[start:start+T]
                valid = lab[lab >= 0]
                if len(valid) < T * .5 or not np.isfinite(values[start:start+T, selected]).any():
                    continue
                y = int(np.argmax(np.bincount(valid, minlength=9)))
                entries.append([int(r["cache_id"]), start, y, SOURCES.index(r["source"])])
        entries = np.asarray(entries, dtype=np.int32)
        if len(entries) == 0:
            raise ValueError(f"Empty windows: {part}")
        n, d = len(entries), len(selected)
        xmap = np.lib.format.open_memmap(destination / "x.npy", mode="w+", dtype=np.float32, shape=(n, T, d))
        mmap = np.lib.format.open_memmap(destination / "mask.npy", mode="w+", dtype=np.uint8, shape=(n, T, d))
        counts_by_source = np.zeros((3, 9), dtype=np.int64)
        coverage = np.zeros((3, d), dtype=np.int64)
        offset = 0
        for rel in split[part]:
            r = by_rel[rel]
            ii = int(r["cache_id"])
            local = entries[entries[:, 0] == ii]
            if not len(local):
                continue
            values = np.load(output / "cache" / "instances" / f"{r['cache_id']}_values.npy")[:, selected]
            observed = np.isfinite(values)
            # No backward filling: future observations never fill earlier inputs.
            filled = pd.DataFrame(values).ffill().to_numpy(dtype=np.float32)
            filled = np.clip(np.nan_to_num((filled - mean) / std, nan=0, posinf=0, neginf=0), -20, 20).astype(np.float32)
            for _, start, y, source in local:
                xmap[offset] = filled[start:start+T]
                mmap[offset] = observed[start:start+T]
                counts_by_source[source, y] += 1
                coverage[source] += observed[start+68:start+84].sum(0)
                offset += 1
        assert offset == n
        xmap.flush(); mmap.flush()
        np.save(destination / "index.npy", entries)
        meta = {"windows": n, "features": stats["features"], "T": T, "stride": stride, "label_policy": "majority of normalized valid labels; at least 50% labeled; transient 100+c -> c; ties choose lowest id", "split_sha256": stats["split_sha256"], "class_counts": np.bincount(entries[:, 2], minlength=9).tolist(), "source_class_counts": {s: counts_by_source[i].tolist() for i, s in enumerate(SOURCES)}, "chain_future_observed_fraction_by_source": {s: (coverage[i] / max(1, int(counts_by_source[i].sum()) * 16)).tolist() for i, s in enumerate(SOURCES)}}
        save_json(destination / "metadata.json", meta)
        summary[part] = meta
        print(f"cached {part}: {n} windows, class_counts={meta['class_counts']}", flush=True)
    audit_summary = {"files": len(records), "rows": sum(r["rows"] for r in records), "files_by_source": dict(collections.Counter(r["source"] for r in records)), "files_by_class_source": {str(c): dict(collections.Counter(r["source"] for r in records if r["event_class"] == c)) for c in range(9)}, "all_features": FEATURES, "features": stats["features"], "valid_counts_all": np.array([r["valid_counts"] for r in records]).sum(0).tolist(), "frozen_instance_counts": np.array([r["frozen"] for r in records]).sum(0).tolist(), "grouping": group_info, "file_counts_by_split": {s: len(split[s]) for s in split}, "windows": summary, "limitations": ["Class 8 has no real instances in this release", "CSV package dataset.ini reports 1.1.0; Figshare 1.1.1 archive checksum verified separately", "latent-only valve proxies have no observed valve calibration", "No cross-version transfer or independence claim"]}
    save_json(output / "audit" / "summary.json", audit_summary)
    lines = ["# 3W 1.1.1 数据审计", "", f"事件文件：{len(records)}；原始行数：{audit_summary['rows']}。", f"输入变量：{', '.join(stats['features'])}。", f"仅依据训练集剔除全缺失变量：{stats['excluded_no_training_observations']}。", "", "## 实例来源", "", "| 类别 | 真实 | 仿真 | 人工绘制 |", "|---|---:|---:|---:|"]
    for c in range(9):
        counts = audit_summary["files_by_class_source"][str(c)]
        lines.append(f"| {c} {CLASS_NAMES[c]} | {counts.get('real',0)} | {counts.get('simulated',0)} | {counts.get('drawn',0)} |")
    lines += ["", "## 分组与窗口", "", json.dumps(group_info, ensure_ascii=False), "", "| 划分 | 文件数 | 窗口数 |", "|---|---:|---:|"]
    for part in summary:
        lines.append(f"| {part} | {len(split[part])} | {summary[part]['windows']} |")
    lines += ["", "第 8 类只有仿真实例；真实子集指标不得声称覆盖该类。所有目标损失保留原始观测掩码。", "", "流量 QGL 的监督覆盖率、各来源各类窗口支持数详见 summary.json。"]
    (output / "audit" / "数据审计.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(output / "audit" / "READY.json", {"status": "complete", "split_sha256": stats["split_sha256"]})
    print("PREPARATION COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA)
    ap.add_argument("--output", type=Path, default=ROOT)
    ap.add_argument("--frozen-split", type=Path)
    args = ap.parse_args()
    prepare(args.data_root, args.output, args.frozen_split)
