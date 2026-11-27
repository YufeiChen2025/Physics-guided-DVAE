"""Export per-run confusion matrices and manuscript-ready metric tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reproduce import GROUPS, REFERENCE, ROOT, SEEDS, V2_EXTRA, metric_record, prf, read

CLASSES = (
    "Normal", "Abrupt increase of BSW", "Spurious closure of DHSV", "Severe slugging",
    "Flow instability", "Rapid productivity loss", "Quick restriction in PCK",
    "Scaling in PCK", "Hydrate in production line", "Hydrate in service line",
)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def result_paths(source, work, extras):
    for group in GROUPS["2.0.0"] + (V2_EXTRA if extras else ()):
        path = (REFERENCE / "v2" / group / "result.json" if source == "reference"
                else work / "v2" / source / f"{group}_results.json")
        yield "2.0.0", group, 42, path
    base = REFERENCE / "v111" if source == "reference" else work / "v111" / source
    for group in GROUPS["1.1.1"]:
        for seed in SEEDS:
            yield "1.1.1", group, seed, base / f"seed{seed}" / group / "result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("reference", "evaluate", "train"), default="reference")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-extras", action="store_true")
    args = parser.parse_args()
    overall = []
    per_class = []
    progress = []
    for version, group, seed, path in result_paths(args.source, args.work_dir, args.include_extras):
        if not path.is_file():
            continue
        result = read(path)
        metrics = metric_record(result)
        confusion = metrics["confusion_matrix"]
        macro_p, macro_r, macro_f1 = prf(result)
        n_classes = len(confusion)
        run = (version, group, seed)
        hits = sum(row[i] for i, row in enumerate(confusion))
        total = sum(map(sum, confusion))
        overall.append((*run, hits / total, macro_p, macro_r, macro_f1, total))
        for i, row in enumerate(confusion):
            tp = row[i]
            estimated = sum(r[i] for r in confusion)
            truth = sum(row)
            per_class.append((*run, i, CLASSES[i], truth, tp / max(1, estimated),
                              tp / max(1, truth), 2 * tp / max(1, truth + estimated)))
        progress.append((*run, confusion[0][8], confusion[0][7], confusion[8][0], confusion[8][5]))
        write_csv(args.output_dir / "confusion" / f"{version}_{group}_seed{seed}.csv",
                  ["true/predicted", *range(n_classes)],
                  ((k, *row) for k, row in enumerate(confusion)))
    if not overall:
        parser.error(f"No {args.source} results found")
    keys = ("version", "group", "seed")
    write_csv(args.output_dir / "overall.csv",
              (*keys, "accuracy", "macro_precision", "macro_recall", "macro_f1", "n_test"), overall)
    write_csv(args.output_dir / "per_class.csv",
              (*keys, "label", "class", "support", "precision", "recall", "f1"), per_class)
    write_csv(args.output_dir / "key_confusions.csv",
              (*keys, "normal_to_8", "normal_to_7", "8_to_normal", "8_to_5"), progress)
    print(f"Exported metrics for {len(overall)} runs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
