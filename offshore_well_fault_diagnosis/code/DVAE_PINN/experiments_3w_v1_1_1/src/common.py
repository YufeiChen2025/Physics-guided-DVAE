"""All generated artifacts are rooted in this isolated experiment directory."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ["DVAE_V111_WORK"]).resolve() / "data" if "DVAE_V111_WORK" in os.environ else CODE
WORKSPACE = CODE.parents[1]
DATA = WORKSPACE / "TRANS" / "3w_dataset_1.1.1"
FEATURES = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL", "T-JUS-CKGL", "QGL"]
CLASS_NAMES = ["Normal", "BSW", "DHSV", "Severe slugging", "Flow instability", "Rapid productivity loss", "PCK quick", "PCK scaling", "Hydrate production line"]
SOURCES = ["real", "simulated", "drawn"]


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_of(name):
    if name.startswith("WELL"):
        return "real"
    if name.startswith("SIMULATED"):
        return "simulated"
    if name.startswith("DRAWN"):
        return "drawn"
    raise ValueError(f"Unknown instance source: {name}")
