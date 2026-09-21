"""Verify that Module 2 runs on the frozen PBridge-B baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "module2_baseline.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def module1_hashes() -> dict[str, str]:
    manifest = baseline_manifest()
    return {
        name: sha256(ROOT / "module1_ppr" / name)
        for name in manifest["module1_sha256"]
    }


def validate_fixed_module1(checkpoint_path: Path) -> dict:
    manifest = baseline_manifest()
    expected_checkpoint = (ROOT / manifest["checkpoint"]).resolve()
    actual_checkpoint = checkpoint_path.resolve()
    if actual_checkpoint != expected_checkpoint:
        raise RuntimeError(
            "Module 2 is locked to the formal PBridge-B checkpoint: "
            f"{expected_checkpoint}"
        )
    actual_checkpoint_hash = sha256(actual_checkpoint)
    if actual_checkpoint_hash != manifest["checkpoint_sha256"]:
        raise RuntimeError(
            "The formal PBridge-B checkpoint changed; Module 2 must not "
            "modify or replace Module 1"
        )
    actual_sources = module1_hashes()
    if actual_sources != manifest["module1_sha256"]:
        raise RuntimeError(
            "Module 1 source changed after the Module 2 baseline was fixed"
        )
    return manifest
