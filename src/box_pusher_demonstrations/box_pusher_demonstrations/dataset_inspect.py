#!/usr/bin/env python3
"""Inspect a recorded episode and report integrity issues."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def inspect_episode(episode_dir: Path) -> Dict[str, Any]:
    metadata_path = episode_dir / "episode_metadata.json"
    samples_path = episode_dir / "samples.jsonl"
    issues = []
    if not metadata_path.exists():
        return {"valid": False, "issues": ["missing episode_metadata.json"]}
    if not samples_path.exists():
        return {"valid": False, "issues": ["missing samples.jsonl"]}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    count = 0
    prev_stamp = -1
    missing_files = 0
    missing_policy = 0
    missing_depth = 0
    ages = []
    for line_no, line in enumerate(samples_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        count += 1
        stamp = int(sample.get("action_stamp_ns", 0))
        if stamp <= prev_stamp:
            issues.append(f"line {line_no}: action timestamps are not strictly increasing")
        prev_stamp = stamp
        obs = sample.get("observation", {})
        if "policy_rgb_96x72" not in obs:
            missing_policy += 1
        left = obs.get("policy_rgb_96x72")
        if left:
            p = episode_dir / left["path"]
            if not p.exists():
                missing_files += 1
        depth = obs.get("depth_mm")
        if depth is None:
            missing_depth += 1
        left_raw = obs.get("left_raw")
        if left_raw and left_raw.get("age_ms") is not None:
            ages.append(float(left_raw["age_ms"]))

    if count == 0:
        issues.append("episode contains no samples")
    if missing_policy:
        issues.append(f"{missing_policy} samples missing policy RGB path")
    if missing_files:
        issues.append(f"{missing_files} policy image files missing")
    if metadata.get("status") != "complete":
        issues.append(f"episode status is {metadata.get('status')!r}, expected 'complete'")

    return {
        "valid": len(issues) == 0,
        "episode_id": metadata.get("episode_id"),
        "condition": metadata.get("condition"),
        "operator_view": metadata.get("operator_view"),
        "status": metadata.get("status"),
        "sample_count": count,
        "samples_without_depth": missing_depth,
        "left_image_age_ms": {
            "mean": sum(ages) / len(ages) if ages else None,
            "max": max(ages) if ages else None,
        },
        "issues": issues,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Path to one recorded episode folder")
    parsed = parser.parse_args(args=args)
    report = inspect_episode(parsed.episode.expanduser().resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
