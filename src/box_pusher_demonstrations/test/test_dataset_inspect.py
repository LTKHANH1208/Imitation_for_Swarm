import json
from pathlib import Path

from box_pusher_demonstrations.dataset_inspect import inspect_episode


def test_inspector_accepts_minimal_complete_episode(tmp_path: Path):
    (tmp_path / "policy_rgb_96x72").mkdir()
    (tmp_path / "policy_rgb_96x72" / "000000.png").write_bytes(b"fake-png")
    metadata = {
        "episode_id": "single_test",
        "condition": "single",
        "operator_view": "onboard",
        "status": "complete",
    }
    (tmp_path / "episode_metadata.json").write_text(json.dumps(metadata))
    sample = {
        "action_stamp_ns": 100,
        "observation": {
            "policy_rgb_96x72": {"path": "policy_rgb_96x72/000000.png"},
            "left_raw": {"age_ms": 20.0},
        },
    }
    (tmp_path / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    report = inspect_episode(tmp_path)
    assert report["valid"] is True
    assert report["sample_count"] == 1


def test_inspector_flags_missing_image(tmp_path: Path):
    (tmp_path / "episode_metadata.json").write_text(json.dumps({"status": "complete"}))
    sample = {
        "action_stamp_ns": 100,
        "observation": {"policy_rgb_96x72": {"path": "missing.png"}},
    }
    (tmp_path / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    report = inspect_episode(tmp_path)
    assert report["valid"] is False
    assert any("image files missing" in issue for issue in report["issues"])
