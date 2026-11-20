from pathlib import Path


def test_required_package_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'package.xml').exists()
    assert (root / 'setup.py').exists()
    assert (root / 'launch' / 'recording.launch.py').exists()
    assert (root / 'box_pusher_demonstrations' / 'keyboard_teleop_node.py').exists()
    assert (root / 'box_pusher_demonstrations' / 'demonstration_recorder_node.py').exists()
