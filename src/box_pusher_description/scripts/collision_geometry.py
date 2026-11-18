#!/usr/bin/env python3
"""Name parsing and collision-pair classification for Gazebo contact messages."""
from __future__ import annotations
import re
from typing import Tuple

_BOX_RE = re.compile(r"orange_box_\d+")
_WALL_RE = re.compile(r"wall_(?:north|south|east|west)")
_ROBOT_RE = re.compile(r"(?:box_pusher|robot_?\d+|robot[\w-]*)")


def entity_name(entity) -> str:
    """Extract the Gazebo scoped collision name from ros_gz_interfaces/Entity."""
    value = getattr(entity, 'name', '')
    return str(value).strip() if value else ''


def model_token(scoped_name: str) -> str:
    """Return the most useful model token from a scoped Gazebo collision name."""
    text = str(scoped_name or '').replace('/', '::')
    for pattern in (_BOX_RE, _WALL_RE):
        match = pattern.search(text)
        if match:
            return match.group(0)
    # Robot model names in this project are box_pusher (single robot) and may
    # become box_pusher_robot_XX in future multi-robot launches.
    match = re.search(r"box_pusher(?:_[A-Za-z0-9-]+)?", text)
    if match:
        return match.group(0)
    # Generic future naming convention, e.g. robot_01.
    match = re.search(r"robot_?\d+", text)
    if match:
        return match.group(0)
    return text.split('::')[0] if text else 'unknown'


def classify_pair(name_a: str, name_b: str) -> str:
    """Classify a pair into robot/box/wall interaction categories."""
    a, b = model_token(name_a), model_token(name_b)
    a_robot = a.startswith('box_pusher') or re.fullmatch(r'robot_?\d+', a) is not None
    b_robot = b.startswith('box_pusher') or re.fullmatch(r'robot_?\d+', b) is not None
    a_box, b_box = _BOX_RE.fullmatch(a) is not None, _BOX_RE.fullmatch(b) is not None
    a_wall, b_wall = _WALL_RE.fullmatch(a) is not None, _WALL_RE.fullmatch(b) is not None
    if a_robot and b_robot:
        return 'robot_robot'
    if (a_robot and b_box) or (b_robot and a_box):
        return 'robot_box'
    if a_box and b_box:
        return 'box_box'
    if (a_robot and b_wall) or (b_robot and a_wall):
        return 'robot_wall'
    if (a_box and b_wall) or (b_box and a_wall):
        return 'box_wall'
    return 'other'


def is_expected_ground_contact(name_a: str, name_b: str) -> bool:
    """Ignore intended support contacts between boxes and floor/carpet surfaces."""
    names = (str(name_a or '').lower(), str(name_b or '').lower())
    support_tokens = ('arena_floor_', 'carpet', 'pit_floor', 'pit_bottom', 'ground_plane')
    return any(any(token in name for token in support_tokens) for name in names)
