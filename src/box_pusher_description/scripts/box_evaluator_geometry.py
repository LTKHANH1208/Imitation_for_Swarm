#!/usr/bin/env python3
"""Pure geometry rules for the V2 box-collection evaluator.

Coordinates are in the Gazebo world frame. The target carpet is a 1 x 1 m
square with a 0.5 x 0.5 m pit opening removed from its centre.
"""
from __future__ import annotations

from typing import Literal

BoxState = Literal["ACTIVE", "DELIVERED_CARPET", "DELIVERED_PIT"]


def classify_box_center(
    x: float,
    y: float,
    z: float,
    *,
    carpet_half_extent_m: float = 0.50,
    carpet_inner_half_extent_m: float = 0.25,
    pit_half_extent_m: float = 0.25,
    pit_center_x_m: float = 0.0,
    pit_center_y_m: float = 0.0,
    pit_fallen_z_threshold_m: float = -0.02,
) -> BoxState:
    """Classify the box centre against carpet ring and pit volume.

    A box is delivered when its centre enters the carpet ring, or when its
    centre is inside the pit opening and has fallen below the opening plane.
    Delivery is latched by the ROS node; this function is stateless.
    """
    dx = abs(float(x) - pit_center_x_m)
    dy = abs(float(y) - pit_center_y_m)

    in_pit_xy = dx <= pit_half_extent_m and dy <= pit_half_extent_m
    if in_pit_xy and float(z) <= pit_fallen_z_threshold_m:
        return "DELIVERED_PIT"

    in_outer_carpet = (
        abs(float(x) - pit_center_x_m) <= carpet_half_extent_m
        and abs(float(y) - pit_center_y_m) <= carpet_half_extent_m
    )
    outside_opening = (
        abs(float(x) - pit_center_x_m) >= carpet_inner_half_extent_m
        or abs(float(y) - pit_center_y_m) >= carpet_inner_half_extent_m
    )
    if in_outer_carpet and outside_opening:
        return "DELIVERED_CARPET"

    return "ACTIVE"
