"""Shared domain types and constants for the bus network simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Vietnam / HCMC temporal patterns
MORNING_RUSH = (6, 9)
EVENING_RUSH = (16, 19)
RAIN_PROBABILITY = 0.35

DEMAND_RANGE: Dict[str, Tuple[int, int]] = {
    "hub": (40, 80),
    "residential": (20, 50),
    "commercial": (25, 60),
    "school": (15, 45),
    "terminal": (30, 70),
}

RUSH_HOUR_MULTIPLIER = 1.8
RAIN_DEMAND_MULTIPLIER = 1.3
RAIN_SPEED_PENALTY = 0.6


@dataclass
class BusStop:
    id: int
    name: str
    category: str
    lat: float
    lon: float
