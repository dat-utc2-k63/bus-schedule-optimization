"""
GNN-aligned graph propagation for urban congestion stress
=========================================================
Uses the **same** stop graph as the GCN: local stress (demand × slowness)
is diffused along edges to approximate **spatial spillover** of congestion
(hubs affecting neighbours), analogous to message passing without re-running TF.

References the hybrid pipeline: predicted demand from Spatio-Temporal GNN +
observed mean traffic_speed per stop from snapshots.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np


def row_normalize_adjacency(adj: np.ndarray, add_self: bool = True) -> np.ndarray:
    """Row-stochastic matrix P = D^{-1} (A + I) for random-walk diffusion."""
    a = np.asarray(adj, dtype=np.float64)
    if add_self:
        n = a.shape[0]
        a = a + np.eye(n, dtype=np.float64)
    d = a.sum(axis=1, keepdims=True)
    d = np.maximum(d, 1e-9)
    return a / d


def local_congestion_stress(
    demand_pred: np.ndarray,
    speed_mean: np.ndarray,
    ref_speed_kmh: float = 25.0,
) -> np.ndarray:
    """
    Scalar stress per node: high when demand is high and speed is low.

    slowness in [0, 1] = clip((ref - v) / ref, 0, 1).
    """
    d = np.maximum(np.asarray(demand_pred, dtype=np.float64), 0.0)
    v = np.asarray(speed_mean, dtype=np.float64)
    slow = np.clip((ref_speed_kmh - v) / ref_speed_kmh, 0.0, 1.0)
    # Baseline term so isolated slow links still matter without vanishing product
    return d * (0.35 + 0.65 * slow)


def propagate_stress(
    adj: np.ndarray,
    local: np.ndarray,
    steps: int = 3,
    mix: float = 0.55,
) -> np.ndarray:
    """
    K-step diffusion: h <- mix * (P @ h) + (1-mix) * h0, same init h0 = local.

    Models congestion **propagating** along the road network (e.g. jam from
    a hub affecting adjacent segments).
    """
    p = row_normalize_adjacency(adj, add_self=True)
    h = np.asarray(local, dtype=np.float64).copy()
    h0 = h.copy()
    for _ in range(max(1, steps)):
        h = mix * (p @ h) + (1.0 - mix) * h0
    return h


def stress_to_wait_weights(
    stress_propagated: np.ndarray,
    clip: Tuple[float, float] = (0.65, 1.85),
) -> np.ndarray:
    """Normalize to mean 1.0, then clip for stable optimizer scaling."""
    s = np.asarray(stress_propagated, dtype=np.float64)
    s = np.maximum(s, 1e-9)
    w = s / np.mean(s)
    lo, hi = clip
    return np.clip(w, lo, hi)


def build_wait_weights_for_stops(
    stop_ids: List[int],
    adj: np.ndarray,
    demand_pred: np.ndarray,
    speed_mean: np.ndarray,
    diffusion_steps: int = 3,
    diffusion_mix: float = 0.55,
    ref_speed_kmh: float = 25.0,
) -> Dict[int, float]:
    """End-to-end: numpy vectors ordered like `stop_ids` → dict stop_id → weight."""
    loc = local_congestion_stress(demand_pred, speed_mean, ref_speed_kmh)
    prop = propagate_stress(adj, loc, steps=diffusion_steps, mix=diffusion_mix)
    w = stress_to_wait_weights(prop)
    return {int(sid): float(wi) for sid, wi in zip(stop_ids, w)}


def save_gnn_auxiliary(
    path: str,
    wait_weight_by_stop: Dict[int, float],
    meta: Dict[str, object],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "wait_weight_by_stop": {str(k): v for k, v in wait_weight_by_stop.items()},
        "meta": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_gnn_auxiliary(path: str) -> Tuple[Dict[int, float], Dict[str, object]] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    w = {int(k): float(v) for k, v in raw.get("wait_weight_by_stop", {}).items()}
    meta = raw.get("meta") or {}
    return w, meta
