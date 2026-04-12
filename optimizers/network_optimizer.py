"""
Network-wide departure-time optimization (GA + Tabu).

Không dùng headway cố định: quyết định **có / không** khởi hành tại mỗi
khung giờ rời rạc (slot) trên từng tuyến. Phù hợp đô thị phức tạp nơi
tần suất không đều.

Ma trận: schedule[r, k] = True nếu tuyến r có chuyến xuất bến tại slot k
(0 = 05:00, mỗi slot = SLOT_MINUTES phút).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
from sko.GA import RCGA

from utils.network_sim import AVG_SPEED_KMH, COST_PER_BUS_HOUR, HCMCBusNetwork
from utils.schema_sim import SERVICE_END, SERVICE_START

WAIT_WEIGHT = 0.7
COST_WEIGHT = 0.3
FLEET_PENALTY = 2000.0

# Độ phân giải lịch (phút / slot). 30 phút → 28 slot trong 14h.
SLOT_MINUTES = 30


def _service_span_minutes() -> int:
    return (
        SERVICE_END.hour * 60
        + SERVICE_END.minute
        - SERVICE_START.hour * 60
        - SERVICE_START.minute
    )


def n_slots() -> int:
    return _service_span_minutes() // SLOT_MINUTES


def cum_travel_to_stop(net: HCMCBusNetwork, route_code: str, stop_id: int) -> float:
    """Phút từ lúc xuất bến (trạm đầu) đến khi tới stop_id."""
    ids = net.routes[route_code]["stops"]
    if stop_id not in ids:
        return 0.0
    idx = ids.index(stop_id)
    tot = 0.0
    for i in range(idx):
        a, b = ids[i], ids[i + 1]
        tot += net.graph[a][b]["distance_km"] / AVG_SPEED_KMH * 60.0
    return tot


def _mean_gap_wait(arrivals: List[float], horizon: float) -> float:
    """Ước lượng chờ TB ≈ nửa khoảng cách giữa hai lần xe tới (trong khung horizon)."""
    if not arrivals:
        return 60.0
    A = sorted(t for t in arrivals if 0.0 <= t <= horizon)
    if len(A) == 0:
        return 60.0
    if len(A) == 1:
        return min(35.0, horizon / 6.0)
    padded = [0.0] + A + [horizon]
    gaps = [padded[i + 1] - padded[i] for i in range(len(padded) - 1)]
    return (sum(gaps) / len(gaps)) / 2.0


def _wait_at_stops(
    net: HCMCBusNetwork,
    schedule: np.ndarray,
    slot_minutes: int,
) -> Dict[int, float]:
    """wait[stop_id] = proxy chờ (phút) tại trạm."""
    horizon = float(_service_span_minutes())
    n_s = n_slots()
    codes = net.route_codes
    arrivals_at: Dict[int, List[float]] = {s.id: [] for s in net.stops}

    for ri, code in enumerate(codes):
        row = schedule[ri]
        for k in range(min(n_s, row.shape[0])):
            if not row[k]:
                continue
            t0 = k * slot_minutes
            for sid in net.routes[code]["stops"]:
                arr = t0 + cum_travel_to_stop(net, code, sid)
                if 0 <= arr <= horizon:
                    arrivals_at[sid].append(arr)

    return {
        sid: _mean_gap_wait(arrivals_at[sid], horizon)
        for sid in arrivals_at
    }


def _buses_on_route(
    net: HCMCBusNetwork,
    route_code: str,
    sched_row: np.ndarray,
    slot_minutes: int,
) -> int:
    """Số xe tối đa cần cùng lúc (chồng chuyến theo RTT)."""
    rtt = net.route_rtt[route_code]
    depart_times = [
        k * slot_minutes
        for k in range(len(sched_row))
        if sched_row[k]
    ]
    if not depart_times:
        return 0
    horizon = _service_span_minutes()
    best = 0
    for tau in range(0, horizon + 1, 5):
        n = sum(1 for t in depart_times if t <= tau < t + rtt)
        best = max(best, n)
    return best


def schedule_to_binary(vec: np.ndarray, n_routes: int, n_s: int) -> np.ndarray:
    """Chuyển vector GA [0,1] → ma trận nhị phân."""
    m = vec.reshape(n_routes, n_s)
    return (m > 0.5).astype(bool)


def _ensure_min_departures(sched: np.ndarray, min_per_route: int = 1) -> np.ndarray:
    """Mỗi tuyến ít nhất min_per_route chuyến (slot đầu tiên bật nếu trống)."""
    s = sched.copy()
    n_r, n_s = s.shape
    for r in range(n_r):
        if int(s[r].sum()) < min_per_route and n_s > 0:
            s[r, 0] = True
    return s


def compute_metrics_from_schedule(
    net: HCMCBusNetwork,
    schedule: np.ndarray,
    slot_minutes: int = SLOT_MINUTES,
) -> dict:
    """
    schedule: (n_routes, n_slots) bool — khởi hành tại trạm đầu mỗi tuyến.
    """
    codes = net.route_codes
    assert schedule.shape[0] == len(codes)

    wait_map = _wait_at_stops(net, schedule, slot_minutes)
    total_demand = sum(net.demand[s.id] for s in net.stops)

    def _ww(sid: int) -> float:
        w = getattr(net, "wait_weight_by_stop", None)
        if not w:
            return 1.0
        return float(w.get(sid, 1.0))

    total_wait = sum(
        net.demand[sid] * wait_map[sid] * _ww(sid) for sid in wait_map
    )
    avg_wait = total_wait / max(total_demand, 1e-6)

    per_route_wait: Dict[str, float] = {}
    for code in codes:
        rw = sum(
            net.demand[sid] * wait_map[sid] * _ww(sid)
            for sid in net.routes[code]["stops"]
        )
        per_route_wait[code] = round(rw, 2)

    buses: Dict[str, int] = {}
    total_buses = 0
    total_cost = 0.0
    trips_per_route: Dict[str, int] = {}
    hours = _service_span_minutes() / 60.0

    for ri, code in enumerate(codes):
        row = schedule[ri]
        b = _buses_on_route(net, code, row, slot_minutes)
        buses[code] = b
        total_buses += b
        total_cost += b * COST_PER_BUS_HOUR
        tc = int(row.sum())
        trips_per_route[code] = tc

    trips_per_hour = {
        c: (trips_per_route[c] / hours) if hours > 0 else 0.0 for c in codes
    }

    return {
        "slot_minutes": slot_minutes,
        "trips_per_route": trips_per_route,
        "trips_per_hour": {c: round(trips_per_hour[c], 2) for c in codes},
        "buses": buses,
        "total_buses": total_buses,
        "total_cost": round(total_cost, 1),
        "total_weighted_wait": round(total_wait, 1),
        "avg_wait_min": round(avg_wait, 2),
        "per_route_wait": per_route_wait,
        "objective": round(
            WAIT_WEIGHT * total_wait + COST_WEIGHT * total_cost, 2
        ),
    }


def baseline_schedule_slots(
    net: HCMCBusNetwork,
    every_k_slots: int = 2,
) -> np.ndarray:
    """
    Lịch tham chiếu: mỗi tuyến khởi hành đều đặn mỗi `every_k_slots` slot
    (ví dụ every_k_slots=2, slot 30 phút → mỗi 60 phút một chuyến).
    """
    n_r = len(net.route_codes)
    n_s = n_slots()
    m = np.zeros((n_r, n_s), dtype=bool)
    for r in range(n_r):
        for k in range(0, n_s, every_k_slots):
            m[r, k] = True
    return m


class DepartureScheduleOptimizer:
    """GA + Tabu trên ma trận slot (vector hóa n_routes * n_slots)."""

    def __init__(
        self,
        net: HCMCBusNetwork,
        total_fleet: int,
        ga_pop: int = 40,
        ga_iter: int = 100,
        tabu_iter: int = 120,
        top_k: int = 5,
        slot_minutes: int = SLOT_MINUTES,
        rng_seed: int = 42,
    ) -> None:
        self.net = net
        self.total_fleet = total_fleet
        self.ga_pop = ga_pop
        self.ga_iter = ga_iter
        self.tabu_iter = tabu_iter
        self.top_k = top_k
        self.slot_minutes = slot_minutes
        self.n_r = len(net.route_codes)
        self.n_s = n_slots()
        self.n_dim = self.n_r * self.n_s
        self.rng = np.random.default_rng(rng_seed)

    def _objective(self, vec: np.ndarray) -> float:
        sched = schedule_to_binary(vec, self.n_r, self.n_s)
        sched = _ensure_min_departures(sched)
        m = compute_metrics_from_schedule(self.net, sched, self.slot_minutes)
        over = max(0, m["total_buses"] - self.total_fleet)
        return (
            WAIT_WEIGHT * m["total_weighted_wait"]
            + COST_WEIGHT * m["total_cost"]
            + FLEET_PENALTY * over
        )

    def _run_ga(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        ga = RCGA(
            func=self._objective,
            n_dim=self.n_dim,
            size_pop=self.ga_pop,
            max_iter=self.ga_iter,
            lb=[0.0] * self.n_dim,
            ub=[1.0] * self.n_dim,
        )
        best_x, _ = ga.run()
        X_pop = ga.chrom2x(ga.Chrom)
        fits = np.array([self._objective(X_pop[i]) for i in range(len(X_pop))])
        idx = np.argsort(fits)[: self.top_k]
        elites = [X_pop[i].copy() for i in idx]
        return best_x, elites

    def _tabu_search(self, init: np.ndarray) -> Tuple[np.ndarray, float]:
        current = np.clip(init.ravel().copy(), 0.0, 1.0)
        best = current.copy()
        best_cost = self._objective(best)
        tabu: List[int] = []

        for _ in range(self.tabu_iter):
            candidates: List[Tuple[int, np.ndarray, float]] = []
            for i in range(self.n_dim):
                if i in tabu:
                    continue
                n = current.copy()
                n[i] = 1.0 - n[i]
                candidates.append((i, n, self._objective(n)))

            if not candidates:
                tabu.clear()
                continue

            candidates.sort(key=lambda t: t[2])
            chosen_i, chosen_n, chosen_c = candidates[0]
            current = chosen_n
            tabu.append(chosen_i)
            if len(tabu) > self.n_dim // 4:
                tabu.pop(0)

            if chosen_c < best_cost:
                best = current.copy()
                best_cost = chosen_c

        return best, best_cost

    def optimize(self) -> np.ndarray:
        print("[network_opt] Phase 1: Genetic Algorithm (departure slots) …")
        ga_best, elites = self._run_ga()

        print("[network_opt] Phase 2: Tabu Search (local flips) …")
        best_vec = np.clip(ga_best.ravel().copy(), 0.0, 1.0)
        best_cost = self._objective(best_vec)

        for elite in elites:
            v, cost = self._tabu_search(elite.ravel())
            if cost < best_cost:
                best_vec = v
                best_cost = cost

        sched = schedule_to_binary(best_vec, self.n_r, self.n_s)
        sched = _ensure_min_departures(sched)
        print(f"[network_opt] Best objective = {best_cost:.2f}")
        return sched


# ---------------------------------------------------------------------------
# So sánh & bảng lịch
# ---------------------------------------------------------------------------
def compare(before: dict, after: dict) -> dict:
    def _pct(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return round((new - old) / abs(old) * 100, 1)

    return {
        "wait": {
            "before": before["avg_wait_min"],
            "after": after["avg_wait_min"],
            "pct": _pct(before["avg_wait_min"], after["avg_wait_min"]),
        },
        "cost": {
            "before": before["total_cost"],
            "after": after["total_cost"],
            "pct": _pct(before["total_cost"], after["total_cost"]),
        },
        "buses": {
            "before": before["total_buses"],
            "after": after["total_buses"],
            "pct": _pct(before["total_buses"], after["total_buses"]),
        },
        "objective": {
            "before": before["objective"],
            "after": after["objective"],
            "pct": _pct(before["objective"], after["objective"]),
        },
    }


def build_departure_rows_from_schedule(
    net: HCMCBusNetwork,
    schedule: np.ndarray,
    slot_minutes: int = SLOT_MINUTES,
    reference_date: date | None = None,
) -> List[Dict[str, Any]]:
    """Một dòng = một chuyến (xuất bến tại trạm đầu). Không có cột headway."""
    ref = reference_date or date(2025, 9, 1)
    day0 = datetime.combine(ref, SERVICE_START)
    horizon = _service_span_minutes()
    codes = net.route_codes
    rows: List[Dict[str, Any]] = []

    for ri, code in enumerate(codes):
        rdef = net.routes[code]
        first_id = rdef["stops"][0]
        first_name = net.stop_map[first_id].name
        rtt = net.route_rtt[code]
        row = schedule[ri]
        seq = 0
        prev_min: float | None = None
        for k in range(len(row)):
            if not row[k]:
                continue
            tmin = k * slot_minutes
            if tmin >= horizon:
                break
            t = day0 + timedelta(minutes=tmin)
            seq += 1
            gap = None if prev_min is None else (tmin - prev_min)
            prev_min = float(tmin)
            rows.append(
                {
                    "route_code": code,
                    "route_name": rdef["name"],
                    "first_stop_id": first_id,
                    "first_stop_name": first_name,
                    "trip_seq": seq,
                    "departure_time": t.strftime("%H:%M"),
                    "minutes_from_open": tmin,
                    "gap_from_prev_min": round(gap, 1) if gap is not None else None,
                    "rtt_min": rtt,
                    "buses_on_route": _buses_on_route(
                        net, code, row, slot_minutes
                    ),
                }
            )

    rows.sort(key=lambda r: (r["departure_time"], r["route_code"]))
    return rows
