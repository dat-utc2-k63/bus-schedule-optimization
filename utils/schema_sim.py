"""
utils/schema_sim.py — Normalized 8-table schema simulator
==========================================================
Matches the canonical database schema:

  1. Route_Master          — route-level info
  2. Stops                 — all bus stops
  3. Network_Segments      — graph edges (route × stop pair)
  4. Buses                 — fleet catalogue
  5. Trips                 — each bus trip (one departure per route)
  6. Stop_Times            — planned arrival / departure per stop per trip
  7. SpatioTemporal_Snapshots — long-format time-series for GNN training
  8. Operation_Logs        — actual operation (delays, boarding counts)

All tables are written as CSV to  data/<table_name>.csv
The GNN training pipeline reads SpatioTemporal_Snapshots via
``build_feature_tensor_from_snapshots()``.
"""

from __future__ import annotations

import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.network_sim import AVG_SPEED_KMH, HCMCBusNetwork
from utils.domain import (
    DEMAND_RANGE,
    MORNING_RUSH,
    EVENING_RUSH,
    RAIN_PROBABILITY,
    RUSH_HOUR_MULTIPLIER,
    RAIN_DEMAND_MULTIPLIER,
    RAIN_SPEED_PENALTY,
)

SERVICE_START = time(5, 0)
SERVICE_END = time(19, 0)
_SERVICE_SPAN_MIN = (
    SERVICE_END.hour * 60 + SERVICE_END.minute
    - SERVICE_START.hour * 60 - SERVICE_START.minute
)


# ---------------------------------------------------------------------------
# Schema-level simulator
# ---------------------------------------------------------------------------
class SchemaSimulator:
    """
    Generates all 8 normalized tables for the HCMC bus network.

    Parameters
    ----------
    net : HCMCBusNetwork
        The multi-route network (13 stops, 4 routes).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, net: HCMCBusNetwork, seed: int = 42) -> None:
        self.net = net
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Route_Master
    # ------------------------------------------------------------------
    def build_route_master(self) -> pd.DataFrame:
        """
        PK: route_code.
        Columns: route_name, total_distance_km, service_start, service_end.
        """
        rows = []
        for code in self.net.route_codes:
            rdef = self.net.routes[code]
            ids = rdef["stops"]
            total_km = sum(
                self.net.graph[a][b]["distance_km"]
                for a, b in zip(ids[:-1], ids[1:])
            )
            rows.append(
                {
                    "route_code": code,
                    "route_name": rdef["name"],
                    "total_distance_km": round(total_km, 3),
                    "service_start": SERVICE_START.isoformat(timespec="minutes"),
                    "service_end": SERVICE_END.isoformat(timespec="minutes"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2. Stops
    # ------------------------------------------------------------------
    def build_stops(self) -> pd.DataFrame:
        """
        PK: stop_id.
        Columns: stop_name, stop_category, latitude, longitude.
        """
        rows = [
            {
                "stop_id": s.id,
                "stop_name": s.name,
                "stop_category": s.category,
                "latitude": s.lat,
                "longitude": s.lon,
            }
            for s in self.net.stops
        ]
        return pd.DataFrame(rows).sort_values("stop_id").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Network_Segments
    # ------------------------------------------------------------------
    def build_network_segments(self) -> pd.DataFrame:
        """
        PK: segment_id.
        FK: route_code → Route_Master, from_stop_id / to_stop_id → Stops.
        Columns: distance_km, sequence, avg_travel_time_min.

        Note: different routes can share the same physical stop pair;
        they each get their own segment row.
        """
        rows = []
        for code in self.net.route_codes:
            rdef = self.net.routes[code]
            ids = rdef["stops"]
            for seq, (a, b) in enumerate(zip(ids[:-1], ids[1:]), start=1):
                dist = self.net.graph[a][b]["distance_km"]
                avg_tt = round((dist / AVG_SPEED_KMH) * 60.0, 1)
                rows.append(
                    {
                        "segment_id": f"SEG-{code}-{seq:02d}",
                        "route_code": code,
                        "from_stop_id": a,
                        "to_stop_id": b,
                        "distance_km": dist,
                        "sequence": seq,
                        "avg_travel_time_min": avg_tt,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 8. Buses  (built early; other tables reference bus_id)
    # ------------------------------------------------------------------
    def build_buses(self, n_buses: int = 12) -> pd.DataFrame:
        """
        PK: bus_id.
        Columns: license_plate, capacity, type.
        """
        bus_types = ["standard", "articulated", "minibus"]
        capacities = {"standard": 80, "articulated": 120, "minibus": 30}
        weights = [0.60, 0.30, 0.10]
        rows = []
        for i in range(1, n_buses + 1):
            btype = str(self.rng.choice(bus_types, p=weights))
            a = int(self.rng.integers(101, 999))
            b = int(self.rng.integers(10, 99))
            rows.append(
                {
                    "bus_id": f"BUS-{i:03d}",
                    "license_plate": f"51B-{a:03d}.{b:02d}",
                    "capacity": capacities[btype],
                    "type": btype,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Trips
    # ------------------------------------------------------------------
    def build_trips(
        self,
        buses_df: pd.DataFrame,
        n_days: int = 7,
        trips_per_bus_per_day: int = 6,
        base_date: str = "2025-09-01",
    ) -> pd.DataFrame:
        """
        PK: trip_id.
        FK: route_code → Route_Master, bus_id → Buses.
        Columns: direction, start_time, end_time.

        Buses are assigned to routes round-robin; start times are spread
        across the service window with small random offsets.
        """
        n_routes = len(self.net.route_codes)
        slot_min = _SERVICE_SPAN_MIN / max(trips_per_bus_per_day, 1)
        rows = []

        for day_i in range(n_days):
            trip_date = (
                datetime.fromisoformat(base_date).date() + timedelta(days=day_i)
            )
            for bus_idx, (_, bus) in enumerate(buses_df.iterrows()):
                route_code = self.net.route_codes[bus_idx % n_routes]
                rtt_min = self.net.route_rtt[route_code]

                for k in range(trips_per_bus_per_day):
                    offset = slot_min * k + float(
                        self.rng.uniform(0.0, slot_min * 0.30)
                    )
                    start_dt = datetime.combine(
                        trip_date, SERVICE_START
                    ) + timedelta(minutes=offset)
                    end_dt = start_dt + timedelta(minutes=rtt_min)
                    rows.append(
                        {
                            "trip_id": (
                                f"TRIP-{trip_date.isoformat()}"
                                f"-{bus['bus_id']}-{k+1:02d}"
                            ),
                            "route_code": route_code,
                            "bus_id": bus["bus_id"],
                            "direction": "outbound",
                            "start_time": start_dt.isoformat(timespec="seconds"),
                            "end_time": end_dt.isoformat(timespec="seconds"),
                        }
                    )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Stop_Times
    # ------------------------------------------------------------------
    def build_stop_times(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite PK (trip_id, stop_id).
        FK: trip_id → Trips, stop_id → Stops.
        Columns: arrival_time, departure_time, stop_sequence.

        Travel time between stops = distance / AVG_SPEED_KMH.
        Dwell time ~ Uniform(20, 60) seconds.
        """
        rows = []
        for _, trip in trips_df.iterrows():
            route_code = trip["route_code"]
            stop_ids = self.net.routes[route_code]["stops"]
            current = datetime.fromisoformat(trip["start_time"])

            for seq, sid in enumerate(stop_ids, start=1):
                # Travel leg from previous stop
                if seq > 1:
                    prev = stop_ids[seq - 2]
                    dist = self.net.graph[prev][sid]["distance_km"]
                    travel_min = (dist / AVG_SPEED_KMH) * 60.0
                    current += timedelta(minutes=travel_min)

                arr = current
                dwell_sec = float(self.rng.uniform(20.0, 60.0))
                dep = arr + timedelta(seconds=dwell_sec)
                rows.append(
                    {
                        "trip_id": trip["trip_id"],
                        "stop_id": sid,
                        "arrival_time": arr.isoformat(timespec="seconds"),
                        "departure_time": dep.isoformat(timespec="seconds"),
                        "stop_sequence": seq,
                    }
                )
                current = dep

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 6. SpatioTemporal_Snapshots
    # ------------------------------------------------------------------
    def build_spatiotemporal_snapshots(
        self,
        n_days: int = 7,
        freq_minutes: int = 15,
        base_date: str = "2025-09-01",
    ) -> pd.DataFrame:
        """
        Composite PK: (timestamp, stop_id, route_code).
        Columns: demand_value, traffic_speed, is_rain, day_of_week, hour_of_day.

        One row per (timestamp, stop_id, route_code) combination.
        Stops not served by a route are excluded (avoids artificial zeroes).

        VN-specific patterns
        - Rush hours 6-9h & 16-19h → demand ×1.8, speed ×0.65
        - Rain (p=35%) → demand ×1.3, speed ×0.6
        """
        slots_per_day = (24 * 60) // freq_minutes
        total_slots = n_days * slots_per_day
        timestamps = pd.date_range(
            start=base_date, periods=total_slots, freq=f"{freq_minutes}min"
        )
        hours = timestamps.hour + timestamps.minute / 60.0
        is_rain_arr: np.ndarray = self.rng.random(total_slots) < RAIN_PROBABILITY

        rows = []
        for t_idx in range(total_slots):
            ts = timestamps[t_idx]
            h = float(hours[t_idx])
            rain = bool(is_rain_arr[t_idx])
            rush = (
                MORNING_RUSH[0] <= h < MORNING_RUSH[1]
                or EVENING_RUSH[0] <= h < EVENING_RUSH[1]
            )

            for code in self.net.route_codes:
                for sid in self.net.routes[code]["stops"]:
                    s = self.net.stop_map[sid]
                    lo, hi = DEMAND_RANGE[s.category]

                    demand = float(self.rng.uniform(lo, hi))
                    speed = float(self.rng.uniform(15.0, 35.0))

                    if rush:
                        demand *= RUSH_HOUR_MULTIPLIER
                        speed *= 0.65
                    if rain:
                        demand *= RAIN_DEMAND_MULTIPLIER
                        speed *= RAIN_SPEED_PENALTY

                    demand += float(self.rng.normal(0.0, 2.0))
                    demand = round(max(0.0, demand), 1)
                    speed = round(min(60.0, max(5.0, speed)), 1)

                    rows.append(
                        {
                            "timestamp": ts,
                            "stop_id": sid,
                            "route_code": code,
                            "demand_value": demand,
                            "traffic_speed": speed,
                            "is_rain": rain,
                            "day_of_week": int(ts.day_of_week),
                            "hour_of_day": int(ts.hour),
                        }
                    )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 7. Operation_Logs
    # ------------------------------------------------------------------
    def build_operation_logs(
        self,
        trips_df: pd.DataFrame,
        stop_times_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Composite PK (trip_id, stop_id).
        FK: trip_id → Trips, bus_id → Buses, stop_id → Stops.
        Columns: actual_arrival, actual_departure, boarding_count, dwell_time_sec.

        Actual times = planned ± delay (Normal(0, 3) min).
        Boarding count depends on stop category and rush hours.
        """
        trip_info = (
            trips_df.set_index("trip_id")[["bus_id", "route_code"]]
            .to_dict("index")
        )
        rows = []

        for _, st in stop_times_df.iterrows():
            info = trip_info[st["trip_id"]]
            arr_planned = datetime.fromisoformat(st["arrival_time"])
            dep_planned = datetime.fromisoformat(st["departure_time"])

            delay_min = float(self.rng.normal(0.0, 3.0))
            actual_arr = arr_planned + timedelta(minutes=delay_min)

            sid = int(st["stop_id"])
            s = self.net.stop_map[sid]
            h = actual_arr.hour + actual_arr.minute / 60.0
            rush = (
                MORNING_RUSH[0] <= h < MORNING_RUSH[1]
                or EVENING_RUSH[0] <= h < EVENING_RUSH[1]
            )
            p_board = min(
                0.92,
                0.40
                + 0.10 * (s.category == "hub")
                + 0.15 * rush,
            )
            boarding = (
                int(self.rng.integers(1, 16))
                if self.rng.random() < p_board
                else 0
            )

            planned_dwell = (dep_planned - arr_planned).total_seconds()
            dwell = max(8.0, planned_dwell + float(self.rng.normal(0.0, 10.0)))
            actual_dep = actual_arr + timedelta(seconds=dwell)

            rows.append(
                {
                    "trip_id": st["trip_id"],
                    "bus_id": info["bus_id"],
                    "route_code": info["route_code"],
                    "stop_id": sid,
                    "actual_arrival": actual_arr.isoformat(timespec="seconds"),
                    "actual_departure": actual_dep.isoformat(timespec="seconds"),
                    "boarding_count": boarding,
                    "dwell_time_sec": round(dwell, 1),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # generate_all  — main entry point
    # ------------------------------------------------------------------
    def generate_all(
        self,
        n_days: int = 7,
        n_buses: int = 12,
        trips_per_bus_per_day: int = 6,
        freq_minutes: int = 15,
        base_date: str = "2025-09-01",
        out_dir: str = "data",
    ) -> Dict[str, pd.DataFrame]:
        """
        Build and persist all 8 tables.  Returns a dict of DataFrames.
        """
        os.makedirs(out_dir, exist_ok=True)

        print("[schema_sim] Building Route_Master …")
        route_master = self.build_route_master()

        print("[schema_sim] Building Stops …")
        stops = self.build_stops()

        print("[schema_sim] Building Network_Segments …")
        segments = self.build_network_segments()

        print("[schema_sim] Building Buses …")
        buses = self.build_buses(n_buses)

        print("[schema_sim] Building Trips …")
        trips = self.build_trips(buses, n_days, trips_per_bus_per_day, base_date)

        print("[schema_sim] Building Stop_Times …")
        stop_times = self.build_stop_times(trips)

        print(f"[schema_sim] Building SpatioTemporal_Snapshots ({n_days} days) …")
        snapshots = self.build_spatiotemporal_snapshots(
            n_days, freq_minutes, base_date
        )

        print("[schema_sim] Building Operation_Logs …")
        op_logs = self.build_operation_logs(trips, stop_times)

        tables: Dict[str, pd.DataFrame] = {
            "route_master": route_master,
            "stops": stops,
            "network_segments": segments,
            "buses": buses,
            "trips": trips,
            "stop_times": stop_times,
            "spatiotemporal_snapshots": snapshots,
            "operation_logs": op_logs,
        }

        for name, df in tables.items():
            path = os.path.join(out_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"  wrote {path}  ({len(df):,} rows)")

        return tables


# ---------------------------------------------------------------------------
# GNN tensor builder  (reads from SpatioTemporal_Snapshots long format)
# ---------------------------------------------------------------------------
def build_feature_tensor_from_snapshots(
    snapshots: pd.DataFrame,
    stop_ids: List[int],
    window: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SpatioTemporal_Snapshots (long format) to GNN-ready tensors.

    Strategy: aggregate demand / speed across all routes serving a stop
    (mean per timestamp × stop), yielding one signal per node irrespective
    of how many routes pass through it.

    Parameters
    ----------
    snapshots : DataFrame
        Must contain columns: timestamp, stop_id, demand_value,
        traffic_speed, is_rain.
    stop_ids : list[int]
        Ordered list of stop IDs matching the adjacency matrix row order.
    window : int
        Number of consecutive time-steps used as input context.

    Returns
    -------
    X : ndarray  (samples, window, N, 3)   features: [demand, speed, rain]
    Y : ndarray  (samples, N)              target demand at t+window
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(snapshots["timestamp"]):
        snapshots = snapshots.copy()
        snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"])

    # Aggregate per (timestamp, stop_id) — mean across routes
    agg = (
        snapshots
        .groupby(["timestamp", "stop_id"], sort=True, as_index=False)
        .agg(
            demand_value=("demand_value", "mean"),
            traffic_speed=("traffic_speed", "mean"),
            is_rain=("is_rain", "first"),
        )
    )

    timestamps = sorted(agg["timestamp"].unique())
    T = len(timestamps)
    N = len(stop_ids)

    demand_mat = np.zeros((T, N), dtype=np.float32)
    speed_mat = np.zeros((T, N), dtype=np.float32)
    rain_arr = np.zeros(T, dtype=np.float32)

    ts_idx = {ts: i for i, ts in enumerate(timestamps)}
    sid_idx = {sid: j for j, sid in enumerate(stop_ids)}

    for _, row in agg.iterrows():
        ti = ts_idx.get(row["timestamp"])
        ni = sid_idx.get(int(row["stop_id"]))
        if ti is not None and ni is not None:
            demand_mat[ti, ni] = float(row["demand_value"])
            speed_mat[ti, ni] = float(row["traffic_speed"])
            rain_arr[ti] = float(row["is_rain"])

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for t in range(T - window):
        demand_win = demand_mat[t : t + window]  # (window, N)
        speed_win = speed_mat[t : t + window]    # (window, N)
        feats = np.stack([demand_win, speed_win], axis=-1)  # (window, N, 2)

        rain_feat = np.tile(
            rain_arr[t : t + window].reshape(-1, 1, 1), (1, N, 1)
        )  # (window, N, 1)
        feats = np.concatenate([feats, rain_feat], axis=-1)  # (window, N, 3)

        X_list.append(feats)
        Y_list.append(demand_mat[t + window])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y
