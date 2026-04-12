"""
Multi-route HCMC bus network: graph, demand, and Cytoscape export.

Topology (trạm, thứ tự trạm trên tuyến, màu, tên tuyến) được đọc từ CSV trong
``data/`` — không hard-code lộ trình trong code.

Bắt buộc:
  - ``stops.csv``          — danh sách trạm
  - ``route_master.csv``   — mã tuyến, tên, màu, khung phục vụ
  - ``route_stops.csv``    — lộ trình: (route_code, sequence, stop_id)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from utils.domain import BusStop, DEMAND_RANGE

# Operational constants (không thuộc topology)
AVG_SPEED_KMH = 18.0
AVG_DWELL_MIN = 1.0
TURNAROUND_MIN = 5.0
COST_PER_BUS_HOUR = 45.0  # VND-equivalent abstract unit


def default_data_dir() -> str:
    """Thư mục ``data/`` cạnh thư mục gốc project."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    )


def _norm_route_code(raw: Any) -> str:
    """Chuẩn hóa mã tuyến (vd. 1 → '01')."""
    s = str(raw).strip()
    if s.isdigit():
        return s.zfill(2)
    return s


def load_network_from_csv(data_dir: str) -> Tuple[List[BusStop], Dict[str, Dict[str, Any]], List[str]]:
    """
    Đọc stops + route_master + route_stops → danh sách BusStop và dict routes.

    Mỗi phần tử ``routes[code]`` có: name, color, stops (list[int] theo thứ tự chạy).
    """
    stops_path = os.path.join(data_dir, "stops.csv")
    rm_path = os.path.join(data_dir, "route_master.csv")
    rs_path = os.path.join(data_dir, "route_stops.csv")

    for p in (stops_path, rm_path, rs_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Thiếu file dữ liệu: {p}. Cần stops.csv, route_master.csv, route_stops.csv."
            )

    df_stops = pd.read_csv(stops_path)
    # Giữ mã tuyến dạng chuỗi (vd. "01" không thành số 1)
    df_rm = pd.read_csv(rm_path, dtype={"route_code": str})
    df_rs = pd.read_csv(rs_path, dtype={"route_code": str})
    df_rm["route_code"] = df_rm["route_code"].map(_norm_route_code)
    df_rs["route_code"] = df_rs["route_code"].map(_norm_route_code)

    required_s = {"stop_id", "stop_name", "stop_category", "latitude", "longitude"}
    if not required_s.issubset(df_stops.columns):
        raise ValueError(f"stops.csv thiếu cột: {required_s - set(df_stops.columns)}")

    rm_cols = {"route_code", "route_name", "service_start", "service_end"}
    if not rm_cols.issubset(df_rm.columns):
        raise ValueError("route_master.csv cần: route_code, route_name, service_start, service_end")
    if "color" not in df_rm.columns:
        df_rm = df_rm.copy()
        df_rm["color"] = "#607D8B"

    if not {"route_code", "sequence", "stop_id"}.issubset(df_rs.columns):
        raise ValueError("route_stops.csv cần: route_code, sequence, stop_id")

    stops: List[BusStop] = []
    for _, row in df_stops.sort_values("stop_id").iterrows():
        stops.append(
            BusStop(
                int(row["stop_id"]),
                str(row["stop_name"]),
                str(row["stop_category"]),
                float(row["latitude"]),
                float(row["longitude"]),
            )
        )
    stop_ids_set = {s.id for s in stops}

    routes: Dict[str, Dict[str, Any]] = {}
    route_order = sorted(df_rm["route_code"].unique())

    for code in route_order:
        rrow = df_rm[df_rm["route_code"] == code]
        if rrow.empty:
            continue
        r0 = rrow.iloc[0]
        seg = df_rs[df_rs["route_code"] == code].sort_values("sequence")
        path = [int(x) for x in seg["stop_id"].tolist()]
        if len(path) < 2:
            raise ValueError(f"Tuyến {code}: cần ít nhất 2 trạm trong route_stops.csv")
        for sid in path:
            if sid not in stop_ids_set:
                raise ValueError(f"Tuyến {code}: stop_id {sid} không có trong stops.csv")

        routes[str(code)] = {
            "name": str(r0["route_name"]),
            "color": str(r0["color"]),
            "service_start": str(r0["service_start"]),
            "service_end": str(r0["service_end"]),
            "stops": path,
        }

    rm_codes = set(df_rm["route_code"])
    rs_codes = set(df_rs["route_code"])
    if rm_codes != rs_codes:
        raise ValueError(
            f"route_master và route_stops không khớp tuyến: {rm_codes ^ rs_codes}"
        )

    return stops, routes, sorted(routes.keys(), key=lambda x: (len(x), x))


# ---------------------------------------------------------------------------
# Network class
# ---------------------------------------------------------------------------
class HCMCBusNetwork:
    """Multi-route bus network; topology từ CSV trong ``data_dir``."""

    def __init__(self, seed: int = 42, data_dir: str | None = None) -> None:
        self._data_dir = data_dir if data_dir is not None else default_data_dir()
        self.stops, self.routes, self.route_codes = load_network_from_csv(self._data_dir)
        self.stop_map: Dict[int, BusStop] = {s.id: s for s in self.stops}
        self.rng = np.random.default_rng(seed)

        self.graph = self._build_graph()
        self.demand = self._simulate_demand()
        self.route_rtt = self._compute_route_rtts()
        self.wait_weight_by_stop: Dict[int, float] = {
            s.id: 1.0 for s in self.stops
        }

    # -- graph ---------------------------------------------------------------

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for s in self.stops:
            G.add_node(s.id, name=s.name, category=s.category, pos=(s.lon, s.lat))
        for rdef in self.routes.values():
            ids = rdef["stops"]
            for a, b in zip(ids[:-1], ids[1:]):
                if not G.has_edge(a, b):
                    dist = _haversine(
                        self.stop_map[a].lat, self.stop_map[a].lon,
                        self.stop_map[b].lat, self.stop_map[b].lon,
                    )
                    G.add_edge(a, b, distance_km=round(dist, 3))
        return G

    # -- demand --------------------------------------------------------------

    def _simulate_demand(self) -> Dict[int, float]:
        demand: Dict[int, float] = {}
        for s in self.stops:
            lo, hi = DEMAND_RANGE[s.category]
            base = float(self.rng.uniform(lo, hi))
            n_routes = sum(
                1 for r in self.routes.values() if s.id in r["stops"]
            )
            accessibility_boost = 1.0 + 0.25 * (n_routes - 1)
            demand[s.id] = round(base * accessibility_boost, 1)
        return demand

    # -- route metrics -------------------------------------------------------

    def _compute_route_rtts(self) -> Dict[str, float]:
        rtt: Dict[str, float] = {}
        for code in self.route_codes:
            ids = self.routes[code]["stops"]
            dist = sum(
                self.graph[a][b]["distance_km"]
                for a, b in zip(ids[:-1], ids[1:])
            )
            one_way = (dist / AVG_SPEED_KMH) * 60.0 + len(ids) * AVG_DWELL_MIN
            rtt[code] = round(2 * one_way + TURNAROUND_MIN, 1)
        return rtt

    def route_distance_km(self, code: str) -> float:
        ids = self.routes[code]["stops"]
        return round(sum(
            self.graph[a][b]["distance_km"]
            for a, b in zip(ids[:-1], ids[1:])
        ), 2)

    def routes_serving(self, stop_id: int) -> List[str]:
        return [c for c in self.route_codes
                if stop_id in self.routes[c]["stops"]]

    def get_stop_ids(self) -> List[int]:
        return sorted(s.id for s in self.stops)

    def get_adjacency_matrix(self, weighted: bool = False) -> "np.ndarray":
        node_list = self.get_stop_ids()
        weight = "distance_km" if weighted else None
        A = nx.to_numpy_array(self.graph, nodelist=node_list, weight=weight)
        return A.astype(np.float32)

    # -- Cytoscape.js export -------------------------------------------------

    def to_cytoscape_elements(
        self,
        schedule: np.ndarray,
        slot_minutes: int,
    ) -> List[dict]:
        from utils.schema_sim import SERVICE_END, SERVICE_START

        span_min = (
            SERVICE_END.hour * 60
            + SERVICE_END.minute
            - SERVICE_START.hour * 60
            - SERVICE_START.minute
        )
        hours = span_min / 60.0 if span_min > 0 else 1.0

        positions = self._vis_positions()
        elements: List[dict] = []

        for s in self.stops:
            ww = float(self.wait_weight_by_stop.get(s.id, 1.0))
            elements.append({
                "group": "nodes",
                "data": {
                    "id": str(s.id),
                    "label": s.name,
                    "category": s.category,
                    "demand": self.demand[s.id],
                    "n_routes": len(self.routes_serving(s.id)),
                    "wait_weight": round(ww, 3),
                },
                "position": positions[s.id],
            })

        codes = self.route_codes
        for ri, code in enumerate(codes):
            rdef = self.routes[code]
            row = schedule[ri]
            trips = int(np.sum(row))
            trips_per_hour = trips / hours
            freq = max(0.3, trips_per_hour)
            ids = rdef["stops"]
            for a, b in zip(ids[:-1], ids[1:]):
                elements.append({
                    "group": "edges",
                    "data": {
                        "id": f"e-{code}-{a}-{b}",
                        "source": str(a),
                        "target": str(b),
                        "route": code,
                        "route_name": rdef["name"],
                        "trips_per_day": trips,
                        "frequency": round(freq, 2),
                        "width": max(2.0, min(14.0, freq * 1.2)),
                        "color": rdef["color"],
                        "label": f"T{code}: {trips} ch/ ngày",
                    },
                })
        return elements

    def _vis_positions(self) -> Dict[int, Dict[str, float]]:
        lats = [s.lat for s in self.stops]
        lons = [s.lon for s in self.stops]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        pad_lat = (max_lat - min_lat) * 0.08
        pad_lon = (max_lon - min_lon) * 0.08
        W, H = 800, 650

        pos: Dict[int, Dict[str, float]] = {}
        for s in self.stops:
            x = (s.lon - min_lon + pad_lon) / (max_lon - min_lon + 2 * pad_lon) * W
            y = (1 - (s.lat - min_lat + pad_lat) / (max_lat - min_lat + 2 * pad_lat)) * H
            pos[s.id] = {"x": round(x, 1), "y": round(y, 1)}
        return pos

    # -- summary helpers for the dashboard -----------------------------------

    def network_summary(self) -> dict:
        return {
            "n_stops": len(self.stops),
            "n_routes": len(self.routes),
            "total_demand": round(sum(self.demand.values()), 1),
            "routes": {
                code: {
                    "name": rdef["name"],
                    "color": rdef["color"],
                    "n_stops": len(rdef["stops"]),
                    "distance_km": self.route_distance_km(code),
                    "rtt_min": self.route_rtt[code],
                }
                for code, rdef in self.routes.items()
            },
        }


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
