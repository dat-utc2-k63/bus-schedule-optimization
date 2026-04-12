"""
Multi-route HCMC bus network definition and demand simulation.

Defines a 13-stop, 4-route network with shared transfer stops,
simulates per-stop demand, and exports Cytoscape.js graph elements
for the Flask dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import numpy as np

from utils.domain import BusStop, DEMAND_RANGE

# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------
NETWORK_STOPS: List[BusStop] = [
    # --- Line 19 corridor ---
    BusStop(0,  "Ben Thanh",        "hub",         10.7726, 106.6980),
    BusStop(1,  "Nguyen Trai",      "commercial",  10.7580, 106.6820),
    BusStop(2,  "An Duong Vuong",   "residential", 10.7530, 106.6620),
    BusStop(3,  "Hung Vuong",       "commercial",  10.7560, 106.6510),
    BusStop(4,  "Hong Bang",        "school",      10.7570, 106.6380),
    BusStop(5,  "Kinh Duong Vuong", "residential", 10.7620, 106.6150),
    BusStop(6,  "Tan Ky Tan Quy",  "residential", 10.7830, 106.6060),
    BusStop(7,  "An Suong",         "terminal",    10.8340, 106.6190),
    # --- Additional network stops ---
    BusStop(8,  "Cho Lon",          "hub",         10.7500, 106.6520),
    BusStop(9,  "Phu Nhuan",        "commercial",  10.7980, 106.6820),
    BusStop(10, "Go Vap",           "terminal",    10.8230, 106.6700),
    BusStop(11, "Thu Duc",          "terminal",    10.8500, 106.7530),
    BusStop(12, "Binh Thanh",       "residential", 10.8080, 106.7100),
]

ROUTE_DEFS: Dict[str, Dict[str, Any]] = {
    "19": {
        "name": "Bến Thành – An Sương",
        "stops": [0, 1, 2, 3, 4, 5, 6, 7],
        "color": "#2196F3",
    },
    "01": {
        "name": "Bến Thành – Gò Vấp",
        "stops": [0, 8, 3, 5, 10],
        "color": "#4CAF50",
    },
    "65": {
        "name": "Chợ Lớn – Thủ Đức",
        "stops": [8, 4, 6, 12, 11],
        "color": "#FF9800",
    },
    "52": {
        "name": "Bến Thành – Thủ Đức",
        "stops": [0, 9, 12, 11],
        "color": "#9C27B0",
    },
}

# Operational constants
AVG_SPEED_KMH = 18.0
AVG_DWELL_MIN = 1.0
TURNAROUND_MIN = 5.0
COST_PER_BUS_HOUR = 45.0  # VND-equivalent abstract unit


# ---------------------------------------------------------------------------
# Network class
# ---------------------------------------------------------------------------
class HCMCBusNetwork:
    """Multi-route bus network with demand simulation."""

    def __init__(self, seed: int = 42) -> None:
        self.stops = NETWORK_STOPS
        self.stop_map: Dict[int, BusStop] = {s.id: s for s in self.stops}
        self.routes = ROUTE_DEFS
        self.route_codes = sorted(self.routes.keys())
        self.rng = np.random.default_rng(seed)

        self.graph = self._build_graph()
        self.demand = self._simulate_demand()
        self.route_rtt = self._compute_route_rtts()
        # Per-stop multipliers for passenger wait (GNN congestion propagation); default uniform
        self.wait_weight_by_stop: Dict[int, float] = {
            s.id: 1.0 for s in self.stops
        }

    # -- graph ---------------------------------------------------------------

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for s in self.stops:
            G.add_node(s.id, name=s.name, category=s.category,
                       pos=(s.lon, s.lat))
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
        """
        Per-stop average demand.  Stops served by more routes are busier
        (higher accessibility ⇒ higher ridership).
        """
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
        """Round-trip time (minutes) per route."""
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
        """Ordered stop IDs matching the adjacency matrix row/column order."""
        return sorted(s.id for s in self.stops)

    def get_adjacency_matrix(self, weighted: bool = False) -> "np.ndarray":
        """
        Return (N, N) adjacency matrix in consistent stop-id order.

        Parameters
        ----------
        weighted : bool
            If True, use distance_km as edge weight; otherwise binary.
        """
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
        """
        schedule[r, k]: tuyến r có khởi hành tại slot k (xuất bến trạm đầu).
        Độ dày cạnh ~ số chuyến / ngày (không dùng headway cố định).
        """
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
