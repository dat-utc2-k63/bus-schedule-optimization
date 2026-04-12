"""
Flask dashboard — Bus Schedule Optimization + schema data viewer.

Run:
    python app.py          → http://127.0.0.1:5000

Tab 1: Before/after network graphs (Cytoscape)
Tab 2: Preview of CSV tables in data/ (run `python main.py` to generate)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, render_template

from utils.network_sim import HCMCBusNetwork
from optimizers.network_optimizer import (
    SLOT_MINUTES,
    DepartureScheduleOptimizer,
    baseline_schedule_slots,
    build_departure_rows_from_schedule,
    compare,
    compute_metrics_from_schedule,
)
from utils.gnn_propagation import load_gnn_auxiliary

app = Flask(__name__)

# (key, filename, max_rows or None for full table)
SCHEMA_FILES: List[Tuple[str, str, Optional[int]]] = [
    ("route_master", "route_master.csv", None),
    ("stops", "stops.csv", None),
    ("network_segments", "network_segments.csv", None),
    ("buses", "buses.csv", None),
    ("trips", "trips.csv", 35),
    ("stop_times", "stop_times.csv", 40),
    ("spatiotemporal_snapshots", "spatiotemporal_snapshots.csv", 60),
    ("operation_logs", "operation_logs.csv", 40),
]

SCHEMA_TITLES_VI: Dict[str, str] = {
    "route_master": "Route_Master — thông tin tuyến",
    "stops": "Stops — trạm dừng",
    "network_segments": "Network_Segments — cạnh đồ thị",
    "buses": "Buses — đội xe",
    "trips": "Trips — chuyến xe",
    "stop_times": "Stop_Times — lịch kế hoạch theo trạm",
    "spatiotemporal_snapshots": "SpatioTemporal_Snapshots — dữ liệu chuỗi thời gian (huấn luyện GNN)",
    "operation_logs": "Operation_Logs — vận hành thực tế",
}


def _df_to_records(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif df[col].dtype == bool:
            df[col] = df[col].map({True: "True", False: "False"})
    return list(df.columns), df.to_dict(orient="records")


def load_schema_preview(data_dir: str = "data") -> Dict[str, Any]:
    """Load CSV previews for the data tab (small row caps for large tables)."""
    out: Dict[str, Any] = {}
    for key, fn, max_rows in SCHEMA_FILES:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            out[key] = {
                "title": SCHEMA_TITLES_VI.get(key, key),
                "filename": fn,
                "missing": True,
                "columns": [],
                "rows": [],
                "total_rows": 0,
                "shown": 0,
            }
            continue
        df_full = pd.read_csv(path)
        total = len(df_full)
        df = df_full if max_rows is None else df_full.head(max_rows)
        cols, rows = _df_to_records(df)
        out[key] = {
            "title": SCHEMA_TITLES_VI.get(key, key),
            "filename": fn,
            "missing": False,
            "columns": cols,
            "rows": rows,
            "total_rows": total,
            "shown": len(rows),
        }
    return out


def _run_pipeline() -> dict:
    """Simulate network, compute baseline, run optimisation, build payload."""
    net = HCMCBusNetwork(seed=42)
    gnn_loaded = load_gnn_auxiliary(os.path.join("data", "gnn_auxiliary.json"))
    gnn_aux: Dict[str, Any] = {"loaded": False, "meta": {}}
    if gnn_loaded:
        w_by_stop, meta = gnn_loaded
        net.wait_weight_by_stop.update(w_by_stop)
        gnn_aux = {"loaded": True, "meta": meta}

    summary = net.network_summary()

    sched_before = baseline_schedule_slots(net, every_k_slots=2)
    m_before = compute_metrics_from_schedule(net, sched_before, SLOT_MINUTES)
    cy_before = net.to_cytoscape_elements(sched_before, SLOT_MINUTES)

    total_fleet = m_before["total_buses"]
    opt = DepartureScheduleOptimizer(
        net,
        total_fleet=total_fleet,
        ga_pop=50,
        ga_iter=120,
        tabu_iter=80,
        slot_minutes=SLOT_MINUTES,
    )
    sched_after = opt.optimize()
    m_after = compute_metrics_from_schedule(net, sched_after, SLOT_MINUTES)
    cy_after = net.to_cytoscape_elements(sched_after, SLOT_MINUTES)

    optimized_schedule = build_departure_rows_from_schedule(
        net, sched_after, SLOT_MINUTES
    )

    comp = compare(m_before, m_after)

    per_route = []
    for code in net.route_codes:
        rdef = net.routes[code]
        per_route.append(
            {
                "code": code,
                "name": rdef["name"],
                "color": rdef["color"],
                "distance_km": net.route_distance_km(code),
                "rtt_min": net.route_rtt[code],
                "trips_day_before": m_before["trips_per_route"][code],
                "trips_day_after": m_after["trips_per_route"][code],
                "trips_h_before": m_before["trips_per_hour"][code],
                "trips_h_after": m_after["trips_per_hour"][code],
                "before_buses": m_before["buses"][code],
                "after_buses": m_after["buses"][code],
                "before_wait": m_before["per_route_wait"][code],
                "after_wait": m_after["per_route_wait"][code],
            }
        )

    return {
        "summary": summary,
        "before": {**{k: v for k, v in m_before.items()}, "cy": cy_before},
        "after": {**{k: v for k, v in m_after.items()}, "cy": cy_after},
        "comparison": comp,
        "per_route": per_route,
        "optimized_schedule": optimized_schedule,
        "gnn_auxiliary": gnn_aux,
        "schedule_meta": {
            "day": "2025-09-01",
            "window": "05:00 – 19:00",
            "slot_minutes": SLOT_MINUTES,
            "note": (
                "Lịch tối ưu theo khung giờ rời rạc (slot), không dùng headway cố định. "
                "Mỗi chuyến = xuất bến tại trạm đầu tuyến. "
                "Cột «Khoảng cách chuyến» = phút đến chuyến trước cùng tuyến (nếu có)."
            ),
        },
    }


@app.route("/")
def index():
    opt_data = _run_pipeline()
    schema_data = load_schema_preview()
    page = {
        "optimization": opt_data,
        "schema": schema_data,
        "schema_order": [k for k, _, _ in SCHEMA_FILES],
    }
    return render_template(
        "index.html",
        page_json=json.dumps(page, ensure_ascii=False),
    )


@app.route("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
