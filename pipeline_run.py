"""
Pipeline đầy đủ (trước đây trong main.py).
Notebook và main.py đều import từ đây để tránh trùng mã.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class Module1Result:
    net: Any
    stop_ids: list
    adj: Any
    n_nodes: int
    tables: Dict[str, Any]
    X: np.ndarray
    Y: np.ndarray


@dataclass
class Module2Result:
    model: Any
    avg_predicted_demand: np.ndarray
    wait_w: Dict[int, float]


def run_module1(
    seed: int = 42,
    days: int = 7,
    buses: int = 12,
    trips_per_bus_day: int = 6,
    freq: int = 15,
    window: int = 4,
    base_date: str = "2025-09-01",
    out_dir: str = "data",
) -> Module1Result:
    from utils.network_sim import HCMCBusNetwork
    from utils.schema_sim import SchemaSimulator, build_feature_tensor_from_snapshots

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("MODULE 1 — Schema Data Generation (8 tables)")
    print("=" * 60)

    net = HCMCBusNetwork(seed=seed)
    schema_sim = SchemaSimulator(net, seed=seed)

    tables = schema_sim.generate_all(
        n_days=days,
        n_buses=buses,
        trips_per_bus_per_day=trips_per_bus_day,
        freq_minutes=freq,
        base_date=base_date,
        out_dir=out_dir,
    )

    stop_ids = net.get_stop_ids()
    adj = net.get_adjacency_matrix()
    n_nodes = len(stop_ids)

    snapshots = tables["spatiotemporal_snapshots"]
    X, Y = build_feature_tensor_from_snapshots(snapshots, stop_ids, window=window)

    print(f"\n  Network : {net.graph.number_of_nodes()} stops, "
          f"{net.graph.number_of_edges()} segments, "
          f"{len(net.routes)} routes")
    print(f"  Snapshots: {len(snapshots):,} rows "
          f"({len(tables['spatiotemporal_snapshots']['timestamp'].unique())} time-steps × "
          f"{len(snapshots['stop_id'].unique())} stops)")
    print(f"  GNN tensors: X={X.shape}  Y={Y.shape}\n")

    return Module1Result(
        net=net,
        stop_ids=stop_ids,
        adj=adj,
        n_nodes=n_nodes,
        tables=tables,
        X=X,
        Y=Y,
    )


def run_module2(
    m1: Module1Result,
    seed: int = 42,
    window: int = 4,
    epochs: int = 20,
    batch: int = 32,
    gnn_aux_path: str = "data/gnn_auxiliary.json",
) -> Module2Result:
    print("=" * 60)
    print("MODULE 2 — Spatio-Temporal GNN Training")
    print("=" * 60)

    import tensorflow as tf
    from models.gnn_model import build_model
    from spektral.utils.convolution import gcn_filter

    X, Y = m1.X, m1.Y
    adj = m1.adj
    n_nodes = m1.n_nodes
    stop_ids = m1.stop_ids

    tf.random.set_seed(seed)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    if len(X_val) == 0:
        X_val, Y_val = X_train, Y_train

    model = build_model(
        n_nodes=n_nodes,
        window=window,
        n_features=X.shape[-1],
    )
    model.summary()

    adj_gcn = gcn_filter(np.asarray(adj, dtype=np.float64))
    a_tensor = tf.constant(adj_gcn, dtype=tf.float32)

    def make_dataset(x, y, batch_size, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=seed)
        ds = ds.batch(batch_size)
        ds = ds.map(lambda xi, yi: ((xi, a_tensor), yi))
        return ds.prefetch(tf.data.AUTOTUNE)

    ds_train = make_dataset(X_train, Y_train, batch, shuffle=True)
    ds_val = make_dataset(X_val, Y_val, batch)

    model.fit(ds_train, validation_data=ds_val, epochs=epochs)

    preds = model.predict(make_dataset(X_val, Y_val, batch))
    avg_predicted_demand = np.mean(preds, axis=0)
    speed_mean_per_stop = np.mean(X_val[:, :, :, 1], axis=(0, 1))

    from utils.gnn_propagation import build_wait_weights_for_stops, save_gnn_auxiliary

    wait_w = build_wait_weights_for_stops(
        stop_ids,
        np.asarray(adj, dtype=np.float64),
        avg_predicted_demand,
        speed_mean_per_stop,
        diffusion_steps=3,
        diffusion_mix=0.55,
        ref_speed_kmh=25.0,
    )
    save_gnn_auxiliary(
        gnn_aux_path,
        wait_w,
        meta={
            "description": "Lan truyền stress ùn xe trên đồ thị (cùng topology GCN): "
            "stress cục bộ = nhu cầu dự báo × độ chậm; khuếch tán P @ h.",
            "diffusion_steps": 3,
            "diffusion_mix": 0.55,
        },
    )
    print(f"\n  Avg predicted demand/stop: {avg_predicted_demand.round(1)}")
    print(f"  GNN congestion wait-weights (mean={np.mean(list(wait_w.values())):.3f})\n")

    return Module2Result(
        model=model,
        avg_predicted_demand=avg_predicted_demand,
        wait_w=wait_w,
    )


def run_module3(
    m1: Module1Result,
    m2: Module2Result,
    ga_pop: int = 50,
    ga_iter: int = 150,
    tabu_iter: int = 80,
) -> Tuple[dict, dict, dict]:
    print("=" * 60)
    print("MODULE 3 — Network-wide GA + Tabu Optimizer")
    print("=" * 60)

    from optimizers.network_optimizer import (
        SLOT_MINUTES,
        DepartureScheduleOptimizer,
        baseline_schedule_slots,
        compare,
        compute_metrics_from_schedule,
    )

    net = m1.net
    stop_ids = m1.stop_ids

    for s in net.stops:
        node_idx = stop_ids.index(s.id)
        net.demand[s.id] = float(m2.avg_predicted_demand[node_idx])
    net.wait_weight_by_stop.update(m2.wait_w)

    sched_before = baseline_schedule_slots(net, every_k_slots=2)
    m_before = compute_metrics_from_schedule(net, sched_before, SLOT_MINUTES)

    total_fleet = m_before["total_buses"]
    opt = DepartureScheduleOptimizer(
        net,
        total_fleet=total_fleet,
        ga_pop=ga_pop,
        ga_iter=ga_iter,
        tabu_iter=tabu_iter,
        slot_minutes=SLOT_MINUTES,
    )
    sched_after = opt.optimize()
    m_after = compute_metrics_from_schedule(net, sched_after, SLOT_MINUTES)
    comp = compare(m_before, m_after)

    print("\n  --- Chuyến / ngày (không headway cố định) ---")
    for code in net.route_codes:
        print(f"    Tuyến {code}: {m_before['trips_per_route'][code]} chuyến"
              f"  ->  {m_after['trips_per_route'][code]} chuyến"
              f"  (xe: {m_before['buses'][code]} -> {m_after['buses'][code]})")

    print(f"\n  Wait time : {m_before['avg_wait_min']:.2f} ph"
          f"  ->  {m_after['avg_wait_min']:.2f} ph"
          f"  ({comp['wait']['pct']:+.1f}%)")
    print(f"  Objective : {m_before['objective']:.1f}"
          f"  ->  {m_after['objective']:.1f}"
          f"  ({comp['objective']['pct']:+.1f}%)")
    print("=" * 60)
    print("Done.  Run `python app.py` to view the dashboard.")

    return m_before, m_after, comp


def run_full_pipeline(
    seed: int = 42,
    days: int = 7,
    freq: int = 15,
    window: int = 4,
    epochs: int = 20,
    batch: int = 32,
    ga_pop: int = 50,
    ga_iter: int = 150,
    tabu_iter: int = 80,
    buses: int = 12,
    trips_per_bus_day: int = 6,
    base_date: str = "2025-09-01",
    out_dir: str = "data",
    gnn_aux_path: str = "data/gnn_auxiliary.json",
) -> Tuple[Module1Result, Module2Result, dict, dict, dict]:
    os.makedirs(out_dir, exist_ok=True)
    m1 = run_module1(
        seed=seed,
        days=days,
        buses=buses,
        trips_per_bus_day=trips_per_bus_day,
        freq=freq,
        window=window,
        base_date=base_date,
        out_dir=out_dir,
    )
    m2 = run_module2(
        m1,
        seed=seed,
        window=window,
        epochs=epochs,
        batch=batch,
        gnn_aux_path=gnn_aux_path,
    )
    m_before, m_after, comp = run_module3(
        m1,
        m2,
        ga_pop=ga_pop,
        ga_iter=ga_iter,
        tabu_iter=tabu_iter,
    )
    return m1, m2, m_before, m_after, comp
