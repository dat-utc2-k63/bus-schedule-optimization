#!/usr/bin/env python3
"""
main.py — End-to-end CLI (optional; primary workflow is notebook.ipynb)
========================================================================
Module 1  Load or generate 8 CSV tables
Module 2  Train Spatio-Temporal GNN
Module 3  GA + Tabu departure-slot optimizer
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid AI for Real-time Bus Scheduling – HCMC",
    )
    p.add_argument("--days",  type=int, default=7,  help="Days of simulated data")
    p.add_argument("--freq",  type=int, default=15, help="Snapshot interval (min)")
    p.add_argument("--window",type=int, default=4,  help="GNN look-back window")
    p.add_argument("--epochs",type=int, default=20, help="GNN training epochs")
    p.add_argument("--batch", type=int, default=32, help="Training batch size")
    p.add_argument("--ga-pop",type=int, default=50, help="GA population size")
    p.add_argument("--ga-iter",type=int,default=150,help="GA generations")
    p.add_argument("--buses", type=int, default=12, help="Total fleet size")
    p.add_argument("--trips-per-bus-day", type=int, default=6,
                   help="Trips per bus per day")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Ghi đè và sinh lại toàn bộ 8 bảng CSV trong data/ (mặc định: dùng file có sẵn nếu đủ)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("data", exist_ok=True)

    # ==================================================================
    # MODULE 1 — Normalized dataset generation (8 tables)
    # ==================================================================
    print("=" * 60)
    print("MODULE 1 — Schema Data Generation (8 tables)")
    print("=" * 60)

    from utils.network_sim import HCMCBusNetwork
    from utils.schema_sim import (
        SchemaSimulator,
        all_generated_tables_exist,
        build_feature_tensor_from_snapshots,
        load_generated_tables,
    )

    net = HCMCBusNetwork(seed=args.seed)
    schema_sim = SchemaSimulator(net, seed=args.seed)

    if args.regenerate_data or not all_generated_tables_exist("data"):
        tables = schema_sim.generate_all(
            n_days=args.days,
            n_buses=args.buses,
            trips_per_bus_per_day=args.trips_per_bus_day,
            freq_minutes=args.freq,
            base_date="2025-09-01",
            out_dir="data",
        )
    else:
        print(
            "[schema_sim] All CSVs present in data/ — skipping generation "
            "(use --regenerate-data to rebuild)"
        )
        tables = load_generated_tables("data")

    # Adjacency for GNN — ordered by stop_id (same order as stop_ids list)
    stop_ids = net.get_stop_ids()          # e.g. [0, 1, 2, ..., 12]
    adj = net.get_adjacency_matrix()       # (N, N) float32
    n_nodes = len(stop_ids)

    # Build GNN tensors from SpatioTemporal_Snapshots (long → windowed)
    snapshots = tables["spatiotemporal_snapshots"]
    X, Y = build_feature_tensor_from_snapshots(snapshots, stop_ids, window=args.window)

    print(f"\n  Network : {net.graph.number_of_nodes()} stops, "
          f"{net.graph.number_of_edges()} segments, "
          f"{len(net.routes)} routes")
    print(f"  Snapshots: {len(snapshots):,} rows "
          f"({len(tables['spatiotemporal_snapshots']['timestamp'].unique())} time-steps × "
          f"{len(snapshots['stop_id'].unique())} stops)")
    print(f"  GNN tensors: X={X.shape}  Y={Y.shape}\n")

    # ==================================================================
    # MODULE 2 — Spatio-Temporal GNN Training
    # ==================================================================
    print("=" * 60)
    print("MODULE 2 — Spatio-Temporal GNN Training")
    print("=" * 60)

    import tensorflow as tf
    from models.gnn_model import build_model
    from spektral.utils.convolution import gcn_filter

    tf.random.set_seed(args.seed)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    if len(X_val) == 0:
        X_val, Y_val = X_train, Y_train

    model = build_model(
        n_nodes=n_nodes,
        window=args.window,
        n_features=X.shape[-1],          # 3: demand, speed, is_rain
    )
    model.summary()

    # GCN requires normalized adjacency (D^-1/2 A D^-1/2 with self-loops)
    adj_gcn = gcn_filter(np.asarray(adj, dtype=np.float64))
    a_tensor = tf.constant(adj_gcn, dtype=tf.float32)

    def make_dataset(x, y, batch_size, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=args.seed)
        ds = ds.batch(batch_size)
        ds = ds.map(lambda xi, yi: ((xi, a_tensor), yi))
        return ds.prefetch(tf.data.AUTOTUNE)

    ds_train = make_dataset(X_train, Y_train, args.batch, shuffle=True)
    ds_val   = make_dataset(X_val,   Y_val,   args.batch)

    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs)

    # Average predicted demand per node (fed to the optimizer)
    preds = model.predict(make_dataset(X_val, Y_val, args.batch))
    avg_predicted_demand = np.mean(preds, axis=0)   # shape (N,)
    # Mean traffic speed per stop (validation window) — pairs with GNN demand for congestion stress
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
        "data/gnn_auxiliary.json",
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

    # ==================================================================
    # MODULE 3 — Network-wide Hybrid GA + Tabu Optimizer
    # ==================================================================
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

    # Inject GNN demand predictions + graph-propagated congestion weights into the network
    for s in net.stops:
        node_idx = stop_ids.index(s.id)
        net.demand[s.id] = float(avg_predicted_demand[node_idx])
    net.wait_weight_by_stop.update(wait_w)

    sched_before = baseline_schedule_slots(net, every_k_slots=2)
    m_before = compute_metrics_from_schedule(net, sched_before, SLOT_MINUTES)

    total_fleet = m_before["total_buses"]
    opt = DepartureScheduleOptimizer(
        net,
        total_fleet=total_fleet,
        ga_pop=args.ga_pop,
        ga_iter=args.ga_iter,
        tabu_iter=80,
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
    print("Done.  Open `notebook.ipynb` for experiments and figures.")


if __name__ == "__main__":
    main()
