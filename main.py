#!/usr/bin/env python3
"""
CLI cho pipeline — logic đầy đủ nằm trong `pipeline_run.py`
(chạy từ notebook: `notebooks/bus_schedule_optimization.ipynb`).
"""

from __future__ import annotations

import argparse
import os

from pipeline_run import run_full_pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid AI for Real-time Bus Scheduling – HCMC",
    )
    p.add_argument("--days", type=int, default=7, help="Days of simulated data")
    p.add_argument("--freq", type=int, default=15, help="Snapshot interval (min)")
    p.add_argument("--window", type=int, default=4, help="GNN look-back window")
    p.add_argument("--epochs", type=int, default=20, help="GNN training epochs")
    p.add_argument("--batch", type=int, default=32, help="Training batch size")
    p.add_argument("--ga-pop", type=int, default=50, help="GA population size")
    p.add_argument("--ga-iter", type=int, default=150, help="GA generations")
    p.add_argument("--buses", type=int, default=12, help="Total fleet size")
    p.add_argument(
        "--trips-per-bus-day",
        type=int,
        default=6,
        help="Trips per bus per day",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("data", exist_ok=True)
    run_full_pipeline(
        seed=args.seed,
        days=args.days,
        freq=args.freq,
        window=args.window,
        epochs=args.epochs,
        batch=args.batch,
        ga_pop=args.ga_pop,
        ga_iter=args.ga_iter,
        tabu_iter=80,
        buses=args.buses,
        trips_per_bus_day=args.trips_per_bus_day,
    )


if __name__ == "__main__":
    main()
