# Hybrid AI — Bus Scheduling Optimization (HCMC)

Graduation project: Spatio-Temporal GNN + Genetic Algorithm + Tabu Search on a multi-route synthetic network.

## Layout

```
bus-schedule-optimization/
├── data/                    # Generated CSVs (8-table schema, see below)
├── models/gnn_model.py      # GCN + GRU (Spektral / TensorFlow)
├── optimizers/network_optimizer.py   # Network-wide GA + Tabu
├── utils/
│   ├── domain.py            # BusStop, demand ranges, rush/rain constants
│   ├── network_sim.py       # 13-stop, 4-route graph + demand
│   └── schema_sim.py        # Exports Route_Master, Stops, … Operation_Logs
├── main.py                  # Train GNN + run optimizer
├── app.py                   # Flask dashboard (before / after graphs)
├── requirements.txt
└── README.md
```

## Data schema (CSV in `data/`)

| File | Role |
|------|------|
| `route_master.csv` | Routes |
| `stops.csv` | Stops |
| `network_segments.csv` | Edges |
| `buses.csv` | Fleet |
| `trips.csv` | Trips |
| `stop_times.csv` | Planned times |
| `spatiotemporal_snapshots.csv` | GNN training (long format) |
| `operation_logs.csv` | Actual operations |

## Quick start

```bash
pip install -r requirements.txt
python main.py --days 7 --epochs 20   # sinh data/*.csv
python app.py   # giao diện: http://127.0.0.1:5000
                # Tab 1: đồ thị trước/sau tối ưu · Tab 2: xem bảng CSV
```

## References

- [Spektral](https://github.com/danielegrattarola/spektral)
- [scikit-opt](https://github.com/guofei9987/scikit-opt)
