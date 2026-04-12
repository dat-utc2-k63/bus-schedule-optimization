# Hybrid AI — Tối Ưu Lịch Xe Buýt (TP.HCM)

Luận văn: **Spatio-Temporal GNN** (GCN + GRU) + **GA + Tabu** trên mạng nhiều tuyến, dữ liệu tổng hợp (8 bảng CSV).

## Cấu trúc

```
├── data/                    # CSV (topology: stops, route_master, route_stops + bảng sinh)
├── models/gnn_model.py
├── optimizers/network_optimizer.py
├── utils/
│   ├── domain.py
│   ├── network_sim.py
│   ├── schema_sim.py
│   └── gnn_propagation.py
├── main.py                  # Pipeline CLI (tùy chọn)
├── notebook.ipynb           # Chạy & đánh giá chính
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

- **Notebook (khuyến nghị):** mở `notebook.ipynb` — đọc `data/*.csv` có sẵn, không sinh lại trừ khi thiếu file hoặc `REGENERATE_DATA = True`.
- **CLI:** `python main.py` — dùng `--regenerate-data` nếu muốn tạo lại toàn bộ CSV.

## Tham chiếu

- [Spektral](https://github.com/danielegrattarola/spektral)
- [scikit-opt](https://github.com/guofei9987/scikit-opt)
