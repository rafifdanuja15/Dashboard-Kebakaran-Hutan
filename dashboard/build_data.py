"""
build_data.py: konsolidasi ERA5 harian (812 file parquet) menjadi file ringan untuk dashboard.

Jalankan SEKALI dari root repo:
    python dashboard/build_data.py

Keluaran (folder dashboard/data/):
    era5_kecamatan_61..65.parquet   ERA5 harian per kecamatan, satu file per provinsi
    era5_harian_provinsi.parquet    rata-rata ERA5 harian per provinsi + seluruh Kalimantan
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "dashboard", "data")
os.makedirs(OUT, exist_ok=True)

VARS = ["rain_mm", "t2m_c", "d2m_c", "wind_ms", "swvl1"]
START = pd.Timestamp("2000-11-01")  # data hotspot MODIS mulai 2000-11-09

PROV = {"61": "Kalimantan Barat", "62": "Kalimantan Tengah", "63": "Kalimantan Selatan",
        "64": "Kalimantan Timur", "65": "Kalimantan Utara"}

files = sorted(glob.glob(os.path.join(SRC, "year=*", "*.parquet")))
root_part = os.path.join(SRC, "part-00029.parquet")  # file di root data/ (bukan di folder year=)
if os.path.exists(root_part):
    files.append(root_part)
if not files:
    sys.exit(f"Tidak ada file parquet di {SRC}")

cols = ["district_id", "date", "rain_mm", "t2m_c", "d2m_c", "u10", "v10", "swvl1"]
df = pa.concat_tables([pq.read_table(f, columns=cols) for f in files]).to_pandas()
df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
df = df[df["date"] >= START].drop_duplicates(["district_id", "date"])

# kecepatan angin (m/s) dari komponen u10, v10
df["wind_ms"] = np.hypot(df["u10"], df["v10"])
df = df[["district_id", "date"] + VARS].sort_values(["district_id", "date"]).reset_index(drop=True)
df[VARS] = df[VARS].astype("float32")

# pastikan hanya satu baris per kecamatan-hari
assert not df.duplicated(["district_id", "date"]).any()

# pembulatan (presisi yang wajar untuk ERA5 harian) agar kompresi jauh lebih efektif
df = df.round({"rain_mm": 2, "t2m_c": 1, "d2m_c": 1, "wind_ms": 2, "swvl1": 3})
df[VARS] = df[VARS].astype("float32")

# satu file per provinsi (tiap file < 100 MB, batas GitHub)
df["prov_kode"] = df["district_id"].str[2:4]
for kode, g in df.groupby("prov_kode"):
    g.drop(columns="prov_kode").to_parquet(
        os.path.join(OUT, f"era5_kecamatan_{kode}.parquet"),
        compression="zstd", compression_level=15, row_group_size=100_000, index=False)

# agregat provinsi + Kalimantan
prov = df.groupby(["prov_kode", "date"])[VARS].mean().reset_index()
prov["provinsi"] = prov["prov_kode"].map(PROV)
kal = df.groupby("date")[VARS].mean().reset_index()
kal["provinsi"] = "Kalimantan"
kal["prov_kode"] = "00"
agg = pd.concat([prov, kal], ignore_index=True)
agg[VARS] = agg[VARS].astype("float32")
agg.to_parquet(os.path.join(OUT, "era5_harian_provinsi.parquet"), compression="zstd", index=False)

for f in sorted(os.listdir(OUT)):
    print(f"{f}: {os.path.getsize(os.path.join(OUT, f)) / 1e6:.1f} MB")
print("kecamatan:", df["district_id"].nunique(), "| periode:", df["date"].min().date(), "-", df["date"].max().date())
