# Dashboard Peramalan Hotspot Kalimantan (Streamlit)

```bash
pip install -r requirements.txt
python dashboard/build_data.py        # sekali saja -> dashboard/data/*.parquet (ERA5 ringan)
streamlit run dashboard/app.py
```

Sumber data (semua dari repo ini): `kecamatan_kalimantan.geojson`, `data/hotspot_kecamatan_harian.csv`,
`forecast/*.csv`, `tables/*.csv|parquet`, dan ERA5 hasil `build_data.py`.
Hari setelah data hotspot aktual terakhir (8 Sep 2026) memakai prediksi outlook model.
