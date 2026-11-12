"""
Dashboard Peramalan Hotspot Kalimantan (Streamlit)

Jalankan dari root repo:
    python dashboard/build_data.py          # sekali saja (membuat data ERA5 ringan)
    streamlit run dashboard/app.py
"""
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------------
# Konfigurasi & path
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Hotspot Kalimantan", page_icon="🔥", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = {
    "geojson": os.path.join(ROOT, "kecamatan_kalimantan.geojson"),
    "hotspot": os.path.join(ROOT, "data", "hotspot_kecamatan_harian.csv"),
    "outlook": os.path.join(ROOT, "forecast", "outlook_harian_kecamatan.csv"),
    "siaga": os.path.join(ROOT, "forecast", "forecast_7hari_kecamatan.csv"),
    "metrik": os.path.join(ROOT, "tables", "evaluasi_model_vs_pembanding.csv"),
    "fitur": os.path.join(ROOT, "tables", "pentingnya_fitur_gain.csv"),
    "avp": os.path.join(ROOT, "tables", "prediksi_evaluasi_kecamatan.parquet"),
    "era5_prov": os.path.join(HERE, "data", "era5_harian_provinsi.parquet"),
}

PROV_NAMA = {"61": "Kalimantan Barat", "62": "Kalimantan Tengah", "63": "Kalimantan Selatan",
             "64": "Kalimantan Timur", "65": "Kalimantan Utara"}

ERA5_VARS = {
    "rain_mm": ("Curah hujan (mm)", "#2a6fdb"),
    "t2m_c": ("Suhu udara 2 m (°C)", "#e4572e"),
    "d2m_c": ("Titik embun (°C)", "#17a398"),
    "wind_ms": ("Kecepatan angin (m/s)", "#8e6bbf"),
    "swvl1": ("Kelembapan tanah (m³/m³)", "#9a7b2f"),
}

C_UP, C_DOWN, C_FLAT = "#d62728", "#2e9e4f", "#7f7f7f"      # naik = merah, turun = hijau
C_ACTUAL, C_PRED = "#1f3a5f", "#e4572e"


def fmt(x, d=0):
    """Format angka gaya Indonesia: 1.234,5"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def tgl(d):
    return pd.Timestamp(d).strftime("%d %b %Y")


# ----------------------------------------------------------------------------------
# Pemuatan data (di-cache)
# ----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_geojson():
    with open(P["geojson"], encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_meta():
    gj = load_geojson()
    meta = pd.DataFrame([f["properties"] for f in gj["features"]])[
        ["district_id", "kecamatan", "kabupaten", "provinsi"]]
    return meta


@st.cache_data(show_spinner=False)
def load_hotspot():
    h = pd.read_csv(P["hotspot"], parse_dates=["tanggal"])
    return h.rename(columns={"kec_pcode": "district_id", "tanggal": "date", "hotspot_count": "v"})[
        ["district_id", "date", "v"]]


@st.cache_data(show_spinner=False)
def load_outlook():
    o = pd.read_csv(P["outlook"], parse_dates=["tanggal"])
    return o.rename(columns={"tanggal": "date", "pred_hotspot_harian": "v"})[["district_id", "date", "v"]]


@st.cache_data(show_spinner=False)
def load_combined():
    """Aktual (s.d. tanggal data terakhir) + prediksi outlook sesudahnya, satu tabel kecamatan-hari."""
    a, o = load_hotspot().copy(), load_outlook().copy()
    a["is_pred"], o["is_pred"] = False, True
    return pd.concat([a, o], ignore_index=True)


@st.cache_data(show_spinner=False)
def load_siaga():
    return pd.read_csv(P["siaga"])


@st.cache_data(show_spinner=False)
def load_metrik():
    return pd.read_csv(P["metrik"])


@st.cache_data(show_spinner=False)
def load_fitur():
    return pd.read_csv(P["fitur"])


@st.cache_data(show_spinner="Memuat data aktual vs prediksi…")
def load_avp():
    t = pq.read_table(P["avp"], columns=["date", "district_id", "actual_7d", "pred_final"])
    df = t.to_pandas()
    df["district_id"] = df["district_id"].astype("category")
    return df


@st.cache_data(show_spinner="Memuat data iklim ERA5…")
def load_era5(ids: tuple, start: str, end: str, level_prov: str | None):
    """Rata-rata ERA5 harian untuk wilayah terpilih.
    level_prov = 'Kalimantan' / nama provinsi -> pakai file agregat; selain itu baca per kecamatan."""
    cols = list(ERA5_VARS)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if level_prov is not None:
        a = pd.read_parquet(P["era5_prov"])
        a = a[(a["provinsi"] == level_prov) & (a["date"] >= s) & (a["date"] <= e)]
        return a.set_index("date")[cols].sort_index()
    frames = []
    for kode in sorted({i[2:4] for i in ids}):
        fp = os.path.join(HERE, "data", f"era5_kecamatan_{kode}.parquet")
        if not os.path.exists(fp):
            continue
        sel = [i for i in ids if i[2:4] == kode]
        tb = pq.read_table(fp, filters=[("district_id", "in", sel), ("date", ">=", s), ("date", "<=", e)])
        frames.append(tb.to_pandas())
    if not frames:
        return pd.DataFrame(columns=cols)
    d = pd.concat(frames)
    return d.groupby("date")[cols].mean().sort_index()


# ----------------------------------------------------------------------------------
# Pemeriksaan file
# ----------------------------------------------------------------------------------
missing = [k for k, v in P.items() if not os.path.exists(v)]
kec_files = [os.path.join(HERE, "data", f"era5_kecamatan_{k}.parquet") for k in PROV_NAMA]
if missing or not all(os.path.exists(f) for f in kec_files):
    st.error("File data belum lengkap. Jalankan `python dashboard/build_data.py` dari root repo, "
             "dan pastikan `kecamatan_kalimantan.geojson` ada di root repo.")
    if missing:
        st.write("File yang tidak ditemukan:", {k: P[k] for k in missing})
    st.stop()

meta = load_meta()
combined = load_combined()
hot_all = load_hotspot()
LAST_OBS = hot_all["date"].max()                       # tanggal data hotspot aktual terakhir
LAST_FC = load_outlook()["date"].max()                 # tanggal terakhir outlook
FIRST_OBS = hot_all["date"].min()
ids_data = set(hot_all["district_id"].unique())
meta_d = meta[meta["district_id"].isin(ids_data)].copy()

# ----------------------------------------------------------------------------------
# Sidebar: filter
# ----------------------------------------------------------------------------------
st.sidebar.title("Filter")

# --- 1. periode waktu
st.sidebar.subheader("Periode waktu")
preset = st.sidebar.selectbox(
    "Preset",
    ["90 hari terakhir data", "30 hari terakhir data", "Musim kebakaran 2026 (Jun–Des)",
     "Tahun 2026", "Tahun 2023", "Seluruh data", "Kustom"],
    index=0,
)
if preset == "90 hari terakhir data":
    rng = (LAST_OBS - pd.Timedelta(days=89), LAST_OBS)
elif preset == "30 hari terakhir data":
    rng = (LAST_OBS - pd.Timedelta(days=29), LAST_OBS)
elif preset == "Musim kebakaran 2026 (Jun–Des)":
    rng = (pd.Timestamp("2026-06-01"), LAST_FC)
elif preset == "Tahun 2026":
    rng = (pd.Timestamp("2026-01-01"), LAST_FC)
elif preset == "Tahun 2023":
    rng = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
elif preset == "Seluruh data":
    rng = (FIRST_OBS, LAST_FC)
else:
    picked = st.sidebar.date_input(
        "Rentang tanggal",
        value=((LAST_OBS - pd.Timedelta(days=89)).date(), LAST_OBS.date()),
        min_value=FIRST_OBS.date(), max_value=LAST_FC.date(), format="DD/MM/YYYY",
    )
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.sidebar.info("Pilih tanggal akhir pada kalender.")
        st.stop()
    rng = (pd.Timestamp(picked[0]), pd.Timestamp(picked[1]))
start, end = rng
st.sidebar.caption(f"{tgl(start)} → {tgl(end)} ({(end - start).days + 1} hari)")

# --- 2. wilayah
st.sidebar.subheader("Wilayah")
provs = ["Semua Kalimantan"] + [PROV_NAMA[k] for k in sorted(PROV_NAMA)]
sel_prov = st.sidebar.selectbox("Provinsi", provs)
m1 = meta_d if sel_prov == "Semua Kalimantan" else meta_d[meta_d["provinsi"] == sel_prov]
kabs = ["Semua kabupaten/kota"] + sorted(m1["kabupaten"].unique())
sel_kab = st.sidebar.selectbox("Kabupaten/Kota", kabs, disabled=(sel_prov == "Semua Kalimantan"))
if sel_prov == "Semua Kalimantan":
    sel_kab = kabs[0]
m2 = m1 if sel_kab == kabs[0] else m1[m1["kabupaten"] == sel_kab]
m2 = m2.sort_values("kecamatan")
kec_label = {r.district_id: r.kecamatan for r in m2.itertuples()}
dup = pd.Series(list(kec_label.values())).duplicated(keep=False)
kec_label = {k: (v if not d else f"{v} ({k})") for (k, v), d in zip(kec_label.items(), dup)}
kec_opts = ["Semua kecamatan"] + list(kec_label.values())
sel_kec = st.sidebar.selectbox("Kecamatan", kec_opts, disabled=(sel_kab == kabs[0]))
if sel_kab == kabs[0]:
    sel_kec = kec_opts[0]

if sel_kec != kec_opts[0]:
    sel_ids = [k for k, v in kec_label.items() if v == sel_kec]
    region_name = f"{sel_kec}, {sel_kab}"
elif sel_kab != kabs[0]:
    sel_ids, region_name = list(m2["district_id"]), f"{sel_kab}, {sel_prov}"
elif sel_prov != "Semua Kalimantan":
    sel_ids, region_name = list(m2["district_id"]), sel_prov
else:
    sel_ids, region_name = list(meta_d["district_id"]), "Seluruh Kalimantan"
sel_ids_t = tuple(sorted(sel_ids))
is_all_region = sel_prov == "Semua Kalimantan"
st.sidebar.caption(f"{len(sel_ids)} kecamatan dipilih")

# ----------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------
st.title("🔥 Dashboard Peramalan Hotspot Kalimantan")
n_fc_days = max(0, (min(end, LAST_FC) - max(start, LAST_OBS + pd.Timedelta(days=1))).days + 1)
st.caption(
    f"Wilayah: **{region_name}** · Periode: **{tgl(start)} – {tgl(end)}** · "
    f"Data hotspot aktual terakhir: **{tgl(LAST_OBS)}** (hari sesudahnya adalah prediksi model)"
)


# ----------------------------------------------------------------------------------
# Fungsi bantu agregasi periode
# ----------------------------------------------------------------------------------
def period_slice(df, s, e, ids=None):
    m = (df["date"] >= s) & (df["date"] <= e)
    if ids is not None:
        m &= df["district_id"].isin(ids)
    return df[m]


def total_in(s, e, ids):
    return period_slice(combined, s, e, ids)["v"].sum()


# ==================================================================================
# 1. KARTU PERFORMA MODEL
# ==================================================================================
st.subheader("1. Performa model")
mt = load_metrik()
split_pick = st.radio("Evaluasi", ["Holdout 2026 (murni)", "Walk-forward 2015–2025"],
                      horizontal=True, label_visibility="collapsed")
split_key = "holdout" if split_pick.startswith("Holdout") else "walk-forward"
mm = mt[(mt["split"] == split_key) & (mt["model"] == "Model")].iloc[0]
bs = mt[(mt["split"] == split_key) & (mt["model"] == "Persistensi 7h")].iloc[0]

c = st.columns(4)
cards = [
    ("AUC (ROC-AUC)", "roc_auc", 3, "normal", "Kemampuan membedakan kecamatan-hari yang ada hotspot vs tidak. 1 = sempurna, 0,5 = acak."),
    ("PR-AUC", "pr_auc", 3, "normal", "Presisi–recall pada kejadian hotspot yang jarang (≈1% sel kecamatan-hari). Lebih jujur daripada AUC untuk data tidak seimbang."),
    ("RMSE", "rmse", 3, "inverse", "Galat kuadrat rata-rata (hotspot per 7 hari). Menghukum meleset besar di puncak. Makin kecil makin baik."),
    ("MAE", "mae", 3, "inverse", "Galat absolut rata-rata (hotspot per 7 hari). Pada data 99% nol, prediktor nol pun bisa kecil MAE-nya, jadi baca bersama metrik lain."),
]
for col, (lab, key, d, dc, tip) in zip(c, cards):
    diff = mm[key] - bs[key]
    col.metric(lab, fmt(mm[key], d), delta=f"{diff:+.3f} vs persistensi".replace(".", ","),
               delta_color=dc, help=tip)
st.caption(f"Selisih dibandingkan baseline persistensi 7 hari. Bias total {split_pick.split(' (')[0].lower()}: "
           f"{fmt(mm['bias_pct'], 1)}% (negatif = model meremehkan jumlah hotspot).")

# ringkasan cepat
prev_len = (end - start).days + 1
prev_s, prev_e = start - pd.Timedelta(days=prev_len), start - pd.Timedelta(days=1)
tot_now = total_in(start, end, sel_ids)
tot_prev = total_in(prev_s, prev_e, sel_ids)
sg = load_siaga()
sg_sel = sg[sg["district_id"].isin(sel_ids)]
k1, k2, k3 = st.columns(3)
k1.metric("Total hotspot pada periode", fmt(tot_now, 1 if n_fc_days else 0),
          help="Aktual sampai tanggal data terakhir; hari sesudahnya memakai prediksi outlook." if n_fc_days else None)
if tot_prev > 0:
    chg = (tot_now - tot_prev) / tot_prev * 100
    k2.metric("Total periode sebelumnya", fmt(tot_prev, 1 if (prev_e > LAST_OBS) else 0),
              delta=f"{'+' if chg > 0 else '-'}{fmt(abs(chg), 1)}% di periode ini", delta_color="inverse" if chg != 0 else "off",
              help=f"Periode pembanding: {tgl(prev_s)} – {tgl(prev_e)} (panjang sama). Naik = merah, turun = hijau.")
else:
    k2.metric("Total periode sebelumnya", fmt(tot_prev), help=f"Pembanding: {tgl(prev_s)} – {tgl(prev_e)}. Tidak ada hotspot, persentase tidak dihitung.")
k3.metric("Kecamatan berstatus Siaga (7 hari ke depan)", f"{int((sg_sel['status'] == 'Siaga').sum())} dari {len(sg_sel)}")
if n_fc_days:
    st.info(f"Periode mencakup **{n_fc_days} hari prediksi** (setelah {tgl(LAST_OBS)}). "
            "Prediksi di luar 7 hari pertama bersifat skenario klimatologis, bukan ramalan pasti.")

# ==================================================================================
# 2. GRAFIK IKLIM ERA5
# ==================================================================================
st.subheader("2. Variabel iklim ERA5")
ERA5_END = pd.Timestamp("2026-09-23")
ce1, ce2, ce3 = st.columns([3, 1.3, 1.3])
sel_vars = ce1.multiselect("Variabel", list(ERA5_VARS), default=list(ERA5_VARS),
                           format_func=lambda k: ERA5_VARS[k][0])
mode_era = ce2.radio("Skala", ["Terstandarisasi (z-score)", "Nilai asli"], index=0)
agg_era = ce3.selectbox("Agregasi", ["Otomatis", "Harian", "Mingguan", "Bulanan"], key="agg_era")

e_start, e_end = max(start, pd.Timestamp("2000-11-01")), min(end, ERA5_END)
if e_start > e_end:
    st.warning("Data ERA5 hanya tersedia sampai 23 Sep 2026. Pilih periode yang lebih awal.")
elif not sel_vars:
    st.info("Pilih minimal satu variabel.")
else:
    lvl = None
    if is_all_region:
        lvl = "Kalimantan"
    elif sel_kab == kabs[0]:
        lvl = sel_prov
    e = load_era5(sel_ids_t, str(e_start.date()), str(e_end.date()), lvl)
    if e.empty:
        st.warning("Tidak ada data ERA5 untuk wilayah ini.")
    else:
        ndays = (e_end - e_start).days + 1
        rule = {"Harian": None, "Mingguan": "W", "Bulanan": "MS"}.get(agg_era)
        if agg_era == "Otomatis":
            rule = None if ndays <= 200 else ("W" if ndays <= 1100 else "MS")
        if rule:
            e = e.resample(rule).mean()
        z = (e[sel_vars] - e[sel_vars].mean()) / e[sel_vars].std(ddof=0).replace(0, np.nan)
        if mode_era.startswith("Terstandarisasi"):
            fig = go.Figure()
            for v in sel_vars:
                fig.add_trace(go.Scatter(
                    x=e.index, y=z[v], name=ERA5_VARS[v][0], mode="lines",
                    line=dict(color=ERA5_VARS[v][1], width=1.8), customdata=e[v],
                    hovertemplate="%{customdata:.2f}<extra>" + ERA5_VARS[v][0] + "</extra>"))
            fig.update_yaxes(title="z-score (simpangan baku dari rata-rata periode)", zeroline=True)
            fig.update_layout(height=380)
        else:
            fig = make_subplots(rows=len(sel_vars), cols=1, shared_xaxes=True, vertical_spacing=0.03)
            for i, v in enumerate(sel_vars, 1):
                fig.add_trace(go.Scatter(x=e.index, y=e[v], name=ERA5_VARS[v][0], mode="lines",
                                         line=dict(color=ERA5_VARS[v][1], width=1.6)), row=i, col=1)
                fig.update_yaxes(title_text=ERA5_VARS[v][0].split(" (")[0], row=i, col=1, title_font_size=11)
            fig.update_layout(height=max(380, 140 * len(sel_vars)))
        fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Rata-rata {'harian' if not rule else ('mingguan' if rule == 'W' else 'bulanan')} seluruh kecamatan terpilih. "
                   "Kecepatan angin = √(u10² + v10²). Pada mode z-score, angka asli muncul di tooltip.")

# ==================================================================================
# 3. PETA INTERAKTIF
# ==================================================================================
st.subheader("3. Peta sebaran hotspot")


@st.cache_data(show_spinner=False)
def map_stats(s_iso: str, e_iso: str):
    s, e = pd.Timestamp(s_iso), pd.Timestamp(e_iso)
    n_days = (e - s).days + 1
    tot = period_slice(combined, s, e).groupby("district_id")["v"].sum()
    out = pd.DataFrame({"total": tot})
    if n_days >= 2:                       # paruh awal vs paruh akhir periode
        mid = s + pd.Timedelta(days=n_days // 2 - 1)
        out["first"] = period_slice(combined, s, mid).groupby("district_id")["v"].sum()
        out["second"] = period_slice(combined, mid + pd.Timedelta(days=1), e).groupby("district_id")["v"].sum()
    else:
        out["first"], out["second"] = np.nan, np.nan
    return out.fillna({"total": 0.0, "first": 0.0, "second": 0.0}), n_days


stats, n_days_p = map_stats(str(start.date()), str(end.date()))
dec = 1 if n_fc_days else 0
info = meta.set_index("district_id")


def change_of(did):
    """-> (jenis, persen). jenis: naik / turun / sama / baru / kosong / pendek"""
    if n_days_p < 2:
        return "pendek", None
    r = stats.loc[did] if did in stats.index else None
    f1, f2 = (float(r["first"]), float(r["second"])) if r is not None else (0.0, 0.0)
    if f1 == 0 and f2 == 0:
        return "kosong", None
    if f1 == 0:
        return "baru", None
    pct = (f2 - f1) / f1 * 100
    return ("sama" if abs(pct) < 0.05 else ("naik" if pct > 0 else "turun")), pct


def change_text(did, html):
    k, pct = change_of(did)
    if k == "naik":
        t, c = f"▲ +{fmt(pct, 1)}%", C_UP
    elif k == "turun":
        t, c = f"▼ {fmt(pct, 1)}%", C_DOWN
    elif k == "baru":
        t, c = "▲ baru muncul", C_UP
    elif k == "sama":
        t, c = "– tidak berubah", C_FLAT
    elif k == "kosong":
        t, c = "– tidak ada hotspot", C_FLAT
    else:
        t, c = "– periode terlalu pendek", C_FLAT
    return f"<span style='color:{c}'>{t}</span>" if html else (t, c)


ids_list = [i for i in meta["district_id"] if i in ids_data]
tot_vals = np.array([float(stats["total"].get(i, 0.0)) for i in ids_list])
z_log = np.log1p(tot_vals)
zmax = float(np.log1p(np.quantile(tot_vals, 0.98))) if tot_vals.max() > 0 else 1.0
zmax = max(zmax, 1e-6)
hover = []
for i, tv in zip(ids_list, tot_vals):
    r = info.loc[i]
    hover.append(f"<b>{r['kecamatan']}</b><br>{r['kabupaten']}, {r['provinsi']}<br>"
                 f"Total hotspot: <b>{fmt(tv, dec)}</b><br>Rata-rata/hari: <b>{fmt(tv / n_days_p, 2)}</b><br>"
                 f"Perubahan awal→akhir: {change_text(i, True)}")
no_data = [i for i in meta["district_id"] if i not in ids_data]

gj = load_geojson()
tick_cand = [t for t in [0, 1, 10, 100, 1000, 10000] if np.log1p(t) <= zmax * 1.001]
fig = go.Figure()
if no_data:
    fig.add_trace(go.Choroplethmap(
        geojson=gj, locations=no_data, featureidkey="properties.district_id", z=[0] * len(no_data),
        colorscale=[[0, "#bdbdbd"], [1, "#bdbdbd"]], showscale=False, marker=dict(opacity=0.8, line=dict(width=0.3, color="#777")),
        text=[f"{info.loc[i]['kecamatan']}: tanpa data" for i in no_data], hoverinfo="text", name="Tanpa data"))
fig.add_trace(go.Choroplethmap(
    geojson=gj, locations=ids_list, featureidkey="properties.district_id", z=np.clip(z_log, 0, zmax),
    zmin=0, zmax=zmax, colorscale=[[0, "#2e9e4f"], [0.5, "#f4d03f"], [1, "#d62728"]],
    marker=dict(opacity=0.85, line=dict(width=0.3, color="#555")),
    hovertext=hover, hoverinfo="text", hoverlabel=dict(bgcolor="white", font=dict(color="#222", size=12), align="left"),
    colorbar=dict(title=dict(text="Total hotspot<br>(skala log)", font=dict(size=11)), thickness=14, len=0.8,
                  tickvals=[float(np.log1p(t)) for t in tick_cand], ticktext=[fmt(t) for t in tick_cand]),
    name="Hotspot"))
if not is_all_region:   # garis tepi wilayah terpilih
    fig.add_trace(go.Choroplethmap(
        geojson=gj, locations=sel_ids, featureidkey="properties.district_id", z=[0] * len(sel_ids),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
        marker=dict(line=dict(width=2.6, color="#000")), hoverinfo="skip"))

# batas tampilan peta mengikuti wilayah terpilih
def bbox(feature_ids):
    xs, ys = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for k in c:
                walk(k)
    fs = set(feature_ids)
    for ft in gj["features"]:
        if ft["properties"]["district_id"] in fs:
            walk(ft["geometry"]["coordinates"])
    return (min(xs), max(xs), min(ys), max(ys)) if xs else (108.8, 119.1, -4.3, 4.4)


bm = st.radio("Peta dasar", ["Polos", "OpenStreetMap", "Carto terang"], horizontal=True, key="basemap",
              help="'Polos' tidak butuh internet. Dua lainnya memuat tile dari server luar.")
BASEMAP = {"Polos": "white-bg", "OpenStreetMap": "open-street-map", "Carto terang": "carto-positron"}[bm]
w, e_, s_, n_ = bbox(sel_ids if not is_all_region else meta["district_id"])
padx, pady = (e_ - w) * 0.05 + 0.02, (n_ - s_) * 0.05 + 0.02
fig.update_layout(
    map=dict(style=BASEMAP, bounds=dict(west=w - padx, east=e_ + padx, south=s_ - pady, north=n_ + pady)),
    height=600, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select")
ev = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="peta", width="stretch")
st.caption("Arahkan kursor untuk ringkasan, **klik kecamatan** untuk rincian di bawah peta. Hijau = sedikit hotspot, merah = banyak "
           "(skala logaritmik karena sebaran sangat miring). Abu-abu = tidak ada data hotspot.")

picked_id = None
try:
    pts = ev.selection.points
    if pts:
        picked_id = pts[0].get("location")
except Exception:
    picked_id = None
if picked_id in info.index:
    r = info.loc[picked_id]
    tv = float(stats["total"].get(picked_id, 0.0)) if picked_id in ids_data else None
    with st.container(border=True):
        st.markdown(f"**{r['kecamatan']}** · {r['kabupaten']}, {r['provinsi']}")
        if tv is None:
            st.write("Tidak ada data hotspot untuk kecamatan ini.")
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("Total hotspot pada periode", fmt(tv, dec))
            p2.metric("Rata-rata per hari", fmt(tv / n_days_p, 2))
            t_, c_ = change_text(picked_id, False)
            name_c = {C_UP: "red", C_DOWN: "green", C_FLAT: "gray"}[c_]
            p3.markdown("Perubahan awal → akhir periode")
            p3.markdown(f"### :{name_c}[{t_}]")
            p3.caption("Total paruh akhir periode dibanding paruh awal. \"Baru muncul\" = tidak ada hotspot di paruh awal.")

# ==================================================================================
# 4. GRAFIK HISTORIS + FORECAST
# ==================================================================================
st.subheader("4. Hotspot historis dan forecast")
cf1, cf2 = st.columns([1, 3])
agg_h = cf1.selectbox("Agregasi", ["Otomatis", "Harian", "Mingguan", "Bulanan"], key="agg_h")

act = hot_all[hot_all["district_id"].isin(sel_ids)].groupby("date")["v"].sum()
a_s, a_e = max(start, FIRST_OBS), min(end, LAST_OBS)
act_s = pd.Series(dtype=float)
if a_s <= a_e:
    act_s = act.reindex(pd.date_range(a_s, a_e), fill_value=0.0)
fc = load_outlook()
fc = fc[fc["district_id"].isin(sel_ids)].groupby("date")["v"].sum()
f_s, f_e = max(start, LAST_OBS + pd.Timedelta(days=1)), min(end, LAST_FC)
fc_s = fc[(fc.index >= f_s) & (fc.index <= f_e)] if f_s <= f_e else pd.Series(dtype=float)

span = (end - start).days + 1
rule_h = {"Harian": None, "Mingguan": "W", "Bulanan": "MS"}.get(agg_h)
if agg_h == "Otomatis":
    rule_h = None if span <= 150 else ("W" if span <= 900 else "MS")
if rule_h:
    act_s = act_s.resample(rule_h).sum() if len(act_s) else act_s
    fc_s = fc_s.resample(rule_h).sum() if len(fc_s) else fc_s
unit = "per hari" if not rule_h else ("per minggu" if rule_h == "W" else "per bulan")

if act_s.empty and fc_s.empty:
    st.warning("Tidak ada data pada periode ini.")
else:
    fig = go.Figure()
    if len(act_s):
        fig.add_trace(go.Scatter(x=act_s.index, y=act_s.values, name="Aktual", mode="lines",
                                 line=dict(color=C_ACTUAL, width=1.8)))
    if len(fc_s):
        fx, fy = list(fc_s.index), list(fc_s.values)
        if len(act_s):   # sambungkan garis agar tidak terputus
            fx, fy = [act_s.index[-1]] + fx, [act_s.values[-1]] + fy
        fig.add_trace(go.Scatter(x=fx, y=fy, name="Forecast", mode="lines",
                                 line=dict(color=C_PRED, width=2, dash="dash")))
    if start <= LAST_OBS <= end:
        fig.add_shape(type="line", x0=LAST_OBS, x1=LAST_OBS, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color="#888", width=1, dash="dot"))
        fig.add_annotation(x=LAST_OBS, y=1.0, yref="paper", text="origin forecast", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color="#888"))
    fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title=f"Jumlah hotspot {unit}", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("Garis putus-putus = prediksi model (outlook harian). Di luar 7 hari pertama, model tidak mengenal cuaca masa depan, "
               "sehingga kurva bersifat skenario klimatologis; andalkan tren mingguan/bulanan.")

# ==================================================================================
# 5. FEATURE IMPORTANCE
# ==================================================================================
st.subheader("5. Fitur paling berpengaruh (top 10)")
LABEL = {
    "days_since_hotspot": "Hari sejak hotspot terakhir",
    "prov_sum_7d": "Total hotspot provinsi, 7 hari",
    "prov_sum_30d": "Total hotspot provinsi, 30 hari",
    "prov_sum_90d": "Total hotspot provinsi, 90 hari",
    "kal_sum_30d": "Total hotspot Kalimantan, 30 hari",
    "kal_sum_90d": "Total hotspot Kalimantan, 90 hari",
    "hotspot_sum_3d": "Total hotspot kecamatan, 3 hari",
    "hotspot_sum_7d": "Total hotspot kecamatan, 7 hari",
    "hotspot_sum_14d": "Total hotspot kecamatan, 14 hari",
    "hotspot_sum_28d": "Total hotspot kecamatan, 28 hari",
    "hotspot_max_14d": "Hotspot harian maks., 14 hari",
    "hotspot_hist_rate_7d": "Laju historis hotspot (7 hari)",
    "day_cos": "Musim (komponen kosinus hari)",
    "day_sin": "Musim (komponen sinus hari)",
    "day_of_year": "Hari dalam tahun",
    "rain_mm_lag_0d": "Curah hujan hari ini",
    "centroid_lat": "Lintang kecamatan",
    "centroid_lon": "Bujur kecamatan",
}


def grup(n):
    if n in ("day_cos", "day_sin", "day_of_year") or n.startswith(("hot_season", "month")):
        return "Kalender/musim"
    if n.startswith("centroid"):
        return "Lokasi"
    if n.startswith(("prov_", "kal_", "neighbor", "share_of")):
        return "Agregat spasial"
    if n.startswith(("hotspot", "days_since", "active_days")):
        return "Riwayat hotspot"
    return "Cuaca/iklim"


G_COL = {"Riwayat hotspot": "#1f3a5f", "Agregat spasial": "#4f8fc0", "Cuaca/iklim": "#2a9d8f",
         "Kalender/musim": "#e9a23b", "Lokasi": "#9b9b9b"}
fi = load_fitur().sort_values("gain_persen", ascending=False).head(10).copy()
fi["Fitur"] = fi["fitur"].map(lambda n: LABEL.get(n, n))
fi["Kelompok"] = fi["fitur"].map(grup)
fig = px.bar(fi.iloc[::-1], x="gain_persen", y="Fitur", color="Kelompok", orientation="h",
             color_discrete_map=G_COL, hover_data={"fitur": True, "gain_persen": ":.2f", "Fitur": False, "Kelompok": False},
             labels={"gain_persen": "Kontribusi gain (%)", "fitur": "Nama teknis"}, text="gain_persen")
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
fig.update_yaxes(categoryorder="array", categoryarray=list(fi["Fitur"].iloc[::-1]))
fig.update_layout(height=400, margin=dict(l=10, r=40, t=10, b=10), yaxis_title=None,
                  legend=dict(orientation="h", y=1.1, title=None))
st.plotly_chart(fig, width="stretch")
st.caption("Gain = kontribusi fitur pada penurunan galat model (rata-rata 3 seed). Statis, tidak ikut filter.")

# ==================================================================================
# 6. TABEL PERINGKAT SIAGA
# ==================================================================================
st.subheader("6. Peringkat kecamatan Siaga, 7 hari ke depan")
w_s, w_e = pd.Timestamp(sg["window_start"].iloc[0]), pd.Timestamp(sg["window_end"].iloc[0])
thr = float(sg["ambang_siaga"].iloc[0])
sg_show = sg_sel[sg_sel["status"] == "Siaga"].sort_values("peringkat")
st.caption(f"Jendela prediksi: **{tgl(w_s)} – {tgl(w_e)}** · Ambang Siaga: ≥ **{fmt(thr, 2)}** hotspot/7 hari/kecamatan · "
           f"**{len(sg_show)}** dari {len(sg_sel)} kecamatan di wilayah terpilih berstatus Siaga.")
if sg_show.empty:
    st.success("Tidak ada kecamatan berstatus Siaga di wilayah terpilih.")
else:
    tbl = sg_show.rename(columns={"peringkat": "Peringkat Kalimantan", "kecamatan": "Kecamatan", "provinsi": "Provinsi",
                                  "pred_hotspot_7d": "Prediksi 7 hari", "aktual_7h_terakhir": "Aktual 7 hari terakhir",
                                  "status": "Status"})[
        ["Peringkat Kalimantan", "Kecamatan", "Provinsi", "Prediksi 7 hari", "Aktual 7 hari terakhir", "Status"]]
    st.dataframe(
        tbl, hide_index=True, width="stretch", height=min(520, 38 + 35 * len(tbl)),
        column_config={
            "Prediksi 7 hari": st.column_config.ProgressColumn(
                "Prediksi 7 hari", format="%.1f", min_value=0, max_value=float(sg["pred_hotspot_7d"].max())),
            "Aktual 7 hari terakhir": st.column_config.NumberColumn(format="%.0f"),
        })

# ==================================================================================
# 7. AKTUAL VS PREDIKSI
# ==================================================================================
st.subheader("7. Aktual vs prediksi")
avp = load_avp()
AV_S, AV_E = avp["date"].min(), avp["date"].max()
v_s, v_e = max(start, AV_S), min(end, AV_E)
if v_s > v_e:
    st.warning(f"Evaluasi aktual vs prediksi tersedia {tgl(AV_S)} – {tgl(AV_E)}. Pilih periode di dalam rentang itu.")
else:
    d = avp[(avp["date"] >= v_s) & (avp["date"] <= v_e) & (avp["district_id"].isin(sel_ids))]
    one_kec = len(sel_ids) == 1
    g = d.groupby("date", observed=True)[["actual_7d", "pred_final"]].mean()
    if g.empty:
        st.warning("Tidak ada data evaluasi untuk wilayah/periode ini.")
    else:
        mean_a, mean_p = g["actual_7d"].mean(), g["pred_final"].mean()
        bias = (mean_p - mean_a) / mean_a * 100 if mean_a > 0 else np.nan
        x1, x2, x3 = st.columns(3)
        lbl = "kecamatan" if one_kec else "kecamatan (rata-rata wilayah)"
        x1.metric(f"Rata-rata aktual / {lbl}", fmt(mean_a, 2), help="Rata-rata hotspot 7 hari ke depan per kecamatan-hari")
        x2.metric(f"Rata-rata prediksi / {lbl}", fmt(mean_p, 2))
        x3.metric("Bias prediksi", f"{fmt(bias, 1)}%" if not np.isnan(bias) else "–",
                  help="(prediksi − aktual) / aktual. Negatif = model meremehkan.")
        span_v = (v_e - v_s).days + 1
        rv = None if span_v <= 150 else ("W" if span_v <= 900 else "MS")
        gg = g.resample(rv).mean() if rv else g
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gg.index, y=gg["actual_7d"], name="Aktual", mode="lines",
                                 line=dict(color=C_ACTUAL, width=1.8)))
        fig.add_trace(go.Scatter(x=gg.index, y=gg["pred_final"], name="Prediksi model", mode="lines",
                                 line=dict(color=C_PRED, width=1.8)))
        if v_e >= pd.Timestamp("2026-01-01") and v_s < pd.Timestamp("2026-01-01"):
            fig.add_shape(type="line", x0="2026-01-01", x1="2026-01-01", y0=0, y1=1, xref="x", yref="paper",
                          line=dict(color="#888", width=1, dash="dot"))
            fig.add_annotation(x="2026-01-01", y=1.0, yref="paper", text="holdout murni →", showarrow=False,
                               xanchor="left", yanchor="bottom", font=dict(size=11, color="#888"))
        fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
                          yaxis_title="Hotspot 7 hari ke depan" + (" (per kecamatan)" if not one_kec else ""),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Tiap titik = tanggal asal prediksi; nilainya total hotspot 7 hari sesudahnya. Rentang evaluasi: {tgl(AV_S)} – {tgl(AV_E)}. "
                   "Mulai 1 Jan 2026 = holdout murni; sebelumnya walk-forward (sebagian tahun ikut dipakai saat memilih skema model).")

# ----------------------------------------------------------------------------------
st.divider()
st.caption(
    "Catatan: model memprediksi rata-rata bersyarat (Tweedie) dan cenderung meremehkan puncak pada tahun kebakaran besar "
    "(bias holdout 2026 ≈ −49%) serta melebihkan pada tahun tenang. Data hotspot hanya berisi baris positif, jadi hari tanpa "
    "hotspot dianggap 0 (tidak dibedakan dari hari tanpa pengamatan satelit). Sumber: MODIS FIRMS, ERA5, batas BPS ADM3."
)
