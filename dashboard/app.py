"""
Dashboard Peramalan Hotspot Kalimantan (Streamlit)

Jalankan dari root repo:
    python dashboard/build_data.py          # sekali saja (membuat data ERA5 ringan)
    streamlit run dashboard/app.py
"""
import base64
import html
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dashboard Hotspot Kalimantan", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------------
# Konstanta
# ----------------------------------------------------------------------------------
AUTHOR = "Rafif Danuja"                                   # ubah sesuai namamu
AFFIL = "Statistika dan Sains Data · IPB University"

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
PROV_SHORT = {"Kalimantan Barat": "Kalbar", "Kalimantan Tengah": "Kalteng", "Kalimantan Selatan": "Kalsel",
              "Kalimantan Timur": "Kaltim", "Kalimantan Utara": "Kaltara"}
ERA5_VARS = {
    "rain_mm": ("Curah hujan (mm)", "#1f6fb5"),
    "t2m_c": ("Suhu udara 2 m (°C)", "#e8710a"),
    "d2m_c": ("Titik embun (°C)", "#0f8b8d"),
    "wind_ms": ("Angin (m/s)", "#7b61a8"),
    "swvl1": ("Kelembapan tanah (m³/m³)", "#9a7b2f"),
}
NAVY, BLUE, ORANGE, TEAL = "#0b2a4a", "#1f6fb5", "#e8710a", "#0f8b8d"
INK, INK2, GRID = "#10243b", "#44566c", "#e6ebf1"
C_UP, C_DOWN, C_FLAT = "#d62728", "#2e9e4f", "#7f7f7f"      # naik = merah, turun = hijau
FONT = "Poppins, 'Segoe UI', sans-serif"

GLOSS = {
    "AUC": "ROC-AUC: kemampuan membedakan kecamatan-hari yang ada hotspot vs tidak. 1 = sempurna, 0,5 = acak.",
    "PR-AUC": "Presisi-recall pada kejadian hotspot yang jarang (±1% sel kecamatan-hari). Lebih jujur daripada AUC untuk data tidak seimbang.",
    "RMSE": "Galat kuadrat rata-rata (hotspot per 7 hari). Menghukum meleset besar di puncak. Makin kecil makin baik.",
    "MAE": "Galat absolut rata-rata (hotspot per 7 hari). Pada data 99% nol, prediktor nol pun bisa kecil MAE-nya; baca bersama metrik lain.",
    "Siaga": "Kecamatan dengan prediksi hotspot 7 hari ke depan di atas ambang Siaga (dari distribusi kejadian historis).",
}


def fmt(x, d=0):
    """Format angka gaya Indonesia: 1.234,5"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def tgl(d):
    return pd.Timestamp(d).strftime("%d %b %Y")


def esc(x):
    return html.escape(str(x))


def tip(txt, key):
    return f'<span class="tp" title="{html.escape(GLOSS.get(key, ""), quote=True)}">{txt}</span>'


# ----------------------------------------------------------------------------------
# Gaya (CSS)
# ----------------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#eef2f7 !important; color:§INK; color-scheme:light; font-family:§FONT;}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], footer, #MainMenu {display:none !important;}
.block-container, [data-testid="stMainBlockContainer"] {padding:0 .8rem 1rem .8rem !important; max-width:100% !important;}
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {gap:.55rem !important;}
div[data-testid="stElementContainer"] {margin:0 !important; flex-shrink:0 !important;}
[class*="st-key-card_"], [class*="st-key-head_"], .st-key-filt {flex:0 0 auto !important;}
[class*="st-key-card_"] > *, [class*="st-key-head_"] > * {flex-shrink:0 !important;}
div[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {display:none !important;}
[data-testid="stMarkdown"], [data-testid="stMarkdownContainer"] {margin:0 !important;}
[data-testid="stMarkdownContainer"] p {margin:0 !important;}
/* header */
.hdr {margin:0 -.8rem .55rem; min-height:56px; background:#fff; display:flex; align-items:center; gap:12px; padding:6px .8rem; border-bottom:3px solid §ORANGE; box-sizing:border-box; box-shadow:0 1px 6px rgba(11,42,74,.10);}
.hdr .lg img {height:42px; width:auto; display:block;}
.hdr .ttl {flex:1; text-align:center; min-width:0;}
.hdr h1 {font-size:18px; margin:0; padding:0; font-weight:700; color:§NAVY; line-height:1.2; letter-spacing:.2px;}
.hdr .sub {font-size:11px; color:#5b6b7f;}
.hdr .bx {flex:none; border:1px solid #d3dde9; border-radius:8px; padding:4px 14px; font-size:11px; line-height:1.3; text-align:center; color:#41546b; background:#f4f7fb;}
.hdr .bx b {color:§NAVY; font-size:12px;}
/* filter */
.st-key-filt {background:§NAVY; border-radius:10px; padding:8px 12px 10px !important; box-sizing:border-box;}
.st-key-filt [data-testid="stWidgetLabel"], .st-key-filt [data-testid="stWidgetLabel"] * {color:#b9cde3 !important; -webkit-text-fill-color:#b9cde3 !important;}
.st-key-filt [data-baseweb="select"] > div, .st-key-filt [data-baseweb="input"], .st-key-filt [data-baseweb="base-input"],
.st-key-filt [data-testid="stDateInputField"], .st-key-filt div[role="group"] {background:#15395f !important; border-color:rgba(255,255,255,.22) !important; color:#fff !important;}
.st-key-filt input, .st-key-filt [data-baseweb="select"] div, .st-key-filt [data-baseweb="select"] span,
.st-key-filt [data-testid="stDateInput"] * {color:#fff !important; -webkit-text-fill-color:#fff !important;}
.st-key-filt [data-baseweb="select"] svg, .st-key-filt [data-testid="stDateInput"] svg {fill:#fff !important; color:#fff !important;}
.st-key-filt [data-baseweb="select"] > div[aria-disabled="true"], .st-key-filt [data-baseweb="select"][aria-disabled="true"] > div {opacity:.5;}
.st-key-filt button {background:transparent !important; color:#fff !important; border:1px solid rgba(255,255,255,.55) !important; border-radius:6px !important;}
.st-key-filt button * {color:#fff !important; -webkit-text-fill-color:#fff !important;}
.st-key-filt button:hover {background:rgba(255,255,255,.12) !important;}
[data-testid="stWidgetLabel"] p {font-size:10.5px !important; text-transform:uppercase; letter-spacing:.4px; font-weight:600;}
[data-testid="stWidgetLabel"] {min-height:0 !important; margin:0 0 2px 0 !important;}
[data-testid="stSelectbox"] *, [data-testid="stDateInput"] * {font-size:12px !important;}
div[data-baseweb="select"] > div, div[data-baseweb="input"], [data-testid="stDateInputField"] {min-height:34px !important; border-radius:6px !important;}
.stButton button, [data-testid="stPopover"] > button {min-height:34px !important; padding:0 10px !important;}
/* KPI */
.sect {display:flex; align-items:baseline; gap:8px; font-size:13px; font-weight:700; color:§NAVY; text-transform:uppercase; letter-spacing:.3px; padding:2px 2px 0;}
.sect span {font-weight:500; color:#5b6b7f; text-transform:none; letter-spacing:0; font-size:11px;}
.kpis {display:grid; grid-template-columns:repeat(6,1fr); gap:.55rem;}
.kpi {background:#fff; border:1px solid #dde4ec; border-top:3px solid #c5d3e2; border-radius:10px; padding:6px 12px 7px; min-width:0; overflow:hidden; box-shadow:0 1px 2px rgba(11,42,74,.06); font-variant-numeric:tabular-nums;}
.kpi.key {background:#f0f8f8; border-color:#bfe0e0; border-top-color:§TEAL;}
.kpi.hot {background:#fff6ec; border-color:#f3d4b0; border-top-color:§ORANGE;}
.kpi .l {font-size:10.5px; color:§INK2; font-weight:600; text-transform:uppercase; letter-spacing:.3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.kpi .v {font-size:22px; font-weight:700; line-height:1.25; color:§INK; white-space:nowrap;}
.kpi .s {font-size:11px; line-height:1.3; color:§INK2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.tp {border-bottom:1px dotted #5b6b7f; cursor:help;}
.chip {display:inline-block; font-size:10.5px; font-weight:700; padding:0 7px; border-radius:9px; vertical-align:middle; margin-right:4px;}
.chip.good {background:#d9eeee; color:#0a6a6c;} .chip.bad {background:#fdebd9; color:#a14a00;} .chip.up {background:#fde3e3; color:#b01e1e;} .chip.down {background:#dff3e5; color:#1d7a3a;} .chip.flat {background:#e8ecf1; color:#5b6b7f;}
/* kartu */
[class*="st-key-card_"] {background:#fff; border:1px solid #dde4ec; border-radius:10px; box-shadow:0 1px 2px rgba(11,42,74,.06); overflow:hidden; gap:0 !important; padding:0 !important;}
[class*="st-key-head_"] {padding:7px 10px 0 !important;}
[class*="st-key-head_"] [data-testid="stHorizontalBlock"] {align-items:center;}
.ct {font-size:12.5px; font-weight:600; color:§NAVY; line-height:1.3;}
.ct span {font-weight:500; color:§INK2; font-size:11px; margin-left:8px; background:#eef2f7; border-radius:10px; padding:1px 9px; white-space:nowrap;}
.ins {font-size:11.5px; color:§INK2; padding:2px 12px 0; line-height:1.4;} .ins b {color:§NAVY;}
.note {font-size:11px; line-height:1.35; color:#5b6b7f; padding:0 12px 8px;}
[data-testid="stPlotlyChart"] {padding:0 4px;}
[class*="st-key-card_"] [data-testid="stSegmentedControl"] button, [class*="st-key-card_"] [data-baseweb="button-group"] button {min-height:26px !important; height:26px !important; padding:0 10px !important;}
[class*="st-key-card_"] [data-testid="stSegmentedControl"] p, [class*="st-key-card_"] [data-baseweb="button-group"] p {font-size:11px !important;}
/* ringkasan peta */
.sumw {display:grid; grid-template-columns:repeat(4,1fr); gap:8px; padding:4px 12px 10px;}
.sum {border-left:3px solid §BLUE; padding-left:8px; min-width:0; overflow:hidden;}
.sum .l {font-size:10.5px; color:#5b6b7f; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.sum .v {font-size:13.5px; font-weight:700; color:§INK; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.25; font-variant-numeric:tabular-nums;}
.sum .s {font-size:11px; color:§INK2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
/* tabel Siaga */
.tbw {max-height:290px; overflow-y:auto; padding:0 8px 8px;}
table.ev {border-collapse:collapse; width:100%; font-size:11px; font-variant-numeric:tabular-nums;}
table.ev th {position:sticky; top:0; background:#fff; font-size:10.5px; text-transform:uppercase; color:#5b6b7f; font-weight:600; text-align:right; padding:3px 8px; border-bottom:1px solid #dde4ec;}
table.ev th:nth-child(-n+2), table.ev td:nth-child(-n+2) {text-align:left;}
table.ev td {padding:4px 6px; text-align:right; border-bottom:1px solid #eef2f7; white-space:nowrap; color:§INK;}
table.ev tr:hover td {background:#f4f8fc;}
.bar {display:inline-block; height:6px; background:§TEAL; border-radius:3px; vertical-align:middle; margin-right:6px; opacity:.7;}
/* footer */
.foot {display:grid; grid-template-columns:auto 1fr; gap:.55rem; margin-top:.2rem;}
.foot > div {background:#fff; border:1px solid #dde4ec; border-radius:8px; display:flex; align-items:center; padding:7px 12px; font-size:11px; color:§INK2; gap:8px; flex-wrap:wrap;}
.foot .pill {background:§NAVY; color:#d7e4f2; padding:3px 10px; border-radius:12px;}
.foot .dot {display:inline-block; width:7px; height:7px; border-radius:50%; background:#5ad18f; margin-right:6px;}
.foot .w {color:#8a4300;}
@media (max-width: 1200px) {.kpis {grid-template-columns:repeat(3,1fr);} .sumw {grid-template-columns:repeat(2,1fr);}}
@media (max-width: 700px) {.kpis {grid-template-columns:repeat(2,1fr);} .hdr .bx {display:none;}}
</style>
"""
for _k, _v in {"§INK2": INK2, "§INK": INK, "§NAVY": NAVY, "§ORANGE": ORANGE, "§TEAL": TEAL, "§BLUE": BLUE, "§FONT": FONT}.items():
    _CSS = _CSS.replace(_k, _v)
st.markdown(_CSS, unsafe_allow_html=True)


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
    return pd.DataFrame([f["properties"] for f in gj["features"]])[["district_id", "kecamatan", "kabupaten", "provinsi"]]


@st.cache_data(show_spinner=False)
def load_hotspot():
    h = pd.read_csv(P["hotspot"], parse_dates=["tanggal"])
    return h.rename(columns={"kec_pcode": "district_id", "tanggal": "date", "hotspot_count": "v"})[["district_id", "date", "v"]]


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
    df = pq.read_table(P["avp"], columns=["date", "district_id", "actual_7d", "pred_final"]).to_pandas()
    df["district_id"] = df["district_id"].astype("category")
    return df


@st.cache_data(show_spinner="Memuat data iklim ERA5…")
def load_era5(ids: tuple, start: str, end: str, level_prov: str | None):
    """Rata-rata ERA5 harian untuk wilayah terpilih (agregat provinsi bila seluruh provinsi/Kalimantan)."""
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
        frames.append(pq.read_table(fp, filters=[("district_id", "in", sel), ("date", ">=", s), ("date", "<=", e)]).to_pandas())
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames).groupby("date")[cols].mean().sort_index()


# ----------------------------------------------------------------------------------
# Pemeriksaan file
# ----------------------------------------------------------------------------------
missing = [k for k, v in P.items() if not os.path.exists(v)]
kec_files = [os.path.join(HERE, "data", f"era5_kecamatan_{k}.parquet") for k in PROV_NAMA]
if missing or not all(os.path.exists(f) for f in kec_files):
    st.error("File data belum lengkap. Jalankan `python dashboard/build_data.py` dari root repo, lalu commit folder "
             "`dashboard/data/`. Pastikan juga `kecamatan_kalimantan.geojson` ada di root repo.")
    if missing:
        st.write("File yang tidak ditemukan:", {k: P[k] for k in missing})
    st.stop()

meta = load_meta()
combined = load_combined()
hot_all = load_hotspot()
LAST_OBS = hot_all["date"].max()
LAST_FC = load_outlook()["date"].max()
FIRST_OBS = hot_all["date"].min()
ERA5_END = pd.Timestamp("2026-09-23")
ids_data = set(hot_all["district_id"].unique())
meta_d = meta[meta["district_id"].isin(ids_data)].copy()
info = meta.set_index("district_id")

# ----------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------


def _logo(fn):
    fp = os.path.join(HERE, "assets", fn)
    if not os.path.exists(fp):
        return ""
    with open(fp, "rb") as f:
        return f'<div class="lg"><img src="data:image/png;base64,{base64.b64encode(f.read()).decode()}"></div>'


st.markdown(
    f"""<div class="hdr">{_logo("logo_ipb.png")}
    <div class="ttl"><h1>Dashboard Peramalan Hotspot Kalimantan</h1>
    <div class="sub">Hotspot kecamatan, iklim ERA5, dan performa model LightGBM untuk 5 provinsi</div></div>
    <div class="bx"><b>{esc(AUTHOR)}</b><br>{esc(AFFIL)}</div></div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# Filter (bar navy)
# ----------------------------------------------------------------------------------
PRESETS = ["90 hari terakhir data", "30 hari terakhir data", "Musim kebakaran 2026 (Jun–Des)",
           "Tahun 2026", "Tahun 2023", "Seluruh data", "Kustom"]
ALL_PROV, ALL_KAB, ALL_KEC = "Semua Kalimantan", "Semua kabupaten/kota", "Semua kecamatan"


def preset_range(name):
    if name == "90 hari terakhir data":
        return LAST_OBS - pd.Timedelta(days=89), LAST_OBS
    if name == "30 hari terakhir data":
        return LAST_OBS - pd.Timedelta(days=29), LAST_OBS
    if name == "Musim kebakaran 2026 (Jun–Des)":
        return pd.Timestamp("2026-06-01"), LAST_FC
    if name == "Tahun 2026":
        return pd.Timestamp("2026-01-01"), LAST_FC
    if name == "Tahun 2023":
        return pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    return FIRST_OBS, LAST_FC


def _set_preset_rng():
    n = st.session_state["f_preset"]
    if n != "Kustom":
        a, b = preset_range(n)
        st.session_state["f_rng"] = (a.date(), b.date())


def _custom():
    st.session_state["f_preset"] = "Kustom"


def _reset_kab():
    st.session_state["f_kab"], st.session_state["f_kec"] = ALL_KAB, ALL_KEC


def _reset_kec():
    st.session_state["f_kec"] = ALL_KEC


def _reset_all():
    st.session_state["f_preset"] = PRESETS[0]
    a, b = preset_range(PRESETS[0])
    st.session_state["f_rng"] = (a.date(), b.date())
    st.session_state["f_prov"], st.session_state["f_kab"], st.session_state["f_kec"] = ALL_PROV, ALL_KAB, ALL_KEC


if "f_preset" not in st.session_state:
    _reset_all()

prov_opts = [ALL_PROV] + [PROV_NAMA[k] for k in sorted(PROV_NAMA)]
m1 = meta_d if st.session_state["f_prov"] == ALL_PROV else meta_d[meta_d["provinsi"] == st.session_state["f_prov"]]
kab_opts = [ALL_KAB] + sorted(m1["kabupaten"].unique())
if st.session_state["f_kab"] not in kab_opts:
    st.session_state["f_kab"] = ALL_KAB
m2 = m1 if st.session_state["f_kab"] == ALL_KAB else m1[m1["kabupaten"] == st.session_state["f_kab"]]
m2 = m2.sort_values("kecamatan")
_lab = {r.district_id: r.kecamatan for r in m2.itertuples()}
_dup = pd.Series(list(_lab.values())).duplicated(keep=False).tolist()
kec_label = {k: (v if not d else f"{v} ({k})") for (k, v), d in zip(_lab.items(), _dup)}
kec_opts = [ALL_KEC] + list(kec_label.values())
if st.session_state["f_kec"] not in kec_opts:
    st.session_state["f_kec"] = ALL_KEC

with st.container(key="filt"):
    f1, f2, f3, f4, f5, f6, f7 = st.columns([1.35, 1.7, 1.15, 1.2, 1.2, 0.55, 0.65], vertical_alignment="bottom")
    f1.selectbox("Preset periode", PRESETS, key="f_preset", on_change=_set_preset_rng)
    f2.date_input("Rentang waktu", key="f_rng", min_value=FIRST_OBS.date(), max_value=LAST_FC.date(),
                  format="DD/MM/YYYY", on_change=_custom)
    f3.selectbox("Provinsi", prov_opts, key="f_prov", on_change=_reset_kab)
    f4.selectbox("Kabupaten/Kota", kab_opts, key="f_kab", on_change=_reset_kec,
                 disabled=(st.session_state["f_prov"] == ALL_PROV))
    f5.selectbox("Kecamatan", kec_opts, key="f_kec", disabled=(st.session_state["f_kab"] == ALL_KAB))
    f6.button("Reset", on_click=_reset_all, width="stretch")
    with f7.popover("Catatan", width="stretch"):
        st.markdown(
            "- **Total hotspot**: aktual sampai data terakhir; hari sesudahnya memakai prediksi outlook model.\n"
            "- **Perubahan di peta**: total paruh akhir periode dibanding paruh awal. Naik = merah, turun = hijau.\n"
            "- **Siaga**: prediksi 7 hari ke depan ≥ ambang dari distribusi historis.\n"
            "- **Aktual vs prediksi**: tiap titik = tanggal asal prediksi, nilainya total hotspot 7 hari sesudahnya.\n"
            "- Model memprediksi rata-rata bersyarat (Tweedie): cenderung **meremehkan puncak** di tahun api besar "
            "dan melebihkan di tahun tenang.\n"
            "- Hotspot di luar 7 hari pertama adalah skenario klimatologis (tanpa cuaca masa depan).\n"
            "- Data hotspot hanya baris positif; hari tanpa hotspot dianggap 0.")

rng = st.session_state["f_rng"]
if not isinstance(rng, (tuple, list)) or len(rng) != 2:
    st.info("Pilih tanggal akhir pada kalender untuk melengkapi rentang waktu.")
    st.stop()
start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])

sel_prov, sel_kab, sel_kec = st.session_state["f_prov"], st.session_state["f_kab"], st.session_state["f_kec"]
is_all_region = sel_prov == ALL_PROV
if sel_kec != ALL_KEC:
    sel_ids = [k for k, v in kec_label.items() if v == sel_kec]
    region_name = f"{sel_kec}, {sel_kab}"
elif sel_kab != ALL_KAB:
    sel_ids, region_name = list(m2["district_id"]), f"{sel_kab}, {sel_prov}"
elif not is_all_region:
    sel_ids, region_name = list(m2["district_id"]), sel_prov
else:
    sel_ids, region_name = list(meta_d["district_id"]), "Seluruh Kalimantan"
sel_ids_t = tuple(sorted(sel_ids))
sel_set = set(sel_ids)
n_fc_days = max(0, (min(end, LAST_FC) - max(start, LAST_OBS + pd.Timedelta(days=1))).days + 1)
n_days_p = (end - start).days + 1


def period_slice(df, s, e, ids=None):
    m = (df["date"] >= s) & (df["date"] <= e)
    if ids is not None:
        m &= df["district_id"].isin(ids)
    return df[m]


def total_in(s, e, ids):
    return float(period_slice(combined, s, e, ids)["v"].sum())


def style_fig(fig, h, legend=True):
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=26 if legend else 6, b=6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=INK2), hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=11, color=INK)),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)) if legend else dict(visible=False))
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


CFG = dict(displaylogo=False, displayModeBar=False)

# ----------------------------------------------------------------------------------
# KPI
# ----------------------------------------------------------------------------------
sec_l, sec_r = st.columns([3, 1.6], vertical_alignment="center")
sec_l.markdown(f'<div class="sect">Performa model dan ringkasan <span>{esc(region_name)} · {tgl(start)} – {tgl(end)}</span></div>',
               unsafe_allow_html=True)
split_pick = sec_r.segmented_control("Evaluasi", ["Holdout 2026", "Walk-forward 2015–25"], default="Holdout 2026",
                                     key="f_split", label_visibility="collapsed") or "Holdout 2026"
split_key = "holdout" if split_pick.startswith("Holdout") else "walk-forward"
mt = load_metrik()
mm = mt[(mt["split"] == split_key) & (mt["model"] == "Model")].iloc[0]
bs = mt[(mt["split"] == split_key) & (mt["model"] == "Persistensi 7h")].iloc[0]


def model_card(label, key, col, higher_better):
    d = mm[col] - bs[col]
    good = (d > 0) if higher_better else (d < 0)
    chip = f'<span class="chip {"good" if good else "bad"}">{"lebih baik" if good else "lebih buruk"}</span>'
    return (f'<div class="kpi key"><div class="l">{tip(label, key)}</div><div class="v">{fmt(mm[col], 3)}</div>'
            f'<div class="s">{chip}persistensi {fmt(bs[col], 3)}</div></div>')


tot_now = total_in(start, end, sel_ids)
prev_len = n_days_p
prev_s, prev_e = start - pd.Timedelta(days=prev_len), start - pd.Timedelta(days=1)
tot_prev = total_in(prev_s, prev_e, sel_ids)
if tot_prev > 0:
    chg = (tot_now - tot_prev) / tot_prev * 100
    kind = "up" if chg > 0.05 else ("down" if chg < -0.05 else "flat")
    arrow = {"up": "▲ +", "down": "▼ ", "flat": "– "}[kind]
    prev_chip = f'<span class="chip {kind}">{arrow}{fmt(chg, 1)}%</span>vs periode sebelumnya'
else:
    prev_chip = '<span class="chip flat">–</span>periode sebelumnya tanpa hotspot'
sg = load_siaga()
sg_sel = sg[sg["district_id"].isin(sel_ids)]
n_siaga = int((sg_sel["status"] == "Siaga").sum())
fc_note = f" · termasuk {n_fc_days} hari prediksi" if n_fc_days else ""
st.markdown(
    '<div class="kpis">'
    + model_card("AUC", "AUC", "roc_auc", True) + model_card("PR-AUC", "PR-AUC", "pr_auc", True)
    + model_card("RMSE", "RMSE", "rmse", False) + model_card("MAE", "MAE", "mae", False)
    + f'<div class="kpi hot"><div class="l">Total hotspot periode</div><div class="v">{fmt(tot_now, 1 if n_fc_days else 0)}</div>'
      f'<div class="s">{prev_chip}</div></div>'
    + f'<div class="kpi"><div class="l">{tip("Kecamatan Siaga", "Siaga")}</div><div class="v">{n_siaga} <small style="font-size:13px;color:{INK2}">dari {len(sg_sel)}</small></div>'
      f'<div class="s">prediksi 7 hari ke depan</div></div>'
    + "</div>", unsafe_allow_html=True)
st.caption(f"Selisih terhadap baseline persistensi 7 hari ({split_pick}). Bias total model: {fmt(mm['bias_pct'], 1)}% "
           f"(negatif = meremehkan jumlah hotspot){fc_note}.")


# ----------------------------------------------------------------------------------
# Peta: statistik periode
# ----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def map_stats(s_iso: str, e_iso: str):
    s, e = pd.Timestamp(s_iso), pd.Timestamp(e_iso)
    n = (e - s).days + 1
    out = pd.DataFrame({"total": period_slice(combined, s, e).groupby("district_id")["v"].sum()})
    if n >= 2:
        mid = s + pd.Timedelta(days=n // 2 - 1)
        out["first"] = period_slice(combined, s, mid).groupby("district_id")["v"].sum()
        out["second"] = period_slice(combined, mid + pd.Timedelta(days=1), e).groupby("district_id")["v"].sum()
    else:
        out["first"], out["second"] = np.nan, np.nan
    return out.fillna({"total": 0.0, "first": 0.0, "second": 0.0})


stats = map_stats(str(start.date()), str(end.date()))
dec = 1 if n_fc_days else 0


def change_of(f1_, f2_):
    if n_days_p < 2:
        return "pendek", None
    if f1_ == 0 and f2_ == 0:
        return "kosong", None
    if f1_ == 0:
        return "baru", None
    pct = (f2_ - f1_) / f1_ * 100
    return ("sama" if abs(pct) < 0.05 else ("naik" if pct > 0 else "turun")), pct


def change_text(f1_, f2_, html_=True):
    k, pct = change_of(f1_, f2_)
    t, c = {"naik": (f"▲ +{fmt(pct, 1)}%" if pct is not None else "", C_UP), "turun": (f"▼ {fmt(pct, 1)}%" if pct is not None else "", C_DOWN),
            "baru": ("▲ baru muncul", C_UP), "sama": ("– tidak berubah", C_FLAT), "kosong": ("– tidak ada hotspot", C_FLAT),
            "pendek": ("– periode terlalu pendek", C_FLAT)}[k]
    return f"<span style='color:{c}'>{t}</span>" if html_ else t


def f12(did):
    if did in stats.index:
        r = stats.loc[did]
        return float(r["first"]), float(r["second"])
    return 0.0, 0.0


# ----------------------------------------------------------------------------------
# Tata letak utama
# ----------------------------------------------------------------------------------
L, R = st.columns([1.45, 1], gap="small")

# ======================= KIRI =======================
with L:
    st.markdown('<div class="sect">Forecast dan evaluasi model <span>historis, prediksi, kecamatan Siaga</span></div>', unsafe_allow_html=True)

    # ---- Hotspot historis + forecast
    with st.container(key="card_ts"):
        with st.container(key="head_ts"):
            h1, h2 = st.columns([2.2, 1.6], vertical_alignment="center")
            h1.markdown(f'<div class="ct">Hotspot historis dan forecast <span>{esc(region_name)}</span></div>', unsafe_allow_html=True)
            agg_h = h2.segmented_control("Agregasi", ["Otomatis", "Harian", "Mingguan", "Bulanan"], default="Otomatis",
                                         key="agg_h", label_visibility="collapsed") or "Otomatis"
        act = hot_all[hot_all["district_id"].isin(sel_ids)].groupby("date")["v"].sum()
        a_s, a_e = max(start, FIRST_OBS), min(end, LAST_OBS)
        act_s = act.reindex(pd.date_range(a_s, a_e), fill_value=0.0) if a_s <= a_e else pd.Series(dtype=float)
        fc_all = load_outlook()
        fc = fc_all[fc_all["district_id"].isin(sel_ids)].groupby("date")["v"].sum()
        f_s, f_e = max(start, LAST_OBS + pd.Timedelta(days=1)), min(end, LAST_FC)
        fc_s = fc[(fc.index >= f_s) & (fc.index <= f_e)] if f_s <= f_e else pd.Series(dtype=float)
        rule_h = {"Harian": None, "Mingguan": "W", "Bulanan": "MS"}.get(agg_h)
        if agg_h == "Otomatis":
            rule_h = None if n_days_p <= 150 else ("W" if n_days_p <= 900 else "MS")
        if rule_h:
            act_s = act_s.resample(rule_h).sum() if len(act_s) else act_s
            fc_s = fc_s.resample(rule_h).sum() if len(fc_s) else fc_s
        unit = "per hari" if not rule_h else ("per minggu" if rule_h == "W" else "per bulan")
        if act_s.empty and fc_s.empty:
            st.warning("Tidak ada data pada periode ini.")
        else:
            ta, tf = float(act_s.sum()), float(fc_s.sum())
            parts = []
            if len(act_s):
                parts.append(f"Aktual: <b>{fmt(ta)}</b> hotspot")
            if len(fc_s):
                parts.append(f"Prediksi: <b>{fmt(tf)}</b> hotspot")
            st.markdown(f'<div class="ins">{" · ".join(parts)} · satuan {unit}</div>', unsafe_allow_html=True)
            fig = go.Figure()
            if len(fc_s):
                fig.add_vrect(x0=LAST_OBS if len(act_s) else fc_s.index[0], x1=fc_s.index[-1], fillcolor=TEAL, opacity=0.07, line_width=0)
            if len(act_s):
                fig.add_trace(go.Scatter(x=act_s.index, y=act_s.values, name="Aktual", mode="lines", line=dict(color=NAVY, width=1.8)))
            if len(fc_s):
                fx, fy = list(fc_s.index), list(fc_s.values)
                if len(act_s):
                    fx, fy = [act_s.index[-1]] + fx, [act_s.values[-1]] + fy
                fig.add_trace(go.Scatter(x=fx, y=fy, name="Forecast", mode="lines", line=dict(color=TEAL, width=2, dash="dash")))
            if start <= LAST_OBS <= end and len(fc_s):
                fig.add_annotation(x=LAST_OBS, y=1.0, yref="paper", text="origin forecast →", showarrow=False, xanchor="right",
                                   yanchor="bottom", font=dict(size=10, color=TEAL))
            style_fig(fig, 280)
            st.plotly_chart(fig, width="stretch", config=CFG)
            st.markdown('<div class="note">Garis putus-putus = prediksi model. Di luar 7 hari pertama model tidak mengenal cuaca masa depan, '
                        'jadi kurva bersifat skenario klimatologis; andalkan tren mingguan/bulanan.</div>', unsafe_allow_html=True)

    # ---- Aktual vs prediksi
    with st.container(key="card_avp"):
        with st.container(key="head_avp"):
            st.markdown(f'<div class="ct">Aktual vs prediksi model <span>hotspot 7 hari ke depan · {esc(region_name)}</span></div>',
                        unsafe_allow_html=True)
        avp = load_avp()
        AV_S, AV_E = avp["date"].min(), avp["date"].max()
        v_s, v_e = max(start, AV_S), min(end, AV_E)
        if v_s > v_e:
            st.markdown(f'<div class="note" style="padding-top:10px">Evaluasi tersedia {tgl(AV_S)} – {tgl(AV_E)}. Pilih periode di dalam rentang itu.</div>',
                        unsafe_allow_html=True)
        else:
            d = avp[(avp["date"] >= v_s) & (avp["date"] <= v_e) & (avp["district_id"].isin(sel_ids))]
            g = d.groupby("date", observed=True)[["actual_7d", "pred_final"]].mean()
            if g.empty:
                st.markdown('<div class="note" style="padding-top:10px">Tidak ada data evaluasi untuk wilayah ini.</div>', unsafe_allow_html=True)
            else:
                one = len(sel_ids) == 1
                mean_a, mean_p = float(g["actual_7d"].mean()), float(g["pred_final"].mean())
                bias = (mean_p - mean_a) / mean_a * 100 if mean_a > 0 else np.nan
                st.markdown(
                    f'<div class="ins">Rata-rata aktual <b>{fmt(mean_a, 2)}</b> · prediksi <b>{fmt(mean_p, 2)}</b> '
                    f'{"per kecamatan-hari" if not one else "per hari"} · bias <b>{fmt(bias, 1) + "%" if not np.isnan(bias) else "–"}</b> '
                    f'(negatif = meremehkan)</div>', unsafe_allow_html=True)
                span_v = (v_e - v_s).days + 1
                rv = None if span_v <= 150 else ("W" if span_v <= 900 else "MS")
                gg = g.resample(rv).mean() if rv else g
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=gg.index, y=gg["actual_7d"], name="Aktual", mode="lines", line=dict(color=NAVY, width=1.8)))
                fig.add_trace(go.Scatter(x=gg.index, y=gg["pred_final"], name="Prediksi model", mode="lines", line=dict(color=TEAL, width=1.8)))
                if v_s < pd.Timestamp("2026-01-01") <= v_e:
                    fig.add_vrect(x0="2026-01-01", x1=v_e, fillcolor=ORANGE, opacity=0.06, line_width=0)
                    fig.add_annotation(x="2026-01-01", y=1.0, yref="paper", text="holdout murni →", showarrow=False, xanchor="left",
                                       yanchor="bottom", font=dict(size=10, color=ORANGE))
                style_fig(fig, 240)
                st.plotly_chart(fig, width="stretch", config=CFG)
                st.markdown(f'<div class="note">Titik = tanggal asal prediksi. Rentang evaluasi {tgl(AV_S)} – {tgl(AV_E)}; '
                            'mulai 1 Jan 2026 holdout murni, sebelumnya walk-forward.</div>', unsafe_allow_html=True)

    # ---- Feature importance | Siaga
    a_col, b_col = st.columns([1, 1.15], gap="small")
    with a_col:
        with st.container(key="card_fi"):
            with st.container(key="head_fi"):
                st.markdown('<div class="ct">Fitur terpenting, top 10 <span>kontribusi gain (%)</span></div>', unsafe_allow_html=True)
            LABEL = {
                "days_since_hotspot": "Hari sejak hotspot terakhir", "prov_sum_7d": "Hotspot provinsi, 7 hari",
                "prov_sum_30d": "Hotspot provinsi, 30 hari", "prov_sum_90d": "Hotspot provinsi, 90 hari",
                "kal_sum_30d": "Hotspot Kalimantan, 30 hari", "kal_sum_90d": "Hotspot Kalimantan, 90 hari",
                "hotspot_sum_3d": "Hotspot kecamatan, 3 hari", "hotspot_sum_7d": "Hotspot kecamatan, 7 hari",
                "hotspot_sum_14d": "Hotspot kecamatan, 14 hari", "hotspot_sum_28d": "Hotspot kecamatan, 28 hari",
                "hotspot_max_14d": "Hotspot harian maks., 14 hari", "hotspot_hist_rate_7d": "Laju historis (7 hari)",
                "day_cos": "Musim (kosinus hari)", "day_sin": "Musim (sinus hari)", "day_of_year": "Hari dalam tahun",
                "rain_mm_lag_0d": "Curah hujan hari ini", "centroid_lat": "Lintang kecamatan", "centroid_lon": "Bujur kecamatan"}

            def grup(n):
                if n in ("day_cos", "day_sin", "day_of_year") or n.startswith(("hot_season", "month")):
                    return "Kalender"
                if n.startswith("centroid"):
                    return "Lokasi"
                if n.startswith(("prov_", "kal_", "neighbor", "share_of")):
                    return "Agregat spasial"
                if n.startswith(("hotspot", "days_since", "active_days")):
                    return "Riwayat hotspot"
                return "Cuaca/iklim"
            G_COL = {"Riwayat hotspot": NAVY, "Agregat spasial": BLUE, "Cuaca/iklim": TEAL, "Kalender": ORANGE, "Lokasi": "#9db4cc"}
            fi = load_fitur().sort_values("gain_persen", ascending=False).head(10).copy()
            fi["Fitur"] = fi["fitur"].map(lambda n: LABEL.get(n, n))
            fi["Kelompok"] = fi["fitur"].map(grup)
            fig = px.bar(fi.iloc[::-1], x="gain_persen", y="Fitur", color="Kelompok", orientation="h", color_discrete_map=G_COL,
                         hover_data={"fitur": True, "gain_persen": ":.2f", "Fitur": False, "Kelompok": False},
                         labels={"gain_persen": "Gain (%)", "fitur": "Nama teknis"}, text="gain_persen")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_yaxes(categoryorder="array", categoryarray=list(fi["Fitur"].iloc[::-1]), title=None)
            fig.update_xaxes(title=None, showgrid=True, gridcolor=GRID)
            style_fig(fig, 372, legend=True)
            fig.update_layout(hovermode="closest", legend=dict(orientation="h", y=-0.13, yanchor="top", x=0, title=None, font=dict(size=10)),
                              margin=dict(l=8, r=30, t=8, b=78), bargap=0.25)
            st.plotly_chart(fig, width="stretch", config=CFG)
    with b_col:
        with st.container(key="card_siaga"):
            w_s, w_e = pd.Timestamp(sg["window_start"].iloc[0]), pd.Timestamp(sg["window_end"].iloc[0])
            thr = float(sg["ambang_siaga"].iloc[0])
            with st.container(key="head_siaga"):
                st.markdown(f'<div class="ct">Peringkat kecamatan Siaga <span>{tgl(w_s)} – {tgl(w_e)}</span></div>', unsafe_allow_html=True)
            sg_show = sg_sel[sg_sel["status"] == "Siaga"].sort_values("peringkat")
            st.markdown(f'<div class="ins">Ambang Siaga ≥ <b>{fmt(thr, 2)}</b> hotspot/7 hari · <b>{len(sg_show)}</b> dari {len(sg_sel)} '
                        'kecamatan di wilayah terpilih</div>', unsafe_allow_html=True)
            if sg_show.empty:
                st.markdown('<div class="note" style="padding-top:10px">Tidak ada kecamatan berstatus Siaga di wilayah ini.</div>', unsafe_allow_html=True)
            else:
                vmax = float(sg["pred_hotspot_7d"].max())
                rows = "".join(
                    f"<tr><td>{int(r.peringkat)}</td><td>{esc(r.kecamatan)}</td>"
                    f"<td><span class='bar' style='width:{max(2, int(r.pred_hotspot_7d / vmax * 34))}px'></span>{fmt(r.pred_hotspot_7d, 1)}</td>"
                    f"<td>{fmt(r.aktual_7h_terakhir, 0)}</td></tr>" for r in sg_show.itertuples())
                st.markdown(
                    '<div class="tbw"><table class="ev"><thead><tr><th>#</th><th>Kecamatan</th>'
                    f'<th>Prediksi 7h</th><th>Aktual 7h</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

# ======================= KANAN =======================
with R:
    st.markdown('<div class="sect">Sebaran spasial dan iklim <span>peta kecamatan, ERA5</span></div>', unsafe_allow_html=True)

    # ---- Peta
    with st.container(key="card_map"):
        with st.container(key="head_map"):
            m1_, m2_ = st.columns([1.5, 1.5], vertical_alignment="center")
            m1_.markdown('<div class="ct">Peta sebaran hotspot <span>total pada periode</span></div>', unsafe_allow_html=True)
            bm = m2_.segmented_control("Peta dasar", ["Polos", "OpenStreetMap", "Carto"], default="Polos", key="basemap",
                                       label_visibility="collapsed") or "Polos"
        BASEMAP = {"Polos": "white-bg", "OpenStreetMap": "open-street-map", "Carto": "carto-positron"}[bm]

        ids_list = [i for i in meta["district_id"] if i in ids_data]
        tot_vals = np.array([float(stats["total"].get(i, 0.0)) for i in ids_list])
        zmax = float(np.log1p(np.quantile(tot_vals, 0.98))) if tot_vals.max() > 0 else 1.0
        zmax = max(zmax, 1e-6)
        hover = []
        for i, tv in zip(ids_list, tot_vals):
            r = info.loc[i]
            hover.append(f"<b>{esc(r['kecamatan'])}</b><br>{esc(r['kabupaten'])}, {esc(r['provinsi'])}<br>"
                         f"Total hotspot: <b>{fmt(tv, dec)}</b><br>Rata-rata/hari: <b>{fmt(tv / n_days_p, 2)}</b><br>"
                         f"Perubahan awal→akhir: {change_text(*f12(i))}")
        no_data = [i for i in meta["district_id"] if i not in ids_data]
        gj = load_geojson()
        tick_cand = [t for t in [0, 1, 10, 100, 1000, 10000] if np.log1p(t) <= zmax * 1.001]
        fig = go.Figure()
        if no_data:
            fig.add_trace(go.Choroplethmap(
                geojson=gj, locations=no_data, featureidkey="properties.district_id", z=[0] * len(no_data),
                colorscale=[[0, "#bdbdbd"], [1, "#bdbdbd"]], showscale=False, marker=dict(opacity=0.8, line=dict(width=0.3, color="#777")),
                text=[f"{info.loc[i]['kecamatan']}: tanpa data" for i in no_data], hoverinfo="text"))
        fig.add_trace(go.Choroplethmap(
            geojson=gj, locations=ids_list, featureidkey="properties.district_id", z=np.clip(np.log1p(tot_vals), 0, zmax),
            zmin=0, zmax=zmax, colorscale=[[0, "#2e9e4f"], [0.5, "#f4d03f"], [1, "#d62728"]],
            marker=dict(opacity=0.88, line=dict(width=0.3, color="#555")), hovertext=hover, hoverinfo="text",
            hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK, size=11), align="left"),
            colorbar=dict(title=dict(text="hotspot", font=dict(size=10)), thickness=10, len=0.7, x=1.0, tickfont=dict(size=10),
                          tickvals=[float(np.log1p(t)) for t in tick_cand], ticktext=[fmt(t) for t in tick_cand])))
        if not is_all_region:
            fig.add_trace(go.Choroplethmap(
                geojson=gj, locations=sel_ids, featureidkey="properties.district_id", z=[0] * len(sel_ids),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
                marker=dict(line=dict(width=2.6, color="#000")), hoverinfo="skip"))

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

        w_, e_, s_, n_ = bbox(sel_ids if not is_all_region else meta["district_id"])
        px_, py_ = (e_ - w_) * 0.05 + 0.02, (n_ - s_) * 0.05 + 0.02
        fig.update_layout(map=dict(style=BASEMAP, bounds=dict(west=w_ - px_, east=e_ + px_, south=s_ - py_, north=n_ + py_)),
                          height=400, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select", paper_bgcolor="rgba(0,0,0,0)")
        ev = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="peta", width="stretch",
                             config=dict(displaylogo=False, displayModeBar=False, scrollZoom=True))

        picked_id = None
        try:
            pts = ev.selection.points
            if pts:
                picked_id = pts[0].get("location")
        except Exception:
            picked_id = None

        if picked_id in info.index and picked_id in ids_data:
            r = info.loc[picked_id]
            tv = float(stats["total"].get(picked_id, 0.0))
            cells = [("Kecamatan", esc(r["kecamatan"]), f"{esc(r['kabupaten'])}, {esc(PROV_SHORT.get(r['provinsi'], r['provinsi']))}"),
                     ("Total hotspot", fmt(tv, dec), f"{tgl(start)} – {tgl(end)}"),
                     ("Rata-rata per hari", fmt(tv / n_days_p, 2), "total ÷ jumlah hari"),
                     ("Perubahan", change_text(*f12(picked_id)), "paruh akhir vs paruh awal")]
        else:
            sub = stats.reindex(sel_ids).fillna(0.0)
            tot_r = float(sub["total"].sum())
            top_id = sub["total"].idxmax() if len(sub) and sub["total"].max() > 0 else None
            cells = [("Wilayah", esc(region_name), f"{len(sel_ids)} kecamatan"),
                     ("Total hotspot", fmt(tot_r, dec), f"{fmt(tot_r / n_days_p, 1)} per hari"),
                     ("Kecamatan terpanas", esc(info.loc[top_id]["kecamatan"]) if top_id else "–",
                      f"{fmt(sub.loc[top_id, 'total'], dec)} hotspot" if top_id else "tidak ada hotspot"),
                     ("Perubahan", change_text(float(sub["first"].sum()), float(sub["second"].sum())), "klik kecamatan untuk rincian")]
        st.markdown('<div class="sumw">' + "".join(
            f'<div class="sum"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for l, v, s in cells) + "</div>",
            unsafe_allow_html=True)

    # ---- ERA5
    with st.container(key="card_era"):
        with st.container(key="head_era"):
            e1, e2, e3 = st.columns([1.35, 0.85, 0.9], vertical_alignment="center")
            e1.markdown('<div class="ct">Variabel iklim ERA5 <span>rata-rata wilayah</span></div>', unsafe_allow_html=True)
            mode_era = e2.segmented_control("Skala", ["Z-score", "Asli"], default="Z-score", key="mode_era",
                                            label_visibility="collapsed") or "Z-score"
            with e3.popover("Variabel", width="stretch"):
                sel_vars = st.multiselect("Variabel", list(ERA5_VARS), default=list(ERA5_VARS), format_func=lambda k: ERA5_VARS[k][0])
                agg_era = st.selectbox("Agregasi", ["Otomatis", "Harian", "Mingguan", "Bulanan"], key="agg_era")
        e_start, e_end = max(start, pd.Timestamp("2000-11-01")), min(end, ERA5_END)
        if e_start > e_end:
            st.markdown('<div class="note" style="padding-top:10px">Data ERA5 tersedia sampai 23 Sep 2026. Pilih periode yang lebih awal.</div>',
                        unsafe_allow_html=True)
        elif not sel_vars:
            st.markdown('<div class="note" style="padding-top:10px">Pilih minimal satu variabel.</div>', unsafe_allow_html=True)
        else:
            lvl = "Kalimantan" if is_all_region else (sel_prov if sel_kab == ALL_KAB else None)
            e = load_era5(sel_ids_t, str(e_start.date()), str(e_end.date()), lvl)
            if e.empty:
                st.markdown('<div class="note" style="padding-top:10px">Tidak ada data ERA5 untuk wilayah ini.</div>', unsafe_allow_html=True)
            else:
                nd = (e_end - e_start).days + 1
                rule = {"Harian": None, "Mingguan": "W", "Bulanan": "MS"}.get(agg_era)
                if agg_era == "Otomatis":
                    rule = None if nd <= 200 else ("W" if nd <= 1100 else "MS")
                if rule:
                    e = e.resample(rule).mean()
                z = (e[sel_vars] - e[sel_vars].mean()) / e[sel_vars].std(ddof=0).replace(0, np.nan)
                if mode_era == "Z-score":
                    fig = go.Figure()
                    for v in sel_vars:
                        fig.add_trace(go.Scatter(x=e.index, y=z[v], name=ERA5_VARS[v][0].split(" (")[0], mode="lines",
                                                 line=dict(color=ERA5_VARS[v][1], width=1.6), customdata=e[v],
                                                 hovertemplate="%{customdata:.2f}<extra>" + ERA5_VARS[v][0] + "</extra>"))
                    fig.update_yaxes(title=dict(text="z-score", font=dict(size=10)))
                    style_fig(fig, 300)
                else:
                    fig = make_subplots(rows=len(sel_vars), cols=1, shared_xaxes=True, vertical_spacing=0.04)
                    for i, v in enumerate(sel_vars, 1):
                        fig.add_trace(go.Scatter(x=e.index, y=e[v], name=ERA5_VARS[v][0], mode="lines",
                                                 line=dict(color=ERA5_VARS[v][1], width=1.5)), row=i, col=1)
                        fig.update_yaxes(title=dict(text=ERA5_VARS[v][0].split(" (")[0], font=dict(size=9)), row=i, col=1)
                    style_fig(fig, max(300, 90 * len(sel_vars)), legend=False)
                st.plotly_chart(fig, width="stretch", config=CFG)
                st.markdown('<div class="note">Rata-rata seluruh kecamatan terpilih. Angin = √(u10² + v10²). Pada mode z-score, '
                            'nilai asli muncul di tooltip.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------------
bias26 = float(mt[(mt["split"] == "holdout") & (mt["model"] == "Model")].iloc[0]["bias_pct"])
st.markdown(
    f'<div class="foot"><div>Info data <span class="pill"><span class="dot"></span>Hotspot aktual terakhir: {tgl(LAST_OBS)} · '
    f'Forecast s.d. {tgl(LAST_FC)}</span></div>'
    f'<div>MODIS FIRMS · ERA5 · batas BPS ADM3 · <span class="w">Model cenderung meremehkan puncak tahun api besar '
    f'(bias holdout 2026 ≈ {fmt(bias26, 0)}%).</span></div></div>', unsafe_allow_html=True)
