"""
app.py — Dashboard Sinyal Saham
Jalankan:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pathlib import Path

from sinyal import ambil_banyak, analisa, baca_daftar_syariah

st.set_page_config(page_title="Dashboard Sinyal Saham Syariah", page_icon="📈", layout="wide")
st.title("📈 Dashboard Sinyal Saham Syariah")

import datetime as dt

# Folder berisi daftar per periode. Nama file = tanggal efektif, mis. 2026-12-01.txt
FOLDER = Path(__file__).parent / "daftar_syariah"
FOLDER.mkdir(exist_ok=True)
LAMA = Path(__file__).parent / "syariah.txt"                 # pindahkan file versi lama
if LAMA.exists() and not any(FOLDER.glob("*.txt")):
    (FOLDER / "2026-06-02.txt").write_text(LAMA.read_text())

HARI_INI = dt.date.today()


def periode_berikut(tgl: dt.date) -> dt.date:
    """Tanggal efektif berikutnya: 1 Juni atau 1 Desember."""
    if tgl < dt.date(tgl.year, 6, 1):
        return dt.date(tgl.year, 6, 1)
    if tgl < dt.date(tgl.year, 12, 1):
        return dt.date(tgl.year, 12, 1)
    return dt.date(tgl.year + 1, 6, 1)


def daftar_aktif():
    """Pilih file dengan tanggal efektif terbaru yang sudah berlaku hari ini."""
    berkas = []
    for f in FOLDER.glob("*.txt"):
        try:
            berkas.append((dt.date.fromisoformat(f.stem), f))
        except ValueError:
            pass
    berlaku = sorted([b for b in berkas if b[0] <= HARI_INI])
    menunggu = sorted([b for b in berkas if b[0] > HARI_INI])
    return (berlaku[-1] if berlaku else None), menunggu


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Daftar Syariah")
    with st.expander("Unggah daftar periode baru"):
        tgl_efektif = st.date_input("Tanggal mulai berlaku",
                                    periode_berikut(HARI_INI))
        unggah = st.file_uploader("File Excel/CSV/TXT dari IDX atau OJK",
                                  type=["xlsx", "xls", "csv", "txt"])
        if unggah:
            baru = baca_daftar_syariah(unggah)
            if baru:
                (FOLDER / f"{tgl_efektif.isoformat()}.txt").write_text("\n".join(baru))
                st.success(f"{len(baru)} saham disimpan, berlaku mulai "
                           f"{tgl_efektif:%d-%m-%Y}")

    aktif, menunggu = daftar_aktif()
    if not aktif and menunggu:                       # cadangan: pakai daftar terdekat
        aktif = menunggu.pop(0)
        st.warning("Belum ada daftar yang berlaku hari ini, sementara memakai "
                   f"daftar {aktif[0]:%d-%m-%Y}.")
    if aktif:
        tgl_aktif, file_aktif = aktif
        SYARIAH = baca_daftar_syariah(file_aktif)
        batas = periode_berikut(tgl_aktif)
        st.caption(f"{len(SYARIAH)} saham • berlaku sejak {tgl_aktif:%d-%m-%Y}")
        if HARI_INI >= batas:
            st.error(f"Daftar sudah kedaluwarsa sejak {batas:%d-%m-%Y}. "
                     "Unggah daftar periode terbaru.")
        elif (batas - HARI_INI).days <= 14 and not menunggu:
            st.warning(f"Periode baru mulai {batas:%d-%m-%Y}. "
                       "Siapkan daftar barunya.")
    else:
        SYARIAH = []
        st.error("Belum ada daftar yang berlaku.")
    for tgl, _ in menunggu:
        st.info(f"Daftar baru siap, otomatis aktif {tgl:%d-%m-%Y}")

    st.header("Pengaturan")
    mode = st.radio("Saham yang ditampilkan",
                    ["Semua saham syariah", "Pilih sendiri"])
    if mode == "Pilih sendiri":
        daftar = st.text_area("Kode saham (pisah koma)", "ANTM, BUMI, BRIS, TLKM")
        diketik = [k.strip().upper() for k in daftar.split(",") if k.strip()]
        ditolak = [k for k in diketik if k not in SYARIAH]
        if ditolak:
            st.error("Bukan syariah / tidak ada di daftar: " + ", ".join(ditolak))
        kode_list = [k for k in diketik if k in SYARIAH]
    else:
        kode_list = SYARIAH
    hanya_beli = st.checkbox("Tampilkan hanya sinyal BELI", value=False)
    periode = st.selectbox("Periode data", ["6mo", "1y", "2y"], index=1)
    atr_sl = st.slider("Jarak Stop Loss (x ATR)", 1.0, 3.0, 1.5, 0.1)
    rr1 = st.slider("Take Profit 1 (x risiko)", 1.0, 3.0, 1.5, 0.5)
    rr2 = st.slider("Take Profit 2 (x risiko)", 2.0, 5.0, 3.0, 0.5)
    pakai_ai = st.checkbox("Gabung model AI (SahamAI)", value=False)
    st.caption("Bukan ajakan jual/beli. Selalu cek ulang sebelum transaksi.")

@st.cache_data(ttl=600)          # data disimpan 10 menit agar cepat
def muat_semua(kode_tuple, periode):
    return ambil_banyak(list(kode_tuple), periode)


def skor_model_ai(df):
    """
    >>> TEMPAT MENEMPEL MODEL DARI SahamAI.ipynb <<<
    Kembalikan angka -1 (sangat bearish) .. +1 (sangat bullish), atau None.
    Contoh jika di notebook Anda menyimpan model dengan joblib:
        import joblib
        model = joblib.load("model_saham.pkl")
        fitur = df[["EMA9", "EMA21", "RSI", "VolRasio"]].tail(1)
        peluang_naik = model.predict_proba(fitur)[0][1]
        return (peluang_naik - 0.5) * 2
    """
    return None


# ---------- Tabel ringkasan ----------
hasil = {}
baris = []
if not kode_list:
    st.info("Belum ada saham syariah untuk ditampilkan.")
    st.stop()

with st.spinner(f"Mengambil data {len(kode_list)} saham..."):
    semua = muat_semua(tuple(kode_list), periode)
    for kode in kode_list:
        try:
            if kode not in semua:
                st.warning(f"{kode}: data tidak tersedia")
                continue
            df = semua[kode]
            ai = None
            if pakai_ai:
                from sinyal import hitung_indikator
                ai = skor_model_ai(hitung_indikator(df))
            r = analisa(df, atr_sl, rr1, rr2, skor_ai=ai)
            if r["sinyal"] == "DATA KURANG":
                continue
            hasil[kode] = r
            baris.append({
                "Kode": kode, "Harga": int(r["harga"]), "Sinyal": r["sinyal"],
                "Skor": r["skor"], "RSI": r["RSI"], "Vol x": r["VolRasio"],
                "Stop Loss": int(r["stop_loss"]), "TP1": int(r["tp1"]),
                "TP2": int(r["tp2"]), "Risiko %": r["risiko_%"],
            })
        except Exception as e:
            st.warning(f"{kode}: gagal ambil data ({e})")

if not baris:
    st.stop()

tabel = pd.DataFrame(baris).sort_values("Skor", ascending=False)
if hanya_beli:
    tabel = tabel[tabel["Sinyal"] == "BELI"]
    if tabel.empty:
        st.info("Tidak ada sinyal BELI hari ini.")
        st.stop()


def warna(v):
    return {"BELI": "background-color:#1b5e20;color:white",
            "PANTAU": "background-color:#f9a825;color:black",
            "JUAL": "background-color:#b71c1c;color:white"}.get(v, "")


def warna_vol(v):
    return "color:#1b5e20;font-weight:bold" if v >= 1.5 else (
        "color:#9e9e9e" if v < 1 else "")


st.subheader("Ringkasan Sinyal")
tgl_data = max(r["df"].index[-1] for r in hasil.values()).date()
hari_bursa_terakhir = HARI_INI - dt.timedelta(days=1)       # penutupan kemarin
while hari_bursa_terakhir.weekday() >= 5:                       # lewati Sabtu/Minggu
    hari_bursa_terakhir -= dt.timedelta(days=1)
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
st.caption(f"Data harga per penutupan {HARI[tgl_data.weekday()]}, {tgl_data:%d-%m-%Y} • "
           "diperbarui otomatis tiap 10 menit")
if tgl_data < hari_bursa_terakhir:
    st.warning("Data belum terbaru (mungkin hari libur bursa atau Yahoo terlambat).")
gaya = (tabel.style.map(warna, subset=["Sinyal"])
        .map(warna_vol, subset=["Vol x"])
        .format({"Harga": "{:,.0f}", "Stop Loss": "{:,.0f}", "TP1": "{:,.0f}",
                 "TP2": "{:,.0f}", "RSI": "{:.1f}", "Vol x": "{:.2f}x",
                 "Risiko %": "{:.1f}%"}))
st.dataframe(gaya, hide_index=True, use_container_width=True)
st.caption("BELI = skor ≥3 + volume ≥ rata-rata • PANTAU = teknikal bagus tapi "
           "volume masih sepi • Risiko % = jarak entry ke Stop Loss")

# ---------- Detail per saham ----------
st.subheader("Detail")
pilih = st.selectbox("Pilih saham", tabel["Kode"].tolist())
r = hasil[pilih]
df = r["df"].tail(120)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sinyal", r["sinyal"], f"skor {r['skor']:+d}")
c2.metric("Entry", f"{r['entry']:,.0f}")
c3.metric("Stop Loss", f"{r['stop_loss']:,.0f}", f"-{r['risiko_%']}%")
c4.metric("TP1 / TP2", f"{r['tp1']:,.0f} / {r['tp2']:,.0f}")

st.markdown("**Alasan:**\n" + "\n".join(f"- {x}" for x in r["alasan"]))

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                             low=df["Low"], close=df["Close"], name="Harga"), 1, 1)
for ema, w in [("EMA9", "#2196f3"), ("EMA21", "#ff9800"), ("EMA50", "#9c27b0")]:
    fig.add_trace(go.Scatter(x=df.index, y=df[ema], name=ema,
                             line=dict(color=w, width=1.3)), 1, 1)
for lvl, nama, w in [(r["stop_loss"], "SL", "red"), (r["tp1"], "TP1", "green"),
                     (r["tp2"], "TP2", "darkgreen")]:
    fig.add_hline(y=lvl, line_dash="dash", line_color=w,
                  annotation_text=f"{nama} {lvl:,.0f}", row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                     marker_color="#90a4ae"), 2, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["VolMA20"], name="Vol MA20",
                         line=dict(color="black", width=1)), 2, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                         line=dict(color="#673ab7")), 3, 1)
fig.add_hline(y=70, line_dash="dot", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", row=3, col=1)
fig.update_layout(height=750, xaxis_rangeslider_visible=False,
                  margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)
