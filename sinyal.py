"""
sinyal.py — mesin sinyal saham (EMA + RSI + Volume + ATR)
Bisa dipakai di dashboard (app.py) maupun di notebook SahamAI.ipynb:
    from sinyal import ambil_data, analisa
"""
import numpy as np
import pandas as pd


# ---------- 1. Ambil data ----------
def ambil_data(kode: str, periode: str = "1y") -> pd.DataFrame:
    """Ambil data harian dari Yahoo Finance. Saham IDX pakai akhiran .JK"""
    import yfinance as yf
    ticker = kode.upper().strip()
    if not ticker.endswith(".JK") and "." not in ticker:
        ticker += ".JK"
    df = yf.download(ticker, period=periode, interval="1d",
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):          # yfinance versi baru
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def ambil_banyak(kode_list: list, periode: str = "1y") -> dict:
    """Ambil banyak saham sekaligus (jauh lebih cepat untuk ratusan kode)."""
    import yfinance as yf
    peta = {k: (k if "." in k else k + ".JK") for k in kode_list}
    data = yf.download(list(peta.values()), period=periode, interval="1d",
                       group_by="ticker", auto_adjust=False,
                       threads=True, progress=False)
    hasil = {}
    for kode, t in peta.items():
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna()
            if len(df):
                hasil[kode] = df
        except KeyError:
            pass
    return hasil


# ---------- Daftar saham syariah ----------
def baca_daftar_syariah(sumber) -> list:
    """
    Baca daftar kode syariah dari:
    - file .txt (satu kode per baris, baris diawali # diabaikan), atau
    - file Excel/CSV unduhan IDX/OJK (kolom yang namanya mengandung 'Kode').
    """
    nama = getattr(sumber, "name", str(sumber)).lower()
    if nama.endswith(".txt"):
        teks = sumber.read().decode() if hasattr(sumber, "read") else open(sumber).read()
        kode = [b.strip().upper() for b in teks.splitlines()
                if b.strip() and not b.strip().startswith("#")]
    else:
        df = pd.read_csv(sumber) if nama.endswith(".csv") else pd.read_excel(sumber)
        kol = next((c for c in df.columns if "kode" in str(c).lower()), df.columns[0])
        kode = df[kol].astype(str).str.strip().str.upper().tolist()
    kode = [k.split()[0] for k in kode if k and k != "NAN"]
    return [k for k in dict.fromkeys(kode) if 2 <= len(k) <= 5 and k.isalpha()]


# ---------- 2. Indikator ----------
def hitung_indikator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()

    # RSI 14 (Wilder)
    delta = c.diff()
    naik = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    turun = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = naik / turun.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(100)

    # ATR 14 — dipakai untuk jarak stop loss
    prev = c.shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["VolRasio"] = df["Volume"] / df["VolMA20"]
    return df


# ---------- 3. Fraksi harga BEI ----------
def fraksi(harga: float) -> int:
    if harga < 200:
        return 1
    if harga < 500:
        return 2
    if harga < 2000:
        return 5
    if harga < 5000:
        return 10
    return 25


def bulatkan(harga: float, ke_bawah: bool = True) -> float:
    f = fraksi(harga)
    return (np.floor if ke_bawah else np.ceil)(harga / f) * f


# ---------- 4. Sinyal + SL/TP ----------
def analisa(df: pd.DataFrame, atr_sl: float = 1.5, rr1: float = 1.5,
            rr2: float = 3.0, skor_ai: float | None = None) -> dict:
    """
    Skor -5..+5 dari 5 aturan. skor_ai (opsional, -1..+1) = hasil model dari SahamAI.ipynb
    """
    df = hitung_indikator(df)
    if len(df) < 60:
        return {"sinyal": "DATA KURANG"}
    a, b = df.iloc[-1], df.iloc[-2]
    harga = float(a["Close"])

    skor, alasan = 0, []
    # 1. Tren jangka pendek
    if a["EMA9"] > a["EMA21"]:
        skor += 1; alasan.append("EMA9 > EMA21 (tren naik)")
    else:
        skor -= 1; alasan.append("EMA9 < EMA21 (tren turun)")
    # 2. Tren menengah
    if harga > a["EMA50"]:
        skor += 1; alasan.append("Harga di atas EMA50")
    else:
        skor -= 1; alasan.append("Harga di bawah EMA50")
    # 3. Momentum RSI
    if 50 <= a["RSI"] <= 70:
        skor += 1; alasan.append(f"RSI {a['RSI']:.0f} momentum sehat")
    elif a["RSI"] > 70:
        skor -= 1; alasan.append(f"RSI {a['RSI']:.0f} overbought")
    elif a["RSI"] < 30:
        skor += 1; alasan.append(f"RSI {a['RSI']:.0f} oversold (potensi pantul)")
    else:
        skor -= 1; alasan.append(f"RSI {a['RSI']:.0f} lemah")
    # 4. Volume konfirmasi
    naik_hari_ini = harga > b["Close"]
    if a["VolRasio"] >= 1.5:
        skor += 1 if naik_hari_ini else -1
        alasan.append(f"Volume {a['VolRasio']:.1f}x rata-rata "
                      f"({'akumulasi' if naik_hari_ini else 'distribusi'})")
    # 5. Golden / death cross baru terjadi
    if b["EMA9"] <= b["EMA21"] and a["EMA9"] > a["EMA21"]:
        skor += 1; alasan.append("Golden cross EMA9/21 hari ini")
    if b["EMA9"] >= b["EMA21"] and a["EMA9"] < a["EMA21"]:
        skor -= 1; alasan.append("Death cross EMA9/21 hari ini")
    # 6. Opsional: model AI dari notebook
    if skor_ai is not None:
        skor += round(2 * skor_ai)
        alasan.append(f"Model AI: {skor_ai:+.2f}")

    if skor >= 3 and a["VolRasio"] >= 1.0:
        sinyal = "BELI"
    elif skor >= 3:
        sinyal = "PANTAU"          # teknikal bagus, tapi volume belum mendukung
        alasan.append(f"Volume hanya {a['VolRasio']:.2f}x rata-rata → tunggu konfirmasi")
    elif skor <= -2:
        sinyal = "JUAL"
    else:
        sinyal = "TAHAN / WAIT"

    # Level harga (dihitung untuk posisi beli)
    atr = float(a["ATR"])
    sl = bulatkan(min(harga - atr_sl * atr, float(df["Low"].tail(5).min()) - fraksi(harga)))
    risiko = harga - sl
    tp1 = bulatkan(harga + rr1 * risiko, ke_bawah=False)
    tp2 = bulatkan(harga + rr2 * risiko, ke_bawah=False)

    return {
        "harga": harga,
        "sinyal": sinyal,
        "skor": skor,
        "RSI": round(float(a["RSI"]), 1),
        "VolRasio": round(float(a["VolRasio"]), 2),
        "entry": bulatkan(harga),
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risiko_%": round(risiko / harga * 100, 2),
        "alasan": alasan,
        "df": df,
    }
