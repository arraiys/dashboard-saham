"""
intraday.py — logika SahamAI.ipynb (screener intraday, kondisi BELUM PUNYA)
Indikator: EMA harian + EMA9/21 intraday, RSI, VWAP, Relative Volume, ATR.
"""
import math

import numpy as np
import pandas as pd

from sinyal import fraksi as tick

WIB = "Asia/Jakarta"


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def bulat(p, arah="bawah"):
    t = tick(p)
    f = math.floor if arah == "bawah" else math.ceil
    return int(f(p / t) * t)


def ambil(kode, interval="15m"):
    import yfinance as yf
    tk = yf.Ticker(kode + ".JK")
    d = tk.history(period="6mo", interval="1d", auto_adjust=False)
    i = tk.history(period="10d", interval=interval, auto_adjust=False)
    return d, i


def analisis(kode, d, i, modal=10_000_000, risiko_pct=1.0,
             interval="15m", maks_delay=30):
    if d is None or i is None or len(d) < 50 or len(i) < 30:
        return {"Kode": kode, "Status": "DATA KURANG"}
    i = i[i["Volume"] > 0].copy()
    hari = i.index.date
    hari_ini = hari[-1]
    td = i[hari == hari_ini].copy()

    # Harian (tren)
    c_d = d["Close"]
    ema20_d, ema50_d = ema(c_d, 20).iloc[-1], ema(c_d, 50).iloc[-1]
    rsi_d = rsi(c_d).iloc[-1]
    sebelum = d[d.index.date < hari_ini]["Close"]
    prev_close = sebelum.iloc[-1] if len(sebelum) else c_d.iloc[-1]

    # Intraday
    c = i["Close"]
    harga = c.iloc[-1]
    e9, e21 = ema(c, 9).iloc[-1], ema(c, 21).iloc[-1]
    rsi_i = rsi(c).iloc[-1]
    atr_i = atr(i).iloc[-1]
    tp = (td["High"] + td["Low"] + td["Close"]) / 3
    vwap = (tp * td["Volume"]).cumsum().iloc[-1] / td["Volume"].cumsum().iloc[-1]

    # Relative volume vs rata-rata jam yang sama hari-hari sebelumnya
    jam = td.index[-1].time()
    vol_ini = td["Volume"].sum()
    hist = [i[hari == h].pipe(lambda x: x[x.index.time <= jam])["Volume"].sum()
            for h in sorted(set(hari))[:-1]]
    avg_hist = np.mean(hist) if hist else 0
    rvol = vol_ini / avg_hist if avg_hist > 0 else np.nan

    # Skor
    skor, alasan = 0, []

    def cek(kond, poin, teks):
        nonlocal skor
        if kond:
            skor += poin
            alasan.append(teks)

    cek(c_d.iloc[-1] > ema20_d, 1, "di atas EMA20 harian")
    cek(ema20_d > ema50_d, 1, "EMA20>EMA50 harian")
    cek(harga > vwap, 1, "di atas VWAP")
    cek(e9 > e21, 1, f"EMA9>EMA21 {interval}")
    cek(50 <= rsi_i <= 70, 1, "RSI intraday sehat")
    cek(rvol >= 1.2, 1, f"volume {rvol:.1f}x normal")
    cek(rvol >= 2.0, 1, "volume sangat kuat")
    cek(rsi_i > 75 or rsi_d > 80, -2, "OVERBOUGHT")
    cek(harga < vwap and e9 < e21, -1, "tekanan jual intraday")

    status = "BELI" if skor >= 5 else ("PANTAU" if skor >= 3 else "HINDARI")

    # Level untuk yang belum punya
    entry_atas = bulat(harga, "bawah")
    entry_bawah = min(bulat(max(vwap, e9), "bawah"), entry_atas)
    entry = (entry_atas + entry_bawah) / 2
    stop = bulat(min(entry - 1.5 * atr_i, td["Low"].tail(4).min() - tick(harga)), "bawah")
    r = max(entry - stop, tick(harga))
    tp1 = bulat(entry + 1.5 * r, "atas")
    tp2 = bulat(entry + 2.5 * r, "atas")
    lot_risiko = int(modal * risiko_pct / 100 / (r * 100))
    lot_modal = int(modal / (entry_atas * 100))
    lot = max(0, min(lot_risiko, lot_modal))

    terakhir = i.index[-1].tz_convert(WIB)
    telat = (pd.Timestamp.now(tz=WIB) - terakhir).total_seconds() / 60

    return {
        "Kode": kode, "Status": status, "Skor": skor,
        "Harga": int(harga), "Chg%": round((harga / prev_close - 1) * 100, 2),
        "VWAP": int(vwap), "RSI_i": round(rsi_i, 1), "RSI_d": round(rsi_d, 1),
        "RVOL": None if np.isnan(rvol) else round(rvol, 2),
        "Entry": f"{entry_bawah}-{entry_atas}", "Stop": stop,
        "TP1": tp1, "TP2": tp2, "Lot": lot,
        "Data": terakhir.strftime("%d-%m %H:%M") + (" BASI" if telat > maks_delay else ""),
        "Alasan": ", ".join(alasan),
    }


def kondisi_ihsg():
    """Kembalikan (teks, peringatan) kondisi IHSG."""
    import yfinance as yf
    d = yf.Ticker("^JKSE").history(period="3mo", interval="1d")
    chg = (d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100
    tren = "di atas" if d["Close"].iloc[-1] > ema(d["Close"], 20).iloc[-1] else "di bawah"
    teks = f"IHSG {d['Close'].iloc[-1]:,.0f} ({chg:+.2f}%), {tren} EMA20 harian"
    return teks, chg < -1
