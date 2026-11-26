"""
Carga de datos intradía y construcción de barras de 30 minutos de la sesión regular (RTH).

Fuentes soportadas:
  - Yahoo Finance (yfinance): MES=F. Límites de Yahoo: 30m -> ~60 días, 60m -> ~730 días.
  - CSV propio (recomendado para un backtest serio): 1, 5, 15 o 30 minutos,
    exportado de NinjaTrader, TradingView, Databento, FirstRate Data, etc.

Salida común: DataFrame con una fila por barra de 30 min de la sesión regular
(9:30-16:00 hora de Nueva York), con columnas:
  fecha, k (0..12), hora_cierre, open, high, low, close, volume, vwap
donde k=0 es la barra 9:30-10:00 (se evalúa a las 10:00) y k=12 la barra 15:30-16:00.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd

NY = "America/New_York"
APERTURA_MIN = 9 * 60 + 30   # 9:30
CIERRE_MIN = 16 * 60         # 16:00


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #
def cargar_yahoo(ticker: str = "MES=F", intervalo: str = "30m", periodo: str = "60d") -> pd.DataFrame:
    """Descarga barras intradía de Yahoo Finance y las devuelve en hora de Nueva York."""
    import yfinance as yf

    df = yf.download(ticker, interval=intervalo, period=periodo, auto_adjust=False,
                     progress=False, prepost=True)
    if df is None or df.empty:
        raise ValueError(f"Yahoo no devolvió datos para {ticker} ({intervalo}, {periodo}).")
    if isinstance(df.columns, pd.MultiIndex):          # yfinance >= 0.2 trae MultiIndex
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(NY)
    return df.dropna(subset=["open", "high", "low", "close"])


def cargar_csv(archivo, zona_horaria: str = NY) -> pd.DataFrame:
    """
    Lee un CSV con columnas de fecha/hora y OHLCV. Acepta nombres comunes:
    datetime / timestamp / time / date (+ time), open, high, low, close, volume.
    `zona_horaria` es la zona en la que vienen los timestamps si no traen offset.
    """
    if isinstance(archivo, (bytes, bytearray)):
        archivo = io.BytesIO(archivo)
    df = pd.read_csv(archivo, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    alias = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vol": "volume",
             "last": "close", "price": "close"}
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})

    col_t = next((c for c in ("datetime", "timestamp", "ts_event", "date_time", "time") if c in df.columns), None)
    if col_t is None and "date" in df.columns:
        if "time" in df.columns:
            ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        else:
            ts = pd.to_datetime(df["date"])
    elif col_t is not None:
        ts = pd.to_datetime(df[col_t], utc=False)
    else:
        raise ValueError("No encontré una columna de fecha/hora (datetime, timestamp, date+time).")

    ts = pd.DatetimeIndex(ts)
    ts = ts.tz_localize(zona_horaria, ambiguous="infer", nonexistent="shift_forward") if ts.tz is None else ts
    df.index = ts.tz_convert(NY)

    faltan = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    return df[~df.index.duplicated(keep="last")]


# --------------------------------------------------------------------------- #
# Barras de 30 minutos de la sesión regular
# --------------------------------------------------------------------------- #
def granularidad_minutos(df: pd.DataFrame) -> float:
    d = pd.Series(df.index).diff().dropna()
    return float(d.dt.total_seconds().median() / 60) if len(d) else float("nan")


def barras_rth_30m(df: pd.DataFrame, min_barras_dia: int = 13) -> pd.DataFrame:
    """
    Agrupa barras de <=30 min en cubetas de 30 min alineadas a las 9:30 NY.
    El VWAP se calcula con las barras originales (más fino) y se toma al cierre de cada cubeta.
    Se descartan días incompletos (feriados de medio día, huecos de datos).
    """
    g = granularidad_minutos(df)
    if g > 30.5:
        raise ValueError(
            f"Los datos vienen en barras de ~{g:.0f} min. La estrategia decide cada 30 min, "
            "así que necesita barras de 30 min o menores (1m, 5m, 15m, 30m)."
        )

    x = df.copy()
    minuto = x.index.hour * 60 + x.index.minute
    x = x[(minuto >= APERTURA_MIN) & (minuto < CIERRE_MIN)]
    if x.empty:
        raise ValueError("No hay barras dentro de la sesión regular 9:30-16:00 NY.")

    minuto = x.index.hour * 60 + x.index.minute
    x["fecha"] = x.index.date
    x["k"] = ((minuto - APERTURA_MIN) // 30).astype(int)

    tp = (x["high"] + x["low"] + x["close"]) / 3.0
    x["pv"] = tp * x["volume"]
    x["tp"] = tp

    agg = x.groupby(["fecha", "k"]).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), pv=("pv", "sum"),
        tp_sum=("tp", "sum"), n=("tp", "size"),
    ).reset_index()

    # VWAP acumulado del día (si no hay volumen, promedio acumulado del precio típico)
    agg["cum_pv"] = agg.groupby("fecha")["pv"].cumsum()
    agg["cum_v"] = agg.groupby("fecha")["volume"].cumsum()
    agg["cum_tp"] = agg.groupby("fecha")["tp_sum"].cumsum()
    agg["cum_n"] = agg.groupby("fecha")["n"].cumsum()
    vwap_vol = agg["cum_pv"] / agg["cum_v"].replace(0, np.nan)
    agg["vwap"] = vwap_vol.fillna(agg["cum_tp"] / agg["cum_n"])

    completos = agg.groupby("fecha")["k"].nunique()
    completos = completos[completos >= min_barras_dia].index
    agg = agg[agg["fecha"].isin(completos)].copy()

    agg["hora_cierre"] = agg["k"].map(lambda k: f"{(APERTURA_MIN + 30 * (k + 1)) // 60:02d}:"
                                                f"{(APERTURA_MIN + 30 * (k + 1)) % 60:02d}")
    cols = ["fecha", "k", "hora_cierre", "open", "high", "low", "close", "volume", "vwap"]
    return agg[cols].sort_values(["fecha", "k"]).reset_index(drop=True)
