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

    def _parsear(col: pd.Series) -> pd.Series:
        """Texto con o sin zona horaria, o números Unix (s, ms, µs o ns, como TradingView o Databento)."""
        if pd.api.types.is_numeric_dtype(col):
            v = float(col.abs().median())
            unidad = "ns" if v > 1e17 else "us" if v > 1e14 else "ms" if v > 1e11 else "s"
            return pd.to_datetime(col, unit=unidad, utc=True)
        txt = col.astype(str).str.strip()
        con_zona = txt.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).mean() > 0.5
        # con zona (p. ej. -04:00 y -05:00 por el horario de verano) se pasa todo a UTC primero
        return pd.to_datetime(txt, utc=con_zona, format="mixed")

    if "date" in df.columns and "time" in df.columns:          # NinjaTrader y similares: fecha y hora separadas
        ts = _parsear(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        col_t = next((c for c in ("datetime", "timestamp", "ts_event", "date_time", "time", "date") if c in df.columns), None)
        if col_t is None:
            raise ValueError("No encontré una columna de fecha/hora (datetime, timestamp, time o date + time).")
        ts = _parsear(df[col_t])

    ts = pd.DatetimeIndex(ts)
    if ts.tz is None:
        try:
            ts = ts.tz_localize(zona_horaria, ambiguous="infer", nonexistent="shift_forward")
        except Exception:  # horas repetidas del cambio de horario sin orden claro: se descartan (están fuera de la sesión)
            ts = ts.tz_localize(zona_horaria, ambiguous="NaT", nonexistent="shift_forward")
    df.index = ts.tz_convert(NY)
    df = df[df.index.notna()]

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


# --------------------------------------------------------------------------- #
# Datos de ejemplo (simulados) para probar la herramienta sin un CSV
# --------------------------------------------------------------------------- #
def datos_ejemplo(dias: int = 520, semilla: int = 7, precio_inicial: float = 5800.0) -> pd.DataFrame:
    """
    Barras de 5 minutos SIMULADAS de la sesión regular (no son datos reales del MES).
    La mayoría de días son ruido; ~20% traen algo de tendencia intradía, para que la
    estrategia tenga algo que capturar y se pueda ver cómo funciona cada parte.
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=dias)
    n = 78                                               # barras de 5 min entre 9:30 y 16:00
    vol_dia = 0.010 * np.exp(rng.normal(0, 0.35, dias))  # volatilidad que cambia de un día a otro
    gaps = rng.normal(0.0002, 0.0045, dias)
    tendencia = np.where(rng.random(dias) < 0.20, rng.normal(0, 0.0002, dias), 0.0)  # ~1,5% de tendencia en un día con tendencia
    rets = (rng.standard_t(4, (dias, n)) / np.sqrt(2)) * (vol_dia[:, None] / np.sqrt(n)) + tendencia[:, None]
    cierres_dia = np.empty(dias)
    filas = []
    precio = precio_inicial
    minutos = 9 * 60 + 30 + 5 * np.arange(n)
    for i, f in enumerate(fechas):
        apertura = precio * (1 + gaps[i])
        cierres = apertura * np.cumprod(1 + rets[i])
        aperturas = np.r_[apertura, cierres[:-1]]
        ruido = np.abs(rng.normal(0, vol_dia[i] / np.sqrt(n) * 0.6, (2, n)))
        altos = np.maximum(aperturas, cierres) * (1 + ruido[0])
        bajos = np.minimum(aperturas, cierres) * (1 - ruido[1])
        vol = rng.integers(800, 4000, n) * (1.8 - np.sin(np.linspace(0, np.pi, n)))  # más volumen en apertura y cierre
        t = pd.Timestamp(f).tz_localize(NY) + pd.to_timedelta(minutos, unit="m")
        filas.append(pd.DataFrame({"open": aperturas, "high": altos, "low": bajos, "close": cierres, "volume": vol}, index=t))
        precio = cierres[-1]
        cierres_dia[i] = precio
    df = pd.concat(filas)
    # redondeo al tick del MES (0,25)
    for c in ("open", "high", "low", "close"):
        df[c] = (df[c] / 0.25).round() * 0.25
    return df
