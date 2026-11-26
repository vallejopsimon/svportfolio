"""
Noise Area (Zarattini, Aziz y Barbon, 2024) para micro-futuros MES: datos, estrategia, métricas y
simulador de evaluaciones de prop firm, todo en un solo archivo para subirlo fácil a GitHub.
"""
from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# ===========================================================================
# datos
# ===========================================================================
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

# ===========================================================================
# estrategia
# ===========================================================================
"""
Momentum intradía "Noise Area" (Zarattini, Aziz y Barbon, 2024, "Beat the Market")
adaptado a micro-futuros MES.

Reglas:
  sigma_{d,t} = promedio de |close_{i,t} / open_i - 1| en los últimos `lookback` días (sin incluir hoy)
  UB_{d,t}    = max(open_d, close_{d-1}) * (1 + m * sigma_{d,t})
  LB_{d,t}    = min(open_d, close_{d-1}) * (1 - m * sigma_{d,t})

  Solo se decide al cierre de cada barra de 30 min (10:00, 10:30, ..., 16:00 NY).
  Plano  -> largo si close > UB ; corto si close < LB.
  Largo  -> stop = max(UB, VWAP); corto -> stop = min(LB, VWAP).
  Todo se cierra en la última barra del día (sin posiciones de noche).
"""



@dataclass
class Parametros:
    # Señal
    lookback: int = 14                 # días para promediar el ruido
    multiplicador: float = 1.0         # ancho de la banda (1.0 = paper)
    usar_vwap_stop: bool = True        # stop = max(UB, VWAP) en vez de solo UB
    permitir_cortos: bool = True
    primera_k: int = 0                 # primera decisión: 0 -> 10:00
    ultima_k_entrada: int = 11         # última entrada nueva: 11 -> 15:30
    max_perdidas_dia: int = 2          # deja de operar el día tras N pérdidas (0 = sin límite)

    # Contrato MES
    valor_punto: float = 5.0           # USD por punto (MES)
    tick: float = 0.25
    slippage_ticks: float = 1.0        # por lado
    comision_rt: float = 1.50          # USD por contrato ida y vuelta

    # Tamaño
    modo_tamano: str = "riesgo"        # "riesgo" (prop firm) o "volatilidad" (paper)
    riesgo_usd: float = 150.0          # riesgo por trade en modo "riesgo"
    stop_min_puntos: float = 3.0       # distancia mínima usada para dimensionar
    stop_max_puntos: float = 0.0       # si > 0, no entra cuando el stop queda más lejos
    forzar_un_contrato: bool = False   # si el cálculo da 0 contratos, entra con 1
    max_contratos: int = 40            # 40 micros = LucidFlex 50K
    capital: float = 50_000.0          # modo "volatilidad"
    vol_objetivo: float = 0.02         # volatilidad diaria objetivo (paper: 2%)
    apalancamiento_max: float = 4.0

    # Ejecución del stop
    stop_duro: bool = True             # orden stop real entre decisiones (recomendado en prop firm)

    def dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Bandas de ruido
# --------------------------------------------------------------------------- #
def calcular_bandas(barras: pd.DataFrame, p: Parametros) -> pd.DataFrame:
    b = barras.copy()
    close = b.pivot(index="fecha", columns="k", values="close").sort_index()
    apertura = b[b["k"] == 0].set_index("fecha")["open"].reindex(close.index)
    ultimo_k = close.columns.max()

    mov = close.abs().copy()
    mov.loc[:, :] = np.abs(close.values / apertura.values[:, None] - 1.0)
    sigma = mov.rolling(p.lookback, min_periods=p.lookback).mean().shift(1)

    cierre_ant = close[ultimo_k].shift(1)
    ret_diario = close[ultimo_k] / cierre_ant - 1.0
    vol_diaria = ret_diario.rolling(p.lookback, min_periods=p.lookback).std().shift(1)

    sig_l = sigma.stack().rename("sigma").reset_index()
    b = b.merge(sig_l, on=["fecha", "k"], how="left")
    b["apertura_dia"] = b["fecha"].map(apertura)
    b["cierre_ant"] = b["fecha"].map(cierre_ant)
    b["vol_diaria"] = b["fecha"].map(vol_diaria)

    base_up = np.maximum(b["apertura_dia"], b["cierre_ant"])
    base_dn = np.minimum(b["apertura_dia"], b["cierre_ant"])
    b["UB"] = base_up * (1 + p.multiplicador * b["sigma"])
    b["LB"] = base_dn * (1 - p.multiplicador * b["sigma"])
    return b


# --------------------------------------------------------------------------- #
# Tamaño de posición
# --------------------------------------------------------------------------- #
def _contratos(p: Parametros, precio: float, dist_puntos: float, vol_diaria: float) -> int:
    if p.modo_tamano == "volatilidad":
        if not np.isfinite(vol_diaria) or vol_diaria <= 0:
            return 0
        lev = min(p.apalancamiento_max, p.vol_objetivo / vol_diaria)
        n = math.floor(p.capital * lev / (precio * p.valor_punto))
    else:
        dist = max(dist_puntos, p.stop_min_puntos)
        n = math.floor(p.riesgo_usd / (dist * p.valor_punto))
        if n == 0 and p.forzar_un_contrato:
            n = 1
    return int(max(0, min(n, p.max_contratos)))


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(barras: pd.DataFrame, p: Parametros | None = None):
    """
    Devuelve (trades, diario, bandas):
      trades : una fila por operación
      diario : P&L por día (USD), peor P&L intradía (para reglas de drawdown) y nº de trades
      bandas : las barras con UB/LB/VWAP calculados (para graficar un día)
    """
    p = p or Parametros()
    b = calcular_bandas(barras, p)
    slip = p.slippage_ticks * p.tick
    trades, diario = [], []
    saltados = {"sin_contratos": 0, "stop_muy_lejos": 0}

    for fecha, dia in b.groupby("fecha", sort=True):
        dia = dia.sort_values("k")
        if dia["sigma"].isna().all():
            continue                                     # aún no hay historia suficiente
        realizado, peor, n_tr, perdidas = 0.0, 0.0, 0, 0
        pos, qty, px_in, stop, stop_ini, t_in, k_in = 0, 0, 0.0, np.nan, np.nan, "", -1
        ultima = dia["k"].max()

        def cerrar(px_salida, motivo, hora):
            nonlocal realizado, pos, qty, n_tr, perdidas
            pnl = pos * (px_salida - px_in) * qty * p.valor_punto - p.comision_rt * qty
            riesgo = abs(px_in - stop_ini) * qty * p.valor_punto
            trades.append(dict(
                fecha=fecha, lado="Largo" if pos > 0 else "Corto", contratos=qty,
                hora_entrada=t_in, precio_entrada=round(px_in, 2), stop_inicial=round(stop_ini, 2),
                hora_salida=hora, precio_salida=round(px_salida, 2), motivo=motivo,
                pnl_usd=round(pnl, 2), R=round(pnl / riesgo, 3) if riesgo > 0 else np.nan,
            ))
            realizado += pnl
            n_tr += 1
            perdidas += pnl < 0
            pos, qty = 0, 0

        for fila in dia.itertuples(index=False):
            k = fila.k
            # 1) Stop duro dentro de la barra (nivel fijado en la decisión anterior)
            if pos != 0 and p.stop_duro and np.isfinite(stop):
                tocado = fila.low <= stop if pos > 0 else fila.high >= stop
                if tocado:
                    base = min(fila.open, stop) if pos > 0 else max(fila.open, stop)
                    cerrar(base - pos * slip, "Stop (orden)", fila.hora_cierre)

            # Peor P&L intradía (marcado a mercado en el extremo adverso de la barra)
            if pos != 0:
                adverso = fila.low if pos > 0 else fila.high
                abierto = pos * (adverso - px_in) * qty * p.valor_punto
                peor = min(peor, realizado + abierto)
            peor = min(peor, realizado)

            if not np.isfinite(fila.sigma) or k < p.primera_k:
                continue

            # 2) Salida por cierre de sesión
            if k == ultima:
                if pos != 0:
                    cerrar(fila.close - pos * slip, "Cierre del día", fila.hora_cierre)
                break

            # 3) Gestión de la posición abierta en la decisión de las HH:00 / HH:30
            if pos > 0:
                nivel = max(fila.UB, fila.vwap) if p.usar_vwap_stop else fila.UB
                if fila.close < nivel:
                    cerrar(fila.close - slip, "Stop (decisión)", fila.hora_cierre)
                else:
                    stop = nivel
            elif pos < 0:
                nivel = min(fila.LB, fila.vwap) if p.usar_vwap_stop else fila.LB
                if fila.close > nivel:
                    cerrar(fila.close + slip, "Stop (decisión)", fila.hora_cierre)
                else:
                    stop = nivel

            # 4) Entradas nuevas
            bloqueado = p.max_perdidas_dia > 0 and perdidas >= p.max_perdidas_dia
            if pos == 0 and k <= p.ultima_k_entrada and not bloqueado:
                lado = 0
                if fila.close > fila.UB:
                    lado, nivel = 1, (max(fila.UB, fila.vwap) if p.usar_vwap_stop else fila.UB)
                elif fila.close < fila.LB and p.permitir_cortos:
                    lado, nivel = -1, (min(fila.LB, fila.vwap) if p.usar_vwap_stop else fila.LB)
                if lado != 0:
                    dist = (fila.close - nivel) * lado
                    if dist <= 0:
                        continue                         # el VWAP está del lado equivocado
                    if p.stop_max_puntos > 0 and dist > p.stop_max_puntos:
                        saltados["stop_muy_lejos"] += 1
                        continue
                    n = _contratos(p, fila.close, dist, fila.vol_diaria)
                    if n == 0:
                        saltados["sin_contratos"] += 1
                        continue
                    pos, qty = lado, n
                    px_in = fila.close + lado * slip
                    stop = stop_ini = nivel
                    t_in, k_in = fila.hora_cierre, k

        diario.append(dict(fecha=fecha, pnl_usd=realizado, peor_intradia=peor, trades=n_tr))

    trades = pd.DataFrame(trades)
    diario = pd.DataFrame(diario)
    if not diario.empty:
        diario["fecha"] = pd.to_datetime(diario["fecha"])
        diario["equity"] = diario["pnl_usd"].cumsum()
    if not trades.empty:
        trades["fecha"] = pd.to_datetime(trades["fecha"])
    diario.attrs["saltados"] = saltados
    return trades, diario, b

# ===========================================================================
# metricas
# ===========================================================================
"""Estadísticas del backtest."""



def resumen(trades: pd.DataFrame, diario: pd.DataFrame) -> dict:
    if trades.empty or diario.empty:
        return {"Trades": 0}
    pnl = trades["pnl_usd"]
    gan, per = pnl[pnl > 0], pnl[pnl < 0]
    eq = diario["equity"]
    dd = eq - eq.cummax().clip(lower=0)
    d = diario["pnl_usd"]
    sharpe = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan
    pf = gan.sum() / -per.sum() if len(per) else np.inf
    anios = max((diario["fecha"].iloc[-1] - diario["fecha"].iloc[0]).days / 365.25, 1 / 252)
    dias_op = (diario["trades"] > 0).sum()
    mejor = d.max()
    return {
        "P&L neto (USD)": round(pnl.sum(), 2),
        "Trades": int(len(trades)),
        "Trades por día operado": round(len(trades) / max(dias_op, 1), 2),
        "Días en la muestra": int(len(diario)),
        "% de días con trade": round(100 * dias_op / len(diario), 1),
        "Tasa de acierto (%)": round(100 * len(gan) / len(pnl), 1),
        "Ganancia media (USD)": round(gan.mean(), 2) if len(gan) else 0.0,
        "Pérdida media (USD)": round(per.mean(), 2) if len(per) else 0.0,
        "Profit factor": round(pf, 2),
        "Esperanza por trade (R)": round(trades["R"].mean(), 3),
        "Esperanza por trade (USD)": round(pnl.mean(), 2),
        "Sharpe (diario anualizado)": round(sharpe, 2),
        "Drawdown máximo (USD)": round(dd.min(), 2),
        "P&L por año (USD)": round(pnl.sum() / anios, 2),
        "Mejor día (USD)": round(mejor, 2),
        "Mejor día / P&L total (%)": round(100 * mejor / pnl.sum(), 1) if pnl.sum() > 0 else np.nan,
        "Peor racha de pérdidas (trades)": int(_peor_racha(pnl)),
        "t-stat de la media diaria": round(d.mean() / (d.std() / np.sqrt(len(d))), 2) if d.std() > 0 else np.nan,
    }


def _peor_racha(pnl: pd.Series) -> int:
    peor = actual = 0
    for x in pnl:
        actual = actual + 1 if x < 0 else 0
        peor = max(peor, actual)
    return peor


def por_periodo(trades: pd.DataFrame, freq: str = "Y") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["periodo"] = t["fecha"].dt.year if freq == "Y" else t["fecha"].dt.to_period("M").astype(str)
    g = t.groupby("periodo")["pnl_usd"]
    out = pd.DataFrame({
        "Trades": g.size(),
        "P&L (USD)": g.sum().round(2),
        "Acierto (%)": (g.apply(lambda s: (s > 0).mean()) * 100).round(1),
        "Profit factor": g.apply(lambda s: s[s > 0].sum() / -s[s < 0].sum() if (s < 0).any() else np.inf).round(2),
    })
    out.index.name = "Año" if freq == "Y" else "Mes"
    return out


def por_hora_entrada(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("hora_entrada")["pnl_usd"]
    return pd.DataFrame({
        "Trades": g.size(),
        "P&L (USD)": g.sum().round(2),
        "Acierto (%)": (g.apply(lambda s: (s > 0).mean()) * 100).round(1),
        "P&L medio (USD)": g.mean().round(2),
    })


def drawdown(diario: pd.DataFrame) -> pd.Series:
    eq = diario["equity"]
    return eq - eq.cummax().clip(lower=0)

# ===========================================================================
# prop_firm
# ===========================================================================
"""
Simulador de reglas de evaluación de prop firms sobre el P&L diario real del backtest.

Dos métodos:
  1. Ventanas históricas: arranca una evaluación cada día de la muestra y la sigue
     con los días reales que vinieron después. (Las ventanas se solapan: no son independientes.)
  2. Bootstrap: arma miles de trayectorias remuestreando días reales al azar (con reemplazo).

Reglas modeladas (verifícalas siempre en la página de la firma, cambian seguido):
  - Meta de ganancia.
  - Pérdida máxima trailing EOD: el piso = pico de saldo al cierre - MLL, y se congela en
    saldo inicial + `bloqueo` cuando el pico supera MLL + bloqueo. Se rompe si el peor P&L
    intradía (marcado a mercado) toca el piso.
  - Límite de pérdida diaria (opcional). En Lucid/Topstep es "suave": detiene el día, no quema la
    cuenta. Aquí se modela como suave: el P&L del día se recorta al límite.
  - Consistencia: el mejor día debe ser <= c * ganancia total en el momento de pasar.
"""



@dataclass
class ReglasProp:
    nombre: str = "LucidFlex 50K"
    meta: float = 3000.0
    mll: float = 2000.0
    bloqueo: float = 100.0             # el trailing se congela en saldo inicial + 100
    dll: float = 0.0                   # 0 = sin límite diario
    consistencia: float = 0.50         # 0 = sin regla
    max_dias: int = 0                  # 0 = sin límite de tiempo
    max_contratos_micro: int = 40
    costo_evaluacion: float = 0.0      # opcional, para el cálculo de valor esperado


PRESETS = {
    "LucidFlex 25K": ReglasProp("LucidFlex 25K", 1250, 1000, 100, 0, 0.50, 0, 20),
    "LucidFlex 50K": ReglasProp("LucidFlex 50K", 3000, 2000, 100, 0, 0.50, 0, 40),
    "LucidFlex 100K": ReglasProp("LucidFlex 100K", 6000, 3000, 100, 0, 0.50, 0, 60),
    "Topstep 50K": ReglasProp("Topstep 50K", 3000, 2000, 0, 1000, 0.50, 0, 50),
    "Personalizada": ReglasProp("Personalizada"),
}


def _evaluar(pnl: np.ndarray, peor: np.ndarray, r: ReglasProp):
    """Recorre una secuencia de días. Devuelve (resultado, días usados)."""
    saldo, pico, mejor = 0.0, 0.0, 0.0
    for i in range(len(pnl)):
        d, w = pnl[i], peor[i]
        if r.dll > 0:
            d = max(d, -r.dll)
            w = max(w, -r.dll)
        piso = min(pico - r.mll, r.bloqueo)          # trailing que se congela en inicial + bloqueo
        if saldo + w <= piso:
            return "Falla", i + 1
        saldo += d
        mejor = max(mejor, d)
        pico = max(pico, saldo)
        if saldo >= r.meta and (r.consistencia <= 0 or mejor <= r.consistencia * saldo):
            return "Pasa", i + 1
        if r.max_dias and i + 1 >= r.max_dias:
            return "Falla", i + 1
    return "Sin terminar", len(pnl)


def ventanas_historicas(diario: pd.DataFrame, r: ReglasProp) -> pd.DataFrame:
    pnl = diario["pnl_usd"].to_numpy(float)
    peor = diario["peor_intradia"].to_numpy(float)
    filas = []
    for i in range(len(pnl)):
        res, dias = _evaluar(pnl[i:], peor[i:], r)
        filas.append(dict(inicio=diario["fecha"].iloc[i], resultado=res, dias=dias))
    return pd.DataFrame(filas)


def bootstrap(diario: pd.DataFrame, r: ReglasProp, n: int = 5000, horizonte: int = 250,
              semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    pnl = diario["pnl_usd"].to_numpy(float)
    peor = diario["peor_intradia"].to_numpy(float)
    filas = []
    for _ in range(n):
        idx = rng.integers(0, len(pnl), horizonte)
        res, dias = _evaluar(pnl[idx], peor[idx], r)
        filas.append(dict(resultado=res, dias=dias))
    return pd.DataFrame(filas)


def resumir(res: pd.DataFrame) -> dict:
    if res.empty:
        return {}
    terminadas = res[res["resultado"] != "Sin terminar"]
    pasa = res[res["resultado"] == "Pasa"]
    return {
        "Evaluaciones simuladas": int(len(res)),
        "Pasan (%)": float(round(100 * (res["resultado"] == "Pasa").mean(), 1)),
        "Fallan (%)": float(round(100 * (res["resultado"] == "Falla").mean(), 1)),
        "Sin terminar (%)": float(round(100 * (res["resultado"] == "Sin terminar").mean(), 1)),
        "Pasan entre las terminadas (%)": float(round(100 * (terminadas["resultado"] == "Pasa").mean(), 1))
        if len(terminadas) else np.nan,
        "Días medianos para pasar": float(pasa["dias"].median()) if len(pasa) else np.nan,
        "Días medianos para fallar": float(res.loc[res["resultado"] == "Falla", "dias"].median())
        if (res["resultado"] == "Falla").any() else np.nan,
    }
