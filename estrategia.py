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
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd


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
