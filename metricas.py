"""Estadísticas del backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
