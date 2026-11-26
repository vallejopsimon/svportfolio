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
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


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
