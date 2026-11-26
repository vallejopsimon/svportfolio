"""
Proyecciones de precio con simulación Monte Carlo.

Dos modelos:
  • GBM (movimiento browniano geométrico): los retornos logarítmicos diarios son normales con la media y la
    volatilidad históricas.  ln S_{t+1} = ln S_t + (μ − σ²/2)Δt + σ√Δt · Z
  • Bootstrap histórico: cada día se toma al azar un retorno real del pasado. Conserva las "colas gordas"
    (días extremos) que la normal subestima.

La tendencia (drift) se puede elegir: la histórica, una moderada (mitad histórica, mitad tasa libre de riesgo)
o sin tendencia (solo la tasa libre de riesgo), porque la tendencia pasada suele exagerar el futuro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TENDENCIAS = ("Histórica", "Moderada", "Sin tendencia")
MODELOS = ("GBM", "Bootstrap")


def parametros(precios: pd.Series) -> dict:
    """Retornos logarítmicos diarios, periodos por año, μ y σ anuales (aritméticos)."""
    s = precios.dropna()
    s = s[s > 0]
    r = np.log(s).diff().dropna()
    anios = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    per = len(r) / anios
    sigma = r.std(ddof=1) * np.sqrt(per)
    mu = r.mean() * per + sigma ** 2 / 2  # media aritmética anual
    return dict(r=r.values, per=per, mu=float(mu), sigma=float(sigma), s0=float(s.iloc[-1]),
                n_obs=len(r), desde=s.index[0], hasta=s.index[-1])


def drift_elegido(mu_hist: float, rf: float, tendencia: str) -> float:
    if tendencia == "Sin tendencia":
        return rf
    if tendencia == "Moderada":
        return 0.5 * mu_hist + 0.5 * rf
    return mu_hist


def simular(p: dict, anios: float, n: int = 5000, modelo: str = "GBM", tendencia: str = "Moderada",
            rf: float = 0.04, semilla: int = 42) -> dict:
    rng = np.random.default_rng(semilla)
    pasos = max(int(round(anios * p["per"])), 2)
    dt = 1 / p["per"]
    mu = drift_elegido(p["mu"], rf, tendencia)
    sig = p["sigma"]
    if modelo == "Bootstrap":
        r = p["r"]
        centrados = r - r.mean()  # se quita la tendencia histórica y se pone la elegida
        diarios = rng.choice(centrados, size=(n, pasos), replace=True) + (mu - sig ** 2 / 2) * dt
    else:
        diarios = (mu - sig ** 2 / 2) * dt + sig * np.sqrt(dt) * rng.standard_normal((n, pasos))
    caminos = p["s0"] * np.exp(np.cumsum(diarios, axis=1))
    caminos = np.hstack([np.full((n, 1), p["s0"]), caminos])
    return dict(caminos=caminos, pasos=pasos, mu=mu, sigma=sig, anios=anios, modelo=modelo, tendencia=tendencia, n=n)


def resumen(sim: dict, s0: float, objetivo: float | None = None) -> dict:
    fin = sim["caminos"][:, -1]
    pct = {q: float(np.percentile(fin, q)) for q in (5, 10, 25, 50, 75, 90, 95)}
    out = dict(
        s0=s0, mediana=pct[50], media=float(fin.mean()), pct=pct,
        prob_sube=float((fin > s0).mean()), prob_cae20=float((fin < 0.8 * s0).mean()),
        prob_sube20=float((fin > 1.2 * s0).mean()),
        var95=float(1 - pct[5] / s0),  # pérdida que solo se supera en 5% de los escenarios
        ret_mediano=pct[50] / s0 - 1,
    )
    out["ret_mediano_anual"] = (pct[50] / s0) ** (1 / sim["anios"]) - 1 if sim["anios"] > 0 else np.nan
    if objetivo:
        out["objetivo"] = objetivo
        out["prob_objetivo"] = float((fin >= objetivo).mean()) if objetivo >= s0 else float((fin <= objetivo).mean())
        alcanzo = (sim["caminos"] >= objetivo).any(axis=1) if objetivo >= s0 else (sim["caminos"] <= objetivo).any(axis=1)
        out["prob_toca_objetivo"] = float(alcanzo.mean())
    return out


def datos_animacion(sim: dict, res: dict, puntos: int = 120, caminos: int = 140, bins: int = 36, semilla: int = 7) -> dict:
    """Reduce la simulación a lo que necesita la gráfica animada (liviano para enviar al navegador)."""
    C = sim["caminos"]
    idx = np.unique(np.linspace(0, C.shape[1] - 1, min(puntos, C.shape[1])).round().astype(int))
    rng = np.random.default_rng(semilla)
    elegidos = rng.choice(C.shape[0], size=min(caminos, C.shape[0]), replace=False)
    muestra = C[elegidos][:, idx]
    bandas = {str(q): np.percentile(C[:, idx], q, axis=0).round(6).tolist() for q in (5, 25, 50, 75, 95)}
    fin = C[:, -1]
    lo, hi = np.percentile(fin, 0.5), np.percentile(fin, 99.5)
    cuentas, bordes = np.histogram(np.clip(fin, lo, hi), bins=bins, range=(lo, hi))
    finales = muestra[:, -1]
    return dict(
        x=(idx / sim["pasos"]).round(5).tolist(), dias=int(round(sim["anios"] * 365)),
        caminos=muestra.round(6).tolist(), sube=(finales >= res["s0"]).tolist(),
        bandas=bandas, s0=res["s0"], hist=cuentas.tolist(), bordes=bordes.round(6).tolist(),
        mediana=res["mediana"], p5=res["pct"][5], p95=res["pct"][95], objetivo=res.get("objetivo"),
    )
