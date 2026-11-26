"""
Prueba de humo con datos sintéticos de 1 minuto (no son datos reales: solo verifican que el
código corre y que las reglas se aplican bien). Uso:  python pruebas/prueba_sintetica.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from noise_area import (barras_rth_30m, Parametros, backtest, resumen, PRESETS,
                        ventanas_historicas, bootstrap, resumir)


def datos_sinteticos(dias=400, semilla=1):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2025-01-02", periods=dias)
    filas, precio = [], 6000.0
    for f in fechas:
        precio *= 1 + rng.normal(0, 0.004)                 # gap nocturno
        deriva = rng.normal(0, 0.00012) if rng.random() < 0.3 else 0.0   # días de tendencia
        t0 = pd.Timestamp(f).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for m in range(390):
            o = precio
            precio *= 1 + deriva + rng.normal(0, 0.0004)
            h, l = max(o, precio) * (1 + abs(rng.normal(0, 0.0001))), min(o, precio) * (1 - abs(rng.normal(0, 0.0001)))
            filas.append((t0 + pd.Timedelta(minutes=m), o, h, l, precio, rng.integers(50, 500)))
    df = pd.DataFrame(filas, columns=["t", "open", "high", "low", "close", "volume"]).set_index("t")
    return df


if __name__ == "__main__":
    crudo = datos_sinteticos()
    barras = barras_rth_30m(crudo)
    assert barras.groupby("fecha")["k"].nunique().eq(13).all(), "cada día debe tener 13 barras"

    p = Parametros(riesgo_usd=150, max_contratos=40)
    trades, diario, bandas = backtest(barras, p)
    assert (bandas.dropna(subset=["UB"])["UB"] >= bandas.dropna(subset=["UB"])["LB"]).all()
    assert trades["contratos"].le(40).all() and trades["contratos"].ge(1).all()
    assert trades["hora_salida"].le("16:00").all()
    # ninguna entrada después de la última hora permitida
    assert trades["hora_entrada"].le("15:30").all()

    for k, v in resumen(trades, diario).items():
        print(f"{k:32s} {v}")
    print("Saltados:", diario.attrs["saltados"])

    reglas = PRESETS["LucidFlex 50K"]
    print("\nVentanas históricas:", resumir(ventanas_historicas(diario, reglas)))
    print("Bootstrap:", resumir(bootstrap(diario, reglas, n=1000)))

    # El modo 'volatilidad' (paper) también debe correr
    t2, d2, _ = backtest(barras, Parametros(modo_tamano="volatilidad", max_contratos=200))
    print("\nModo volatilidad, trades:", len(t2))
    print("OK")
