"""
Backtest por línea de comandos (sin Streamlit).

Ejemplos:
  python correr_backtest.py --csv datos/mes_1min.csv --zona America/New_York --riesgo 150 --preset "LucidFlex 50K"
  python correr_backtest.py --yahoo
"""
import argparse

from noise_area import (cargar_csv, cargar_yahoo, barras_rth_30m, Parametros, backtest,
                        resumen, por_periodo, PRESETS, ventanas_historicas, bootstrap, resumir)


def main():
    ap = argparse.ArgumentParser(description="Backtest Noise Area en MES")
    fuente = ap.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--csv", help="ruta a un CSV de 1-30 min")
    fuente.add_argument("--yahoo", action="store_true", help="MES=F de Yahoo (30m, ~60 días)")
    ap.add_argument("--zona", default="America/New_York", help="zona horaria del CSV")
    ap.add_argument("--riesgo", type=float, default=150.0, help="riesgo por trade en USD")
    ap.add_argument("--preset", default="LucidFlex 50K", choices=list(PRESETS))
    ap.add_argument("--slippage", type=float, default=1.0, help="ticks por lado")
    ap.add_argument("--comision", type=float, default=1.5, help="USD ida y vuelta por contrato")
    ap.add_argument("--salida", default="trades_noise_area_mes.csv")
    a = ap.parse_args()

    crudo = cargar_yahoo() if a.yahoo else cargar_csv(a.csv, a.zona)
    barras = barras_rth_30m(crudo)
    reglas = PRESETS[a.preset]
    p = Parametros(riesgo_usd=a.riesgo, slippage_ticks=a.slippage, comision_rt=a.comision,
                   max_contratos=reglas.max_contratos_micro)
    trades, diario, _ = backtest(barras, p)

    print(f"\n=== Noise Area · MES · {barras['fecha'].min()} a {barras['fecha'].max()} ===")
    for k, v in resumen(trades, diario).items():
        print(f"  {k:34s} {v}")
    print("\nPor año:\n", por_periodo(trades, "Y").to_string())
    print(f"\n=== {reglas.nombre} ===")
    print("  Ventanas históricas:", resumir(ventanas_historicas(diario, reglas)))
    print("  Bootstrap:          ", resumir(bootstrap(diario, reglas, n=5000)))
    trades.to_csv(a.salida, index=False)
    print(f"\nTrades guardados en {a.salida}")


if __name__ == "__main__":
    main()
