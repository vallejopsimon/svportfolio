"""Backtest del momentum intradía Noise Area en MES para Radar de inversión."""
from .datos import cargar_yahoo, cargar_csv, barras_rth_30m, datos_ejemplo
from .estrategia import Parametros, backtest, calcular_bandas
from .metricas import resumen, por_periodo, por_hora_entrada, drawdown
from .prop_firm import ReglasProp, PRESETS, ventanas_historicas, bootstrap, resumir

__all__ = [
    "cargar_yahoo", "cargar_csv", "barras_rth_30m", "datos_ejemplo",
    "Parametros", "backtest", "calcular_bandas",
    "resumen", "por_periodo", "por_hora_entrada", "drawdown",
    "ReglasProp", "PRESETS", "ventanas_historicas", "bootstrap", "resumir",
]
