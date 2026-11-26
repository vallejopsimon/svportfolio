# Backtest Noise Area · MES (módulo para Radar de inversión)

Momentum intradía "Noise Area" de Zarattini, Aziz y Barbon (2024), *Beat the Market*,
adaptado a micro-futuros del S&P 500 (MES), con simulador de reglas de prop firm (LucidFlex / Topstep).

## Cómo integrarlo a tu app

```
tu_app/
├── app.py                         ← tu página principal de Streamlit
├── noise_area/                    ← copia esta carpeta
│   ├── datos.py                   carga de CSV / Yahoo y barras de 30 min RTH
│   ├── estrategia.py              señales, stops, tamaño y simulación
│   ├── metricas.py                estadísticas
│   └── prop_firm.py               reglas de evaluación (meta, trailing EOD, consistencia)
├── pages/
│   └── Backtest_Noise_Area_MES.py ← copia este archivo (Streamlit lo muestra como página)
├── correr_backtest.py             ← opcional: versión de línea de comandos
└── pruebas/prueba_sintetica.py    ← prueba de humo con datos inventados
```

Agrega a tu `requirements.txt`: `pandas>=2.0`, `numpy>=1.24`, `plotly>=5.20`,
`streamlit>=1.45`, `yfinance>=0.2.40`. Luego: `streamlit run app.py`.

## Datos

- **Yahoo Finance (`MES=F`)**: solo da unos 60 días de barras de 30 min. Sirve para ver que todo funciona, no para sacar conclusiones.
- **CSV propio (lo recomendado)**: 2 o más años de MES (o ES, que se mueve igual) en barras de 1, 5, 15 o 30 min. Puedes exportarlo de NinjaTrader, TradingView, Databento o FirstRate Data, entre otros.
  Columnas aceptadas: `datetime`/`timestamp` (o `date` + `time`), `open`, `high`, `low`, `close`, `volume`.
  Indica en la app la zona horaria en que vienen los timestamps.

## Reglas implementadas

Para cada día $d$ y cada hora de decisión $t$ (10:00, 10:30, …, 16:00 hora de Nueva York):

$$\sigma_{d,t} = \frac{1}{14}\sum_{i=d-14}^{d-1}\left|\frac{C_{i,t}}{O_i}-1\right|$$

$$UB_{d,t} = \max(O_d,\,C_{d-1})\,(1+m\,\sigma_{d,t}) \qquad LB_{d,t} = \min(O_d,\,C_{d-1})\,(1-m\,\sigma_{d,t})$$

- **Entrada**: si estás plano, compras cuando el cierre de 30 min supera $UB$ y vendes en corto cuando queda por debajo de $LB$.
- **Stop**: en un largo es $\max(UB, VWAP)$ y en un corto $\min(LB, VWAP)$. Se recalcula en cada decisión. Si activas la opción de stop real, también actúa como orden stop dentro de la media hora.
- **Salida**: todo se cierra al cierre de la sesión, en la barra de 15:30 a 16:00.
- **Tamaño (modo prop firm)**:

$$\text{contratos} = \min\left(\left\lfloor \frac{\text{riesgo USD}}{\text{distancia al stop}\times 5} \right\rfloor,\ \text{máximo de micros}\right)$$

  El modo "volatilidad objetivo" replica el tamaño del paper (2% de volatilidad diaria, apalancamiento máximo 4×).

## Lo que este backtest NO modela (tenlo en cuenta)

- Las decisiones se toman con barras de 30 min. El peor momento intradía se aproxima con el máximo y el mínimo de cada barra.
- El slippage y la comisión son fijos. En días de noticias el slippage real es mayor; prueba con 2 ticks.
- No hay vencimientos: usa una serie continua ajustada (back-adjusted) del MES o del ES.
- Las ventanas históricas de prop firm se solapan y no son independientes. El bootstrap supone que los días son independientes entre sí.
- Las reglas de las prop firms cambian seguido; ajústalas en la barra lateral según la página oficial.

Resultados hipotéticos: un backtest no garantiza resultados futuros.

## Prueba rápida

```
python pruebas/prueba_sintetica.py
python correr_backtest.py --csv tus_datos.csv --zona America/New_York --riesgo 150 --preset "LucidFlex 50K"
```
