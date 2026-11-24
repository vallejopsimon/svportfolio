# Radar de inversión

App web (Streamlit) para acciones de EE.UU., Colombia (BVC) y Latinoamérica, y para las criptos principales
(BTC, ETH, SOL, XRP). Está pensada para cualquier persona: lenguaje sencillo, recorrido en 3 pasos y explicaciones en cada pantalla.

1. **Busca oportunidades**: descarga de Yahoo Finance precios y fundamentales de un universo de acciones
   (listas predefinidas, tus propios tickers o un screener de Yahoo) y les da un puntaje 0–100 por
   valor, calidad, crecimiento y momentum, comparando cada empresa contra su sector.
2. **Valora por múltiplos**: precio implícito con P/E, P/E forward, P/B, P/S, EV/EBITDA y EV/Ventas
   frente a la mediana de comparables, más la tabla de fundamentales vs. pares y el objetivo de analistas.
3. **Optimiza el portafolio**: eliges activos y monto (USD, COP o EUR) y te entrega los pesos, el dinero
   y el número de acciones para el portafolio de **máximo Sharpe** y el de **mínima varianza**, con
   frontera eficiente, correlaciones y backtest contra el S&P 500.

## Cómo correrla en tu computador

```bash
# 1. (una sola vez) crear entorno e instalar dependencias
python -m venv .venv
.venv\Scripts\activate          # Windows   |   source .venv/bin/activate  (Mac/Linux)
pip install -r requirements.txt

# 2. abrir la app
streamlit run app.py
```

Se abre sola en el navegador en `http://localhost:8501`.

## Publicarla con un link propio (gratis)

1. Sube esta carpeta a un repositorio de GitHub.
2. Entra a https://share.streamlit.io, conecta tu GitHub y elige el repo con `app.py` como archivo principal.
3. En un par de minutos queda en `https://<tu-app>.streamlit.app`.

## Archivos

- `app.py`: interfaz y descarga de datos (yfinance).
- `.streamlit/config.toml`: colores de la app (azul marino, crema y coral). **Súbelo también a GitHub** (es una carpeta oculta: en Mac pulsa Cmd+Shift+. para verla).
- `fuentes.py`: plan B de datos. Si Yahoo bloquea su “resumen” (pasa en servidores en la nube), la app lee los estados financieros de Yahoo (ingresos, utilidad, EBITDA, patrimonio, deuda, caja, acciones) y calcula ella misma P/E, P/B, P/S, EV/EBITDA, ROE, márgenes y crecimiento, convirtiendo monedas cuando la empresa reporta en otra (p. ej. Mineros reporta en USD).
- `seguimiento.py`: pestaña «Mi seguimiento». A partir de las compras registradas calcula la evolución diaria, la ganancia (separando el efecto del precio y el de la tasa de cambio), la rentabilidad anual (XIRR), la volatilidad, el Sharpe, la peor caída, el beta, la comparación contra el S&P 500 con las mismas compras, y recomendaciones según el perfil. El registro se guarda descargándolo en CSV (la app no guarda datos en servidores).
- `montecarlo.py` y `grafica_mc.py`: pestaña «Proyecciones». Simulación Monte Carlo (bootstrap histórico o movimiento browniano geométrico) con miles de escenarios, precio objetivo (mediana), rango P5–P95, probabilidades y VaR; la gráfica animada se dibuja en el navegador.
- `motor.py`: cálculos puros — puntajes, múltiplos, Ledoit-Wolf, optimización (SciPy SLSQP).

## Notas

- Tickers en formato Yahoo: Colombia termina en `.CL` (`GEB.CL`, `CELSIA.CL`), criptos en `-USD` (`ADA-USD`), otras bolsas con su sufijo (`SAP.DE`, `PETR4.SA`).
- Si Yahoo no tiene datos de una acción colombiana, la app intenta su otro listado (p. ej. Mineros → `MSA.TO` en Toronto, Cibest → `CIB` en Nueva York).
- Las criptos no tienen puntaje de empresa (no tienen estados financieros), pero sí entran a la optimización y se compran en fracciones.
- La tasa libre de riesgo por defecto es el T-Bill a 13 semanas (`^IRX`); puedes cambiarla.
- Los datos se guardan en caché 1 hora; usa “Cargar / actualizar datos” para refrescar.
- Google Finance no tiene API pública, por eso la fuente es Yahoo Finance.
- Herramienta educativa: no es recomendación de inversión.
