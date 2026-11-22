# Radar de Renta Variable

App web (Streamlit) que:

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
- `motor.py`: cálculos puros — puntajes, múltiplos, Ledoit-Wolf, optimización (SciPy SLSQP).

## Notas

- Tickers en formato Yahoo: acciones de otras bolsas llevan sufijo (`SAP.DE`, `PETR4.SA`, `WALMEX.MX`).
- La tasa libre de riesgo por defecto es el T-Bill a 13 semanas (`^IRX`); puedes cambiarla.
- Los datos se guardan en caché 1 hora; usa “Cargar / actualizar datos” para refrescar.
- Google Finance no tiene API pública, por eso la fuente es Yahoo Finance.
- Herramienta educativa: no es recomendación de inversión.
