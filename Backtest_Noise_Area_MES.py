"""
Página de Streamlit: Backtest del momentum intradía Noise Area en MES.
Cópiala a la carpeta `pages/` de tu app y la carpeta `noise_area/` a la raíz.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noise_area import (cargar_yahoo, cargar_csv, barras_rth_30m, Parametros, backtest,
                        resumen, por_periodo, por_hora_entrada, drawdown,
                        PRESETS, ventanas_historicas, bootstrap, resumir)

AZUL, NARANJA, ROJO, GRIS = "#257de0", "#f47230", "#ec4647", "#666666"
HORAS = [f"{(570 + 30 * (k + 1)) // 60:02d}:{(570 + 30 * (k + 1)) % 60:02d}" for k in range(13)]

try:
    st.set_page_config(page_title="Backtest Noise Area · MES", layout="wide")
except Exception:
    pass

st.title("Backtest · Momentum intradía Noise Area en MES")
st.caption("Zarattini, Aziz y Barbon (2024), *Beat the Market*. Decisiones cada 30 min en la "
           "sesión regular (9:30–16:00 NY). Resultados hipotéticos: no garantizan rendimientos futuros.")

# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Datos")
    fuente = st.radio("Fuente", ["Archivo CSV (recomendado)", "Yahoo Finance (MES=F, ~60 días)"])
    archivo, zona = None, "America/New_York"
    if fuente.startswith("Archivo"):
        archivo = st.file_uploader("CSV de 1, 5, 15 o 30 min", type=["csv", "txt"])
        zona = st.selectbox("Zona horaria de los timestamps",
                            ["America/New_York", "UTC", "America/Chicago", "America/Bogota"])
    else:
        st.info("Yahoo solo da ~60 días de barras de 30 min: sirve para ver que todo funciona, "
                "no para sacar conclusiones.")

    st.header("Prop firm")
    preset = st.selectbox("Reglas", list(PRESETS), index=1)
    base = PRESETS[preset]
    with st.expander("Ajustar reglas", expanded=preset == "Personalizada"):
        meta = st.number_input("Meta de ganancia (USD)", 0.0, value=float(base.meta), step=250.0)
        mll = st.number_input("Pérdida máxima trailing EOD (USD)", 0.0, value=float(base.mll), step=250.0)
        bloqueo = st.number_input("El trailing se congela en inicial + (USD)", value=float(base.bloqueo), step=50.0)
        dll = st.number_input("Límite de pérdida diaria (0 = no tiene)", 0.0, value=float(base.dll), step=100.0)
        cons = st.slider("Consistencia: mejor día ≤ % de la ganancia (0 = no tiene)", 0, 100,
                         int(base.consistencia * 100), step=5)
        max_micros = st.number_input("Máximo de micros", 1, 200, int(base.max_contratos_micro))
    reglas = replace(base, meta=meta, mll=mll, bloqueo=bloqueo, dll=dll,
                     consistencia=cons / 100, max_contratos_micro=int(max_micros))

    st.header("Estrategia")
    lookback = st.number_input("Días para medir el ruido (lookback)", 5, 60, 14)
    mult = st.slider("Ancho de la banda (multiplicador)", 0.5, 2.0, 1.0, 0.05)
    vwap_stop = st.checkbox("Stop = máx(UB, VWAP)", True)
    cortos = st.checkbox("Permitir cortos", True)
    k_ini = st.selectbox("Primera decisión", HORAS[:-1], index=0)
    k_fin = st.selectbox("Última entrada nueva", HORAS[:-1], index=11)
    max_perd = st.number_input("Parar el día tras N pérdidas (0 = sin límite)", 0, 10, 2)
    stop_duro = st.checkbox("Stop como orden real entre decisiones", True,
                            help="Recomendado en prop firm: limita la pérdida dentro de la media hora.")

    st.header("Tamaño y costos")
    modo = st.radio("Tamaño de posición", ["Riesgo fijo en USD (prop firm)", "Volatilidad objetivo (paper)"])
    riesgo = st.number_input("Riesgo por trade (USD)", 10.0, value=150.0, step=10.0)
    stop_min = st.number_input("Distancia mínima de stop para dimensionar (pts)", 0.25, value=3.0, step=0.25)
    stop_max = st.number_input("No entrar si el stop está a más de (pts, 0 = sin filtro)", 0.0, value=0.0, step=1.0)
    forzar1 = st.checkbox("Si da 0 contratos, entrar con 1", False)
    capital = st.number_input("Capital (modo volatilidad)", 1000.0, value=50_000.0, step=5000.0)
    slip = st.number_input("Slippage por lado (ticks)", 0.0, value=1.0, step=0.5)
    comision = st.number_input("Comisión ida y vuelta por contrato (USD)", 0.0, value=1.50, step=0.10)

    n_boot = st.select_slider("Trayectorias bootstrap", [1000, 2000, 5000, 10000], 2000)
    correr = st.button("Correr backtest", type="primary", width="stretch")


@st.cache_data(show_spinner=False)
def _barras_csv(contenido: bytes, zona: str):
    return barras_rth_30m(cargar_csv(contenido, zona))


@st.cache_data(show_spinner=False, ttl=3600)
def _barras_yahoo():
    return barras_rth_30m(cargar_yahoo("MES=F", "30m", "60d"))


if not correr and "na_res" not in st.session_state:
    st.markdown(
        "**Cómo usarla:** sube un CSV de MES en barras de 1–30 min (columnas fecha/hora, open, high, low, "
        "close, volume), ajusta los parámetros y pulsa **Correr backtest**. Para conclusiones serias usa "
        "al menos 2–3 años de datos."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
if correr:
    try:
        with st.spinner("Cargando datos..."):
            if fuente.startswith("Archivo"):
                if archivo is None:
                    st.warning("Sube un CSV primero.")
                    st.stop()
                barras = _barras_csv(archivo.getvalue(), zona)
            else:
                barras = _barras_yahoo()
    except Exception as e:
        st.error(f"No pude cargar los datos: {e}")
        st.stop()

    p = Parametros(
        lookback=int(lookback), multiplicador=float(mult), usar_vwap_stop=vwap_stop,
        permitir_cortos=cortos, primera_k=HORAS.index(k_ini), ultima_k_entrada=HORAS.index(k_fin),
        max_perdidas_dia=int(max_perd), slippage_ticks=float(slip), comision_rt=float(comision),
        modo_tamano="riesgo" if modo.startswith("Riesgo") else "volatilidad",
        riesgo_usd=float(riesgo), stop_min_puntos=float(stop_min), stop_max_puntos=float(stop_max),
        forzar_un_contrato=forzar1, max_contratos=int(reglas.max_contratos_micro),
        capital=float(capital), stop_duro=stop_duro,
    )
    with st.spinner("Corriendo backtest..."):
        trades, diario, bandas = backtest(barras, p)
        vent = ventanas_historicas(diario, reglas) if not diario.empty else pd.DataFrame()
        boot = bootstrap(diario, reglas, n=int(n_boot)) if not diario.empty else pd.DataFrame()
    st.session_state["na_res"] = (trades, diario, bandas, vent, boot, reglas, diario.attrs.get("saltados", {}))

trades, diario, bandas, vent, boot, reglas, saltados = st.session_state["na_res"]

if trades.empty:
    st.warning("No hubo operaciones. Revisa que haya al menos lookback + 1 días completos de datos.")
    st.stop()

if len(diario) < 250:
    st.warning(f"La muestra tiene solo {len(diario)} días hábiles. Con menos de ~1 año los resultados "
               "son casi pura suerte; úsalos solo para revisar que todo funciona.")

r = resumen(trades, diario)

# --------------------------------------------------------------------------- #
# Indicadores principales
# --------------------------------------------------------------------------- #
c = st.columns(6)
c[0].metric("P&L neto", f"${r['P&L neto (USD)']:,.0f}")
c[1].metric("Trades", f"{r['Trades']:,}", f"{r['Trades por día operado']} por día operado", delta_color="off")
c[2].metric("Acierto", f"{r['Tasa de acierto (%)']}%")
c[3].metric("Profit factor", f"{r['Profit factor']}")
c[4].metric("Sharpe", f"{r['Sharpe (diario anualizado)']}")
c[5].metric("Drawdown máx.", f"${r['Drawdown máximo (USD)']:,.0f}")

tabs = st.tabs(["Curva", f"Prop firm · {reglas.nombre}", "Trades", "Por periodo y hora", "Ver un día"])

with tabs[0]:
    fig = go.Figure(go.Scatter(x=diario["fecha"], y=diario["equity"], mode="lines",
                               line=dict(color=AZUL, width=2),
                               hovertemplate="%{x|%d %b %Y}<br>P&L acumulado: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="P&L acumulado (USD)", height=380, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    dd = drawdown(diario)
    fig2 = go.Figure(go.Scatter(x=diario["fecha"], y=dd, mode="lines", fill="tozeroy",
                                line=dict(color=ROJO, width=2),
                                hovertemplate="%{x|%d %b %Y}<br>Drawdown: $%{y:,.0f}<extra></extra>"))
    fig2.update_layout(title="Drawdown desde el máximo (USD)", height=240,
                       margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified", showlegend=False)
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Todas las estadísticas")
    st.dataframe(pd.DataFrame(r.items(), columns=["Métrica", "Valor"]).astype({"Valor": str}),
                 hide_index=True, width="stretch")
    if saltados:
        st.caption(f"Señales no tomadas: {saltados.get('sin_contratos', 0)} porque el stop costaba más que "
                   f"el riesgo permitido (0 contratos), {saltados.get('stop_muy_lejos', 0)} por stop muy lejano.")

with tabs[1]:
    rv, rb = resumir(vent), resumir(boot)
    a, b = st.columns(2)
    with a:
        st.markdown("**Ventanas históricas** · una evaluación que empieza cada día de la muestra")
        st.metric("Pasan", f"{rv.get('Pasan (%)', 0)}%")
        st.dataframe(pd.DataFrame(rv.items(), columns=["", "Valor"]), hide_index=True, width="stretch")
    with b:
        st.markdown(f"**Bootstrap** · {len(boot):,} trayectorias de 250 días remuestreando días reales")
        st.metric("Pasan", f"{rb.get('Pasan (%)', 0)}%")
        st.dataframe(pd.DataFrame(rb.items(), columns=["", "Valor"]), hide_index=True, width="stretch")
    st.caption(
        f"Reglas: meta ${reglas.meta:,.0f}, pérdida máxima trailing EOD ${reglas.mll:,.0f} "
        f"(se congela en inicial + ${reglas.bloqueo:,.0f}), límite diario "
        f"{'no tiene' if reglas.dll == 0 else f'${reglas.dll:,.0f}'}, consistencia "
        f"{'no tiene' if reglas.consistencia == 0 else f'{reglas.consistencia:.0%}'}. "
        "La pérdida intradía se mide marcando a mercado en los extremos de cada barra de 30 min. "
        "Las ventanas históricas se solapan y no son independientes. Verifica siempre las reglas vigentes de la firma."
    )
    if r["Mejor día / P&L total (%)"] == r["Mejor día / P&L total (%)"] and r["Mejor día / P&L total (%)"] > 50:
        st.warning("Tu mejor día supera el 50% del P&L total: la regla de consistencia te va a frenar.")

with tabs[2]:
    st.dataframe(trades.sort_values(["fecha", "hora_entrada"], ascending=False),
                 hide_index=True, width="stretch")
    st.download_button("Descargar trades (CSV)", trades.to_csv(index=False).encode("utf-8"),
                       "trades_noise_area_mes.csv", "text/csv")
    st.download_button("Descargar P&L diario (CSV)", diario.to_csv(index=False).encode("utf-8"),
                       "diario_noise_area_mes.csv", "text/csv")

with tabs[3]:
    a, b = st.columns(2)
    with a:
        st.markdown("**Por año**")
        st.dataframe(por_periodo(trades, "Y"), width="stretch")
        st.markdown("**Por mes**")
        st.dataframe(por_periodo(trades, "M"), width="stretch")
    with b:
        st.markdown("**Por hora de entrada (NY)**")
        st.dataframe(por_hora_entrada(trades), width="stretch")

with tabs[4]:
    dias_con_trade = sorted(trades["fecha"].dt.date.unique(), reverse=True)
    dia = st.selectbox("Día", dias_con_trade)
    d = bandas[bandas["fecha"] == dia].sort_values("k")
    t = trades[trades["fecha"].dt.date == dia]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=d["hora_cierre"], y=d["close"], name="Cierre 30 min",
                              mode="lines+markers", line=dict(color=AZUL, width=2), marker=dict(size=8)))
    fig3.add_trace(go.Scatter(x=d["hora_cierre"], y=d["vwap"], name="VWAP",
                              mode="lines", line=dict(color=NARANJA, width=2)))
    fig3.add_trace(go.Scatter(x=d["hora_cierre"], y=d["UB"], name="Banda superior (UB)",
                              mode="lines", line=dict(color=GRIS, width=2, dash="dash")))
    fig3.add_trace(go.Scatter(x=d["hora_cierre"], y=d["LB"], name="Banda inferior (LB)",
                              mode="lines", line=dict(color=GRIS, width=2, dash="dot")))
    for _, tr in t.iterrows():
        fig3.add_annotation(x=tr["hora_entrada"], y=tr["precio_entrada"],
                            text=f"{tr['lado']} ×{tr['contratos']}", showarrow=True, arrowhead=2)
        fig3.add_annotation(x=tr["hora_salida"], y=tr["precio_salida"],
                            text=f"Salida {tr['pnl_usd']:+,.0f} USD", showarrow=True, arrowhead=2)
    fig3.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                       legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig3, width="stretch")
    st.dataframe(t, hide_index=True, width="stretch")
