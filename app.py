"""
Radar de Renta Variable — screener fundamental + valoración por múltiplos
+ optimización de portafolio (máximo Sharpe y mínima varianza).
Datos: Yahoo Finance vía yfinance.   Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import motor as M

st.set_page_config(page_title="Radar de Renta Variable", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Universos predefinidos (tickers de Yahoo Finance)
# ---------------------------------------------------------------------------
PRESETS = {
    "Mega caps EE.UU.": "AAPL MSFT NVDA AMZN GOOGL META BRK-B AVGO TSLA JPM V LLY UNH XOM MA JNJ PG HD COST WMT",
    "Tecnología": "AAPL MSFT NVDA AVGO ORCL CRM ADBE AMD INTC QCOM TXN IBM CSCO NOW INTU MU AMAT",
    "Financiero": "JPM BAC WFC GS MS C BLK SCHW AXP USB PNC",
    "Salud": "LLY UNH JNJ MRK ABBV PFE TMO ABT AMGN BMY GILD CVS",
    "Energía": "XOM CVX COP EOG SLB OXY PSX MPC VLO",
    "Consumo": "AMZN TSLA HD MCD NKE SBUX COST WMT PG KO PEP",
    "Colombia y LatAm (ADRs)": "EC CIB AVAL TGLS ITUB BBD PBR VALE NU MELI BAP SQM AMX FMX CX",
}
SCREENERS_YAHOO = {
    "Ninguno": None,
    "Large caps infravaloradas": "undervalued_large_caps",
    "Crecimiento infravalorado": "undervalued_growth_stocks",
    "Tecnología en crecimiento": "growth_technology_stocks",
    "Small caps agresivas": "aggressive_small_caps",
    "Más activas hoy": "most_actives",
    "Mayores alzas hoy": "day_gainers",
}
BENCH = "^GSPC"
MONEDAS_SUB = {"GBp": ("GBP", 100), "ZAc": ("ZAR", 100), "ILA": ("ILS", 100)}

# ---------------------------------------------------------------------------
# Capa de datos (con caché de 1 hora)
# ---------------------------------------------------------------------------

def _info(t: str) -> dict:
    try:
        return yf.Ticker(t).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_fundamentales(tickers: tuple[str, ...]) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=8) as ex:
        infos = list(ex.map(_info, tickers))
    filas = [M.metricas_desde_info(t, i) for t, i in zip(tickers, infos) if i]
    df = pd.DataFrame(filas)
    if len(df):
        df = df[df["Precio"].notna()]
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_precios(tickers: tuple[str, ...], periodo: str) -> pd.DataFrame:
    data = yf.download(list(tickers), period=periodo, auto_adjust=True, progress=False, threads=True)
    if data is None or data.empty:
        return pd.DataFrame()
    close = data["Close"] if "Close" in data else data
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all", axis=1)


@st.cache_data(ttl=3600, show_spinner=False)
def tasa_libre_riesgo() -> float | None:
    p = cargar_precios(("^IRX",), "1mo")
    try:
        return float(p.iloc[:, 0].dropna().iloc[-1]) / 100
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def tipo_cambio(de: str, a: str) -> float:
    """Unidades de `a` por 1 unidad de `de`."""
    if de == a:
        return 1.0
    p = cargar_precios((f"{de}{a}=X",), "5d")
    if p.empty:
        raise ValueError(f"No encontré el tipo de cambio {de}/{a} en Yahoo")
    return float(p.iloc[:, 0].dropna().iloc[-1])


@st.cache_data(ttl=1800, show_spinner=False)
def screener_yahoo(clave: str, n: int) -> list[str]:
    res = yf.screen(clave, count=n)
    return [q["symbol"] for q in res.get("quotes", []) if q.get("symbol")]


def parse_tickers(txt: str) -> list[str]:
    return [t.strip().upper() for t in txt.replace(",", " ").replace(";", " ").split() if t.strip()]


# ---------------------------------------------------------------------------
# Barra lateral: universo y criterios
# ---------------------------------------------------------------------------
st.sidebar.header("1 · Universo de búsqueda")
presets = st.sidebar.multiselect("Listas predefinidas", list(PRESETS), default=["Mega caps EE.UU."])
manual = st.sidebar.text_area("Tickers adicionales (formato Yahoo)", "",
                              placeholder="Ej: KO, PEP, ECOPETROL.CL, SAP.DE",
                              help="Usa el símbolo exacto de Yahoo Finance. Acciones de otras bolsas llevan sufijo (.DE, .L, .SA, .MX…).")
scr_nombre = st.sidebar.selectbox("Screener de Yahoo (opcional)", list(SCREENERS_YAHOO))
scr_n = st.sidebar.slider("Nº de resultados del screener", 10, 50, 25, 5, disabled=scr_nombre == "Ninguno")

st.sidebar.header("2 · Criterios del puntaje")
pesos = {
    "Valor": st.sidebar.slider("Valor (múltiplos)", 0, 5, 3),
    "Calidad": st.sidebar.slider("Calidad (rentabilidad y balance)", 0, 5, 2),
    "Crecimiento": st.sidebar.slider("Crecimiento", 0, 5, 2),
    "Momentum": st.sidebar.slider("Momentum de precio", 0, 5, 1),
}
rel_sector = st.sidebar.toggle("Comparar contra su propio sector", True,
                               help="Si un sector tiene menos de 4 empresas en el universo, se compara contra todo el universo.")
cargar = st.sidebar.button("Cargar / actualizar datos", type="primary", width="stretch")

if cargar:
    tickers = []
    for p in presets:
        tickers += PRESETS[p].split()
    tickers += parse_tickers(manual)
    if SCREENERS_YAHOO[scr_nombre]:
        try:
            tickers += screener_yahoo(SCREENERS_YAHOO[scr_nombre], scr_n)
        except Exception as e:
            st.sidebar.warning(f"El screener de Yahoo no respondió ({e}). Sigo con las listas.")
    tickers = tuple(dict.fromkeys(tickers))
    if not tickers:
        st.sidebar.error("Agrega al menos un ticker.")
    else:
        with st.spinner(f"Descargando fundamentales y precios de {len(tickers)} activos desde Yahoo Finance…"):
            fund = cargar_fundamentales(tickers)
            px1 = cargar_precios(tuple(fund["Ticker"]), "13mo") if len(fund) else pd.DataFrame()
        if fund.empty:
            st.error("Yahoo no devolvió datos. Revisa los tickers o tu conexión e inténtalo de nuevo.")
        else:
            fund = fund.join(M.momentum(px1), on="Ticker")
            st.session_state["fund"] = fund
            st.session_state["fecha"] = datetime.now().strftime("%d/%m/%Y %H:%M")
            faltan = sorted(set(tickers) - set(fund["Ticker"]))
            st.session_state["faltan"] = faltan
            st.session_state.pop("opt", None)

st.title("Radar de Renta Variable")

if "fund" not in st.session_state:
    st.info("Elige un universo en la barra lateral y pulsa **Cargar / actualizar datos**. "
            "La app trae precios y fundamentales de Yahoo Finance, puntúa cada acción por valor, calidad, "
            "crecimiento y momentum, y luego optimiza el portafolio que armes.")
    st.stop()

base = st.session_state["fund"]
df = M.puntajes(base, pesos, rel_sector)
st.caption(f"Fuente: Yahoo Finance · datos de {st.session_state['fecha']} · {len(df)} activos"
           + (f" · sin datos: {', '.join(st.session_state['faltan'])}" if st.session_state["faltan"] else ""))

FMT = {
    "Precio": st.column_config.NumberColumn(format="%.2f"),
    "Cap. mercado": st.column_config.NumberColumn(format="compact"),
    "Puntaje": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
    **{c: st.column_config.NumberColumn(format="%.1f") for c in ["P/E", "P/E fwd", "PEG", "P/B", "P/S", "EV/EBITDA", "EV/Ventas",
                                                                   "Valor", "Calidad", "Crecimiento", "Momentum", "Deuda/Patrimonio",
                                                                   "Razón corriente", "Beta", "Recomendación"]},
    **{c: st.column_config.NumberColumn(format="percent") for c in ["Div. yield", "FCF yield", "ROE", "ROA", "Margen bruto",
                                                                      "Margen operativo", "Margen neto", "Crec. ventas",
                                                                      "Crec. utilidades", "Pos. 52s", "Upside analistas",
                                                                      "Mom. 12-1", "Ret. 6m"]},
}

tab_op, tab_val, tab_port, tab_met = st.tabs(["Oportunidades", "Valoración por múltiplos", "Portafolio óptimo", "Metodología"])

# ---------------------------------------------------------------------------
# Pestaña 1: Oportunidades
# ---------------------------------------------------------------------------
with tab_op:
    top = df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mejor puntaje", f"{top['Ticker']}", f"{top['Puntaje']:.0f}/100")
    c2.metric("Mediana P/E forward", f"{df['P/E fwd'][df['P/E fwd'] > 0].median():.1f}x")
    c3.metric("Mediana EV/EBITDA", f"{df['EV/EBITDA'][df['EV/EBITDA'] > 0].median():.1f}x")
    c4.metric("Upside mediano analistas", f"{df['Upside analistas'].median():.1%}" if df["Upside analistas"].notna().any() else "—")

    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("Mapa de oportunidades")
        mapa = df.assign(Tamaño=df["Cap. mercado"].fillna(df["Cap. mercado"].median()).clip(lower=1))
        fig = px.scatter(mapa, x="Valor", y="Calidad", size="Tamaño", color="Puntaje", text="Ticker",
                         hover_data={"Nombre": True, "Sector": True, "P/E fwd": ":.1f", "ROE": ":.1%", "Tamaño": False},
                         color_continuous_scale="RdYlGn", range_color=[20, 80], size_max=45)
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
        fig.add_vline(x=50, line_dash="dot", line_color="gray")
        fig.add_annotation(x=95, y=97, text="Baratas y de calidad", showarrow=False, font=dict(color="green"))
        fig.update_traces(textposition="top center")
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Percentil de valor (más barato →)", yaxis_title="Percentil de calidad")
        st.plotly_chart(fig, width="stretch")
    with der:
        st.subheader("Top 15 por puntaje")
        t15 = df.head(15).iloc[::-1]
        fig = go.Figure()
        colores = {"Valor": "#1f77b4", "Calidad": "#2ca02c", "Crecimiento": "#ff7f0e", "Momentum": "#9467bd"}
        tot = sum(pesos.values()) or 1
        for b, col in colores.items():
            fig.add_bar(y=t15["Ticker"], x=t15[b] * pesos[b] / tot, name=b, orientation="h", marker_color=col)
        fig.update_layout(barmode="stack", height=480, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=-0.12), xaxis_title="Aporte al puntaje")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Tabla completa")
    sectores = st.multiselect("Filtrar por sector", sorted(df["Sector"].unique()))
    vista = df[df["Sector"].isin(sectores)] if sectores else df
    cols = ["Ticker", "Nombre", "Sector", "Puntaje", "Valor", "Calidad", "Crecimiento", "Momentum", "Precio", "Moneda",
            "Cap. mercado", "P/E", "P/E fwd", "PEG", "P/B", "P/S", "EV/EBITDA", "FCF yield", "Div. yield", "ROE",
            "Margen operativo", "Margen neto", "Crec. ventas", "Crec. utilidades", "Deuda/Patrimonio", "Beta",
            "Mom. 12-1", "Pos. 52s", "Upside analistas", "Recomendación"]
    st.dataframe(vista[cols], hide_index=True, width="stretch", column_config=FMT, height=460)
    st.download_button("Descargar tabla (CSV)", vista[cols].to_csv(index=False).encode("utf-8"),
                       "oportunidades.csv", "text/csv")

# ---------------------------------------------------------------------------
# Pestaña 2: Valoración por múltiplos
# ---------------------------------------------------------------------------
with tab_val:
    sel = st.selectbox("Empresa a valorar", df["Ticker"], format_func=lambda t: f"{t} — {df.set_index('Ticker').loc[t, 'Nombre']}")
    fila = df.set_index("Ticker").loc[sel]
    mismos = df.loc[(df["Sector"] == fila["Sector"]) & (df["Ticker"] != sel), "Ticker"].tolist()
    pares = st.multiselect("Grupo de comparables", [t for t in df["Ticker"] if t != sel],
                           default=mismos or [t for t in df["Ticker"] if t != sel][:8],
                           help="Por defecto: empresas del mismo sector dentro del universo cargado.")
    if not pares:
        st.warning("Elige al menos un comparable.")
    else:
        val = M.valoracion_multiplos(df, sel, pares)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Precio actual", f"{fila['Precio']:,.2f} {fila['Moneda']}")
        if len(val):
            med = val["Precio implícito"].median()
            k2.metric("Precio implícito (mediana)", f"{med:,.2f}", f"{med / fila['Precio'] - 1:+.1%}")
        if pd.notna(fila["Upside analistas"]):
            k3.metric("Objetivo analistas", f"{fila['Precio'] * (1 + fila['Upside analistas']):,.2f}", f"{fila['Upside analistas']:+.1%}")
        k4.metric("Puntaje oportunidad", f"{fila['Puntaje']:.0f}/100")

        a, b = st.columns([3, 2])
        with a:
            if len(val):
                fig = go.Figure()
                fig.add_bar(y=val["Método"], x=val["Precio implícito"], orientation="h",
                            marker_color=np.where(val["Upside"] >= 0, "#2ca02c", "#d62728"),
                            text=[f"{v:,.2f} ({u:+.0%})" for v, u in zip(val["Precio implícito"], val["Upside"])],
                            textposition="outside")
                fig.add_vline(x=fila["Precio"], line_color="black", line_width=2,
                              annotation_text="Precio actual", annotation_position="top")
                fig.update_layout(height=380, margin=dict(l=0, r=40, t=30, b=0), xaxis_title=f"Precio por acción ({fila['Moneda']})")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No hay múltiplos positivos suficientes para valorar (p. ej. utilidades negativas).")
        with b:
            st.dataframe(val, hide_index=True, width="stretch", column_config={
                "Propio": st.column_config.NumberColumn(format="%.1fx"),
                "Mediana pares": st.column_config.NumberColumn(format="%.1fx"),
                "Precio implícito": st.column_config.NumberColumn(format="%.2f"),
                "Upside": st.column_config.NumberColumn(format="percent")})

        st.subheader("Fundamentales frente a los comparables")
        grupo = df[df["Ticker"].isin(pares)]
        comp_cols = ["P/E", "P/E fwd", "PEG", "P/B", "P/S", "EV/EBITDA", "FCF yield", "Div. yield", "ROE", "ROA",
                     "Margen bruto", "Margen operativo", "Margen neto", "Crec. ventas", "Crec. utilidades",
                     "Deuda/Patrimonio", "Razón corriente", "Beta"]
        pct = {"FCF yield", "Div. yield", "ROE", "ROA", "Margen bruto", "Margen operativo", "Margen neto", "Crec. ventas", "Crec. utilidades"}
        fmt = lambda c, v: "—" if pd.isna(v) else (f"{v:.1%}" if c in pct else f"{v:.2f}")
        comp = pd.DataFrame({
            "Métrica": comp_cols,
            sel: [fmt(c, fila[c]) for c in comp_cols],
            "Mediana pares": [fmt(c, grupo[c].median()) for c in comp_cols],
            "Percentil vs. pares": [f"{(grupo[c] < fila[c]).mean():.0%}" if pd.notna(fila[c]) and grupo[c].notna().any() else "—" for c in comp_cols],
        })
        st.dataframe(comp, hide_index=True, width="stretch")
        st.caption("Precio implícito = precio actual × (múltiplo mediano de pares ÷ múltiplo propio). En múltiplos de EV se descuenta la deuda neta. "
                   "Es valoración relativa: si todo el sector está caro, el precio implícito también lo estará.")

# ---------------------------------------------------------------------------
# Pestaña 3: Portafolio óptimo
# ---------------------------------------------------------------------------
with tab_port:
    with st.form("form_port"):
        activos = st.multiselect("Activos del portafolio (ordenados por puntaje)", df["Ticker"].tolist(),
                                 default=df["Ticker"].head(min(8, len(df))).tolist())
        f1, f2, f3, f4 = st.columns(4)
        moneda = f1.selectbox("Moneda de inversión", ["USD", "COP", "EUR"])
        monto = f2.number_input("Monto a invertir (moneda elegida)", min_value=0.0, value=10_000.0 if moneda != "COP" else 40_000_000.0, step=1000.0)
        periodo = f3.selectbox("Historia para estimar", ["1y", "2y", "3y", "5y"], index=2,
                               format_func=lambda p: p.replace("y", " años").replace("1 años", "1 año"))
        rf_live = tasa_libre_riesgo()
        rf = f4.number_input("Tasa libre de riesgo anual (%)", value=round((rf_live or 0.04) * 100, 2), step=0.1,
                             help="Por defecto: rendimiento actual del T-Bill a 13 semanas (^IRX) en Yahoo.") / 100
        g1, g2, g3, g4 = st.columns(4)
        metodo_mu = g1.selectbox("Retornos esperados", ["Histórica", "CAPM"],
                                 help="Histórica: media anualizada. CAPM: rf + β·prima de mercado (β contra el S&P 500).")
        prima = g2.number_input("Prima de mercado (%) — CAPM", value=5.5, step=0.25) / 100
        metodo_cov = g3.selectbox("Covarianza", ["Ledoit-Wolf", "Muestral"])
        cap = g4.slider("Peso máximo por activo", 0.05, 1.0, 0.35, 0.05)
        fracc = st.checkbox("Permitir fracciones de acción", False)
        enviar = st.form_submit_button("Optimizar portafolio", type="primary")

    if enviar:
        if len(activos) < 2:
            st.error("Selecciona al menos 2 activos.")
        elif cap * len(activos) < 1:
            st.error(f"Con {len(activos)} activos el peso máximo debe ser al menos {1 / len(activos):.0%}.")
        else:
            with st.spinner("Descargando historia de precios y optimizando…"):
                try:
                    hist = cargar_precios(tuple(activos) + (BENCH,), periodo)
                    usables = [a for a in activos if a in hist and hist[a].notna().sum() > 120]
                    if len(usables) < 2:
                        raise ValueError("No hay suficiente historia de precios para estos activos.")
                    ret_all = M.retornos_diarios(hist[usables + ([BENCH] if BENCH in hist else [])])
                    ret = ret_all[usables]
                    mkt = ret_all[BENCH] if BENCH in ret_all else None
                    mu, cov, delta, betas = M.estimar_parametros(ret, metodo_cov, metodo_mu, rf, mkt, prima)
                    cap_ef = max(cap, 1 / len(usables))
                    w_ms = pd.Series(M.max_sharpe(mu, cov, rf, cap_ef), mu.index)
                    w_mv = pd.Series(M.min_varianza(mu, cov, cap_ef), mu.index)
                    info = df.set_index("Ticker").loc[usables]
                    precios_act, fx = [], []
                    for t in usables:
                        cur, p = info.loc[t, "Moneda"], info.loc[t, "Precio"]
                        if cur in MONEDAS_SUB:
                            cur, div = MONEDAS_SUB[cur]
                            p = p / div
                        precios_act.append(p)
                        fx.append(tipo_cambio(cur, moneda))
                    st.session_state["opt"] = dict(
                        mu=mu, cov=cov, rf=rf, delta=delta, betas=betas, ret=ret, mkt=mkt, cap=cap_ef,
                        w={"Máximo Sharpe": w_ms, "Mínima varianza": w_mv}, moneda=moneda, monto=monto,
                        precios=pd.Series(precios_act, usables), fx=pd.Series(fx, usables), fracc=fracc,
                        excluidos=sorted(set(activos) - set(usables)), metodo_cov=metodo_cov, metodo_mu=metodo_mu)
                except Exception as e:
                    st.error(f"No se pudo optimizar: {e}")

    o = st.session_state.get("opt")
    if o:
        mu, cov, rf = o["mu"], o["cov"], o["rf"]
        if o["excluidos"]:
            st.warning(f"Excluidos por falta de historia: {', '.join(o['excluidos'])}")
        if (mu < rf).all():
            st.warning("Todos los activos tienen retorno esperado menor que la tasa libre de riesgo: el 'máximo Sharpe' será el menos malo.")

        cols = st.columns(2)
        for col, (nombre, w) in zip(cols, o["w"].items()):
            r, v, s = M._stats(w.values, mu.values, cov.values, rf)
            with col:
                st.subheader(nombre)
                m1, m2, m3 = st.columns(3)
                m1.metric("Retorno esperado", f"{r:.1%}")
                m2.metric("Volatilidad", f"{v:.1%}")
                m3.metric("Sharpe", f"{s:.2f}")
                asig = M.asignar_monto(w, o["precios"], o["fx"], o["monto"], o["fracc"])
                asig = asig[asig["Peso objetivo"] > 0].sort_values("Peso objetivo", ascending=False)
                st.dataframe(asig, width="stretch", column_config={
                    "Peso objetivo": st.column_config.NumberColumn(format="percent"),
                    "Peso real": st.column_config.NumberColumn(format="percent"),
                    "Monto objetivo": st.column_config.NumberColumn(format="localized"),
                    "Monto invertido": st.column_config.NumberColumn(format="localized"),
                    "Precio (moneda inversión)": st.column_config.NumberColumn(format="localized"),
                    "Acciones": st.column_config.NumberColumn(format="%.4f" if o["fracc"] else "%d")})
                sobrante = o["monto"] - asig["Monto invertido"].sum()
                st.caption(f"Efectivo sin invertir por redondeo: {sobrante:,.0f} {o['moneda']}")
                fsel = df.set_index("Ticker").loc[w.index]
                wpos = w[w > 0]
                pe = fsel.loc[wpos.index, "P/E fwd"]
                ok = pe > 0
                pe_port = wpos[ok].sum() / (wpos[ok] / pe[ok]).sum() if ok.any() else np.nan
                st.caption("Fundamentales ponderados: "
                           f"P/E fwd {pe_port:.1f}x · ROE {np.nansum(wpos * fsel.loc[wpos.index, 'ROE']):.1%} · "
                           f"Div. yield {np.nansum(wpos * fsel.loc[wpos.index, 'Div. yield']):.2%} · "
                           f"Puntaje {np.nansum(wpos * fsel.loc[wpos.index, 'Puntaje']):.0f}/100"
                           + (f" · β {float(wpos @ o['betas'][wpos.index]):.2f}" if o["betas"] is not None else ""))

        st.subheader("Frontera eficiente")
        fr = M.frontera(mu, cov, o["cap"])
        rnd = M.portafolios_aleatorios(mu, cov, rf)
        fig = go.Figure()
        fig.add_scatter(x=rnd["Volatilidad"], y=rnd["Retorno"], mode="markers", name="Portafolios aleatorios",
                        marker=dict(size=4, color=rnd["Sharpe"], colorscale="Viridis", opacity=0.45,
                                    colorbar=dict(title="Sharpe", x=1.02)), hoverinfo="skip")
        fig.add_scatter(x=fr["Volatilidad"], y=fr["Retorno"], mode="lines", name="Frontera eficiente", line=dict(color="black", width=3))
        vols = np.sqrt(np.diag(cov.values))
        fig.add_scatter(x=vols, y=mu.values, mode="markers+text", text=mu.index, textposition="top center",
                        name="Activos individuales", marker=dict(size=9, color="#7f7f7f", symbol="diamond"))
        estilos = {"Máximo Sharpe": ("#d62728", "star"), "Mínima varianza": ("#1f77b4", "circle")}
        for nombre, w in o["w"].items():
            r, v, s = M._stats(w.values, mu.values, cov.values, rf)
            c, sym = estilos[nombre]
            fig.add_scatter(x=[v], y=[r], mode="markers", name=f"{nombre} (S={s:.2f})", marker=dict(size=18, color=c, symbol=sym,
                                                                                               line=dict(color="white", width=1)))
            if nombre == "Máximo Sharpe":
                xmax = max(vols.max(), v) * 1.1
                fig.add_scatter(x=[0, xmax], y=[rf, rf + s * xmax], mode="lines", name="Línea del mercado de capitales",
                                line=dict(color="#d62728", dash="dash", width=1.5))
        fig.update_layout(height=520, xaxis_title="Volatilidad anual", yaxis_title="Retorno esperado anual",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.15),
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        with a:
            st.subheader("Pesos comparados")
            wdf = pd.DataFrame(o["w"]).reset_index(names="Ticker").melt("Ticker", var_name="Portafolio", value_name="Peso")
            fig = px.bar(wdf, x="Ticker", y="Peso", color="Portafolio", barmode="group",
                         color_discrete_map={"Máximo Sharpe": "#d62728", "Mínima varianza": "#1f77b4"})
            fig.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
        with b:
            st.subheader("Matriz de correlación")
            corr = o["ret"].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Desempeño histórico (dentro de muestra)")
        cum = pd.DataFrame({n: (1 + o["ret"] @ w).cumprod() for n, w in o["w"].items()})
        cum["Igual ponderado"] = (1 + o["ret"].mean(axis=1)).cumprod()
        if o["mkt"] is not None:
            cum["S&P 500"] = (1 + o["mkt"]).cumprod()
        fig = px.line(cum * 100, color_discrete_map={"Máximo Sharpe": "#d62728", "Mínima varianza": "#1f77b4",
                                                    "Igual ponderado": "#7f7f7f", "S&P 500": "#2ca02c"})
        fig.update_layout(height=380, yaxis_title="Valor de 100 invertidos", xaxis_title="", legend_title="",
                          margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Covarianza {o['metodo_cov']}" + (f" (encogimiento δ = {o['delta']:.2f})" if o["metodo_cov"] == "Ledoit-Wolf" else "")
                   + f" · retornos esperados {o['metodo_mu']} · tope por activo {o['cap']:.0%} · rebalanceo diario. "
                   "Este backtest usa los mismos datos con los que se optimizó, así que sobreestima el desempeño futuro.")
    else:
        st.info("Elige los activos, el monto y pulsa **Optimizar portafolio**.")

# ---------------------------------------------------------------------------
# Pestaña 4: Metodología
# ---------------------------------------------------------------------------
with tab_met:
    st.subheader("Puntaje de oportunidad")
    st.markdown("Cada métrica se convierte en percentil $p \\in [0,1]$ dentro del sector (o del universo). "
                "Los múltiplos se invierten a rendimientos para que utilidades negativas queden al fondo:")
    st.latex(r"\text{E/P}=\frac{1}{\text{P/E}},\qquad \text{EBITDA/EV}=\frac{1}{\text{EV/EBITDA}},\qquad \text{FCF yield}=\frac{\text{FCF}}{\text{Cap. mercado}}")
    st.latex(r"\text{Puntaje}=100\cdot\frac{\sum_{b} \omega_b \,\bar p_b}{\sum_b \omega_b},\qquad b\in\{\text{Valor},\text{Calidad},\text{Crecimiento},\text{Momentum}\}")
    st.markdown("- **Valor:** E/P forward, EBITDA/EV, B/P, S/P, FCF yield.\n"
                "- **Calidad:** ROE, margen operativo, margen neto, deuda/patrimonio (menor es mejor), razón corriente.\n"
                "- **Crecimiento:** crecimiento de ventas y de utilidades (último trimestre interanual, Yahoo).\n"
                "- **Momentum:** retorno de 12 meses excluyendo el último mes, y retorno de 6 meses.")
    st.subheader("Valoración por múltiplos")
    st.latex(r"P^{\text{impl}}=P\cdot\frac{\tilde m_{\text{pares}}}{m_{\text{propio}}}\quad\text{para } m\in\{\text{P/E},\ \text{P/E fwd},\ \text{P/B},\ \text{P/S}\}")
    st.latex(r"P^{\text{impl}}=P\cdot\frac{EV\cdot\tilde m_{\text{pares}}/m_{\text{propio}}-(EV-\text{Cap})}{\text{Cap}}\quad\text{para } m\in\{\text{EV/EBITDA},\ \text{EV/Ventas}\}")
    st.subheader("Optimización media-varianza")
    st.latex(r"\mu=252\,\bar r \quad\text{o}\quad \mu_i=r_f+\beta_i\,(E[R_m]-r_f),\qquad \Sigma=252\,\big(\delta F+(1-\delta)S\big)")
    st.latex(r"\textbf{Máximo Sharpe:}\ \max_{w}\ \frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}}\quad \text{s.a.}\ \mathbf 1^\top w=1,\ 0\le w_i\le \bar w")
    st.latex(r"\textbf{Mínima varianza:}\ \min_{w}\ w^\top\Sigma w\quad \text{s.a.}\ \mathbf 1^\top w=1,\ 0\le w_i\le \bar w")
    st.markdown("Se resuelve con SLSQP (SciPy), sin ventas en corto. Ledoit-Wolf encoge la covarianza muestral $S$ hacia "
                "$F=\\tfrac{\\operatorname{tr}(S)}{N}I$, lo que estabiliza los pesos cuando hay pocos datos frente al número de activos. "
                "El P/E del portafolio es la media armónica ponderada de los P/E forward positivos.")
    st.subheader("Limitaciones")
    st.markdown("- Los datos de Yahoo pueden tener vacíos o retrasos (en especial ADRs y bolsas pequeñas). Verifica las cifras clave antes de decidir.\n"
                "- Los retornos históricos son un estimador ruidoso del futuro; el portafolio de máximo Sharpe es muy sensible a ellos. "
                "Prueba CAPM, un tope de peso más bajo o distintos periodos.\n"
                "- Herramienta educativa y de análisis: no es una recomendación de inversión.")
