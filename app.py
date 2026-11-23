"""
Radar — encuentra oportunidades, entiende cada empresa y arma tu portafolio.
Acciones de EE.UU., Colombia (BVC) y LatAm + criptomonedas. Datos: Yahoo Finance (yfinance).
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import html
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf

import motor as M

st.set_page_config(page_title="Radar · Inversión inteligente", page_icon="📡", layout="wide")

# ---------------------------------------------------------------------------
# Identidad visual: azul noche + cian neón + grises (violeta solo en el título; verde = positivo)
# ---------------------------------------------------------------------------
CIAN, VERDE, VIOLETA = "#00E5FF", "#39FF88", "#AE65B6"
GRIS1, GRIS2, GRIS3, GRIS4 = "#C9D1E0", "#9AA3B5", "#5B6478", "#2A3142"
NAVY, PANEL, TEXTO, MUTED, LINEA = "#060B1E", "#0D1735", "#E6EEFF", "#8A97B8", "rgba(0,229,255,0.10)"
NARANJA, AZUL, BEIGE = CIAN, GRIS1, TEXTO  # cian = máximo Sharpe, gris claro = mínima varianza

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
.stApp { background:
  radial-gradient(900px 480px at 105% -10%, rgba(154,163,181,0.12), transparent 60%),
  radial-gradient(800px 480px at -10% 5%, rgba(0,229,255,0.14), transparent 60%),
  radial-gradient(700px 400px at 50% 110%, rgba(57,255,136,0.08), transparent 60%),
  linear-gradient(rgba(0,229,255,0.035) 1px, transparent 1px) 0 0 / 44px 44px,
  linear-gradient(90deg, rgba(0,229,255,0.035) 1px, transparent 1px) 0 0 / 44px 44px, #060B1E; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
.hero { border: 1px solid rgba(0,229,255,0.25); border-radius: 22px; padding: 28px 32px; position: relative; overflow: hidden;
  background: linear-gradient(135deg, rgba(13,23,53,0.92), rgba(6,11,30,0.88));
  box-shadow: 0 0 0 1px rgba(154,163,181,0.10), 0 0 40px rgba(0,229,255,0.10), inset 0 0 60px rgba(0,229,255,0.04); margin-bottom: 8px; }
.hero:before { content:""; position:absolute; left:0; right:0; top:0; height:2px;
  background: linear-gradient(90deg, #00E5FF, #9AA3B5 60%, #2A3142); box-shadow: 0 0 14px rgba(0,229,255,0.7); }
.hero:after { content:""; position:absolute; right:-70px; top:-70px; width:260px; height:260px; border-radius:50%;
  border: 2px solid rgba(0,229,255,0.55); box-shadow: 0 0 22px rgba(0,229,255,0.40), 0 0 0 28px rgba(154,163,181,0.06),
  0 0 0 29px rgba(154,163,181,0.35), 0 0 0 66px rgba(57,255,136,0.04), 0 0 0 67px rgba(57,255,136,0.25); }
.titulo { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size: clamp(32px, 4.4vw, 52px); line-height:1.05;
  color:#E6EEFF; margin:0 0 10px; letter-spacing:-0.02em; }
.titulo span { color: #0A2A6E; -webkit-text-stroke: 1.6px #00E5FF; paint-order: stroke fill;
  text-shadow: 0 0 14px rgba(0,229,255,0.55), 0 0 30px rgba(0,229,255,0.25); }
.hero p { color:#B9C6E4; font-size: 17px; max-width: 760px; margin: 0; }
.card { background: rgba(13,23,53,0.88); border: 1px solid rgba(0,229,255,0.16); border-radius: 18px; padding: 18px 20px; height: 100%;
  transition: border-color .2s, box-shadow .2s; }
.card:hover { border-color: rgba(0,229,255,0.55); box-shadow: 0 0 24px rgba(0,229,255,0.16); }
.card h4 { margin: 0 0 4px; font-size: 20px; color:#E6EEFF; }
.card .sub { color:#8A97B8; font-size: 13px; margin-bottom: 10px; }
.card .big { font-family:'Space Grotesk',sans-serif; font-size: 30px; font-weight:700; color:#E6EEFF; }
.card ul { margin: 6px 0 0; padding-left: 18px; color:#C9D4EE; font-size: 14px; }
.card.sharpe { border-color: rgba(0,229,255,0.45); box-shadow: 0 0 26px rgba(0,229,255,0.12); }
.card.minvar { border-color: rgba(201,209,224,0.40); box-shadow: 0 0 26px rgba(201,209,224,0.08); }
.badge { display:inline-block; padding: 3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
.b-alta { background: rgba(57,255,136,0.12); color:#39FF88; border:1px solid rgba(57,255,136,0.6); box-shadow: 0 0 10px rgba(57,255,136,0.25); }
.b-media { background: rgba(0,229,255,0.10); color:#00E5FF; border:1px solid rgba(0,229,255,0.5); }
.b-baja { background: rgba(154,163,181,0.12); color:#9AA3B5; border:1px solid rgba(154,163,181,0.45); }
.explica { border-left: 3px solid #00E5FF; background: rgba(0,229,255,0.06); padding: 14px 18px;
  border-radius: 0 14px 14px 0; color:#D3DCF2; font-size: 15px; margin: 6px 0 14px; }
.explica.azul { border-left-color:#9AA3B5; background: rgba(154,163,181,0.08); }
.explica b { color:#FFFFFF; }
.barra { height:6px; border-radius:6px; background: rgba(230,238,255,0.08); margin: 3px 0 8px; overflow:hidden; }
.barra > div { height:100%; border-radius:6px; background: linear-gradient(90deg, #00E5FF, #39FF88); box-shadow: 0 0 8px rgba(57,255,136,0.6); }
.fila-b { display:flex; justify-content:space-between; font-size:13px; color:#B9C6E4; }
div[data-testid="stMetric"] { background: rgba(13,23,53,0.85); border:1px solid rgba(0,229,255,0.18); border-radius: 16px; padding: 14px 16px; }
div[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; color:#FFFFFF; text-shadow: 0 0 14px rgba(0,229,255,0.35); }
button[data-baseweb="tab"][aria-selected="true"] { color: #00E5FF; text-shadow: 0 0 10px rgba(0,229,255,0.7); }
div[data-baseweb="tab-highlight"] { background: #00E5FF !important; box-shadow: 0 0 10px #00E5FF; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(90deg, #00E5FF, #2FA8FF); border: 0; color: #060B1E; font-weight: 700; border-radius: 12px;
  box-shadow: 0 0 18px rgba(0,229,255,0.45); }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { box-shadow: 0 0 28px rgba(0,229,255,0.6); color:#060B1E; }
div[data-testid="stButtonGroup"] > div { flex-wrap: wrap !important; overflow: visible !important; row-gap: 8px; }
.pie { color:#8A97B8; font-size:13px; margin-top: 24px; }
/* ---- Colores forzados: la app se ve igual aunque falte .streamlit/config.toml ---- */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp summary, .stApp small,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #E6EEFF; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: #8A97B8; }
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricLabel"] * { color: #B9C6E4; }
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricValue"] * { color: #FFFFFF; }
.stApp .titulo span { color: #0A2A6E; -webkit-text-stroke: 1.6px #00E5FF; }
.stApp .fila-b span { color: #B9C6E4; }
.stApp .badge.b-alta { color: #39FF88; } .stApp .badge.b-media { color: #00E5FF; } .stApp .badge.b-baja { color: #9AA3B5; }
.stApp button[data-baseweb="tab"] p { color: #8A97B8; }
.stApp button[data-baseweb="tab"][aria-selected="true"] p { color: #00E5FF; }
div[data-baseweb="tab-border"] { background-color: rgba(0,229,255,0.15) !important; }
/* pills y control segmentado */
.stApp button[data-testid="stBaseButton-pills"], .stApp button[data-testid="stBaseButton-segmented_control"] {
  background: rgba(13,23,53,0.9) !important; border: 1px solid rgba(0,229,255,0.28) !important; }
.stApp button[data-testid="stBaseButton-pills"] *, .stApp button[data-testid="stBaseButton-segmented_control"] * { color: #E6EEFF !important; }
.stApp button[data-testid="stBaseButton-pillsActive"], .stApp button[data-testid="stBaseButton-segmented_controlActive"] {
  background: rgba(0,229,255,0.14) !important; border: 1px solid #00E5FF !important; box-shadow: 0 0 12px rgba(0,229,255,0.35); }
.stApp button[data-testid="stBaseButton-pillsActive"] *, .stApp button[data-testid="stBaseButton-segmented_controlActive"] * { color: #00E5FF !important; }
/* radio */
.stApp [data-testid="stRadioOption"] > div > div:first-child { background-color: #0D1735 !important; border: 2px solid rgba(0,229,255,0.6) !important; }
.stApp [data-testid="stRadioOption"] > div > div:first-child > div { background-color: transparent !important; }
.stApp [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child { border-color: #00E5FF !important; box-shadow: 0 0 10px rgba(0,229,255,0.6); }
.stApp [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div { background-color: #00E5FF !important; }
/* campos de texto, número y listas */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="select"] > div, .stApp textarea, .stApp input {
  background-color: #0D1735 !important; color: #E6EEFF !important; border-color: rgba(0,229,255,0.28) !important; }
.stApp input::placeholder, .stApp textarea::placeholder { color: #6F7DA0 !important; }
.stApp [data-baseweb="select"] svg, .stApp [data-testid="stNumberInput"] button svg { fill: #E6EEFF !important; color: #E6EEFF !important; }
.stApp [data-testid="stNumberInput"] button { background-color: #13204A !important; }
.stApp [data-baseweb="tag"] { background-color: rgba(154,163,181,0.16) !important; border: 1px solid rgba(154,163,181,0.5) !important; }
.stApp [data-baseweb="tag"] * { color: #E6EEFF !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"] { background-color: #0D1735 !important; color: #E6EEFF !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background-color: rgba(0,229,255,0.14) !important; }
div[data-baseweb="popover"] * { color: #E6EEFF; }
/* expanders, formularios y botones secundarios */
.stApp [data-testid="stExpander"] details { background: rgba(13,23,53,0.7); border: 1px solid rgba(0,229,255,0.22) !important; border-radius: 14px; }
.stApp [data-testid="stExpander"] summary:hover * { color: #00E5FF; }
.stApp [data-testid="stForm"] { border: 1px solid rgba(0,229,255,0.22); background: rgba(13,23,53,0.45); border-radius: 18px; }
.stApp button[data-testid="stBaseButton-secondary"] { background: transparent !important; border: 1px solid rgba(0,229,255,0.4) !important; }
.stApp button[data-testid="stBaseButton-secondary"] * { color: #E6EEFF !important; }
.stApp button[data-testid^="stBaseButton-primary"] { background: linear-gradient(90deg, #00E5FF, #2FA8FF) !important; border: 0 !important;
  box-shadow: 0 0 18px rgba(0,229,255,0.45); }
.stApp button[data-testid^="stBaseButton-primary"] * { color: #060B1E !important; font-weight: 700; }
.stApp [data-testid="stSlider"] [role="slider"] { background-color: #00E5FF !important; box-shadow: 0 0 10px rgba(0,229,255,0.7); }
.stApp [data-testid="stTooltipIcon"] svg { stroke: #8A97B8 !important; }
.stApp [data-testid="stAlert"] * { color: #E6EEFF; }
.stApp [data-testid="stCheckbox"] label span:first-child { border-color: #00E5FF !important; }
/* Streamlit 1.5x+ (react-aria): pills, control segmentado, multiselect y selectbox */
.stApp button[data-variant="pills"], .stApp button[data-variant="segmented_control"] {
  background: rgba(13,23,53,0.9) !important; border: 1px solid rgba(0,229,255,0.28) !important; color: #E6EEFF !important; }
.stApp button[data-variant="pills"] *, .stApp button[data-variant="segmented_control"] * { color: #E6EEFF !important; }
.stApp button[data-variant="pills"][aria-pressed="true"], .stApp button[data-variant="segmented_control"][aria-checked="true"] {
  background: rgba(0,229,255,0.14) !important; border: 1px solid #00E5FF !important; box-shadow: 0 0 12px rgba(0,229,255,0.35); }
.stApp button[data-variant="pills"][aria-pressed="true"] *, .stApp button[data-variant="segmented_control"][aria-checked="true"] * { color: #00E5FF !important; }
.stApp [data-testid="stMultiSelect"] [role="group"], .stApp [data-testid="stSelectbox"] [role="group"],
.stApp [data-testid="stSelectbox"] button, .stApp [data-testid="stMultiSelect"] input {
  background-color: #0D1735 !important; border-color: rgba(0,229,255,0.28) !important; color: #E6EEFF !important; }
.stApp [data-testid="stMultiSelectTagsContainer"] [role="group"] { background: transparent !important; }
.stApp [data-testid="stMultiSelectTagsContainer"] > span > span { background-color: rgba(154,163,181,0.16) !important;
  border: 1px solid rgba(154,163,181,0.5) !important; }
.stApp [data-testid="stMultiSelectTagsContainer"] > span > span * { color: #E6EEFF !important; }
[role="listbox"], [role="option"] { background-color: #0D1735 !important; color: #E6EEFF !important; }
[role="option"]:hover, [role="option"][aria-selected="true"], [role="option"][data-focused="true"] { background-color: rgba(0,229,255,0.16) !important; }
[role="listbox"] * { color: #E6EEFF !important; }
.stApp [data-testid="stTabs"] .react-aria-SelectionIndicator { background: #00E5FF !important;
  border-color: transparent !important; box-shadow: 0 0 10px rgba(0,229,255,0.8); }
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] *, .stApp [data-testid="stTabs"] [role="tab"][data-selected="true"] * { color: #00E5FF !important; }
.stApp [data-testid="stMetricDelta"] { background: rgba(57,255,136,0.10) !important; }
.stApp [data-testid="stMetricDelta"] * { color: #39FF88 !important; fill: #39FF88 !important; }
/* navegación principal: pestañas grandes tipo botón */
.stApp [data-testid="stTabs"] [role="tablist"] { gap: 12px; flex-wrap: wrap; border: 0 !important; padding: 6px 0 14px; }
.stApp [data-testid="stTabs"] [role="tab"] { flex: 1 1 200px; min-height: 64px; padding: 14px 22px !important; justify-content: flex-start;
  background: rgba(13,23,53,0.85) !important; border: 1px solid rgba(154,163,181,0.30) !important; border-radius: 16px !important;
  transition: border-color .2s, box-shadow .2s, background .2s; }
.stApp [data-testid="stTabs"] [role="tab"] p { font-family: 'Space Grotesk', sans-serif !important; font-size: 20px !important;
  font-weight: 700 !important; color: #C9D1E0 !important; }
.stApp [data-testid="stTabs"] [role="tab"]:hover { border-color: rgba(0,229,255,0.6) !important; }
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"], .stApp [data-testid="stTabs"] [role="tab"][data-selected="true"] {
  background: rgba(0,229,255,0.12) !important; border-color: #00E5FF !important; box-shadow: 0 0 22px rgba(0,229,255,0.28); }
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] p, .stApp [data-testid="stTabs"] [role="tab"][data-selected="true"] p { color: #00E5FF !important; }
.stApp [data-testid="stTabs"] .react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
/* encuesta y tarjeta de perfil */
.encuesta, .perfil-card { border: 1px solid rgba(0,229,255,0.28); border-radius: 20px; padding: 22px 26px; margin: 6px 0 14px;
  background: linear-gradient(135deg, rgba(13,23,53,0.95), rgba(26,34,56,0.85)); }
.kicker { color: #9AA3B5 !important; font-size: 13px; font-weight: 600; letter-spacing: .02em; }
.encuesta h3 { margin: 4px 0 6px; font-size: 26px; }
.encuesta p, .perfil-card p { color: #C9D1E0 !important; margin: 0; max-width: 820px; }
.perfil-tipo { font-family: 'Space Grotesk', sans-serif; font-size: 34px; font-weight: 700; color: #00E5FF;
  text-shadow: 0 0 16px rgba(0,229,255,0.45); margin: 2px 0 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 8px; }
.stApp .chips span { background: rgba(154,163,181,0.12); border: 1px solid rgba(154,163,181,0.35); border-radius: 999px;
  padding: 5px 12px; font-size: 13px; color: #C9D1E0; }
.stApp .chips span b { color: #FFFFFF; }
.perfil-card .nota { color: #8A97B8 !important; font-size: 13px; }
.stApp [data-testid="stForm"] [data-testid="stRadio"] { background: rgba(42,49,66,0.35); border: 1px solid rgba(154,163,181,0.18);
  border-radius: 14px; padding: 12px 14px; margin-bottom: 6px; width: 100%; box-sizing: border-box; }
@media (prefers-reduced-motion: reduce) { .card { transition: none; } }
</style>
""", unsafe_allow_html=True)

pio.templates["radar"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=BEIGE, size=13),
    colorway=[CIAN, GRIS1, VERDE, GRIS2, VIOLETA, GRIS3],
    xaxis=dict(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA, automargin=True),
    yaxis=dict(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA, automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(bgcolor=PANEL, font_color=BEIGE),
    margin=dict(l=10, r=10, t=30, b=10)))
pio.templates.default = "radar"
ESCALA = [[0, GRIS3], [0.5, CIAN], [1, VERDE]]  # bajo → alto

# ---------------------------------------------------------------------------
# Universos (símbolos de Yahoo Finance)
# ---------------------------------------------------------------------------
GRUPOS = {
    "🇨🇴 Bolsa de Colombia": "CIBEST.CL PFCIBEST.CL ECOPETROL.CL GEB.CL CELSIA.CL MINEROS.CL ISA.CL GRUPOSURA.CL "
                            "GRUPOARGOS.CL CEMARGOS.CL PFAVAL.CL BOGOTA.CL PROMIGAS.CL CORFICOLCF.CL PFDAVVNDA.CL TERPEL.CL BVC.CL",
    "₿ Criptomonedas": "BTC-USD ETH-USD SOL-USD XRP-USD",
    "🇺🇸 Gigantes de EE.UU.": "AAPL MSFT NVDA AMZN GOOGL META BRK-B AVGO TSLA JPM V LLY UNH XOM MA JNJ PG HD COST WMT",
    "💻 Tecnología": "AAPL MSFT NVDA AVGO ORCL CRM ADBE AMD INTC QCOM TXN IBM CSCO NOW INTU MU AMAT",
    "🏦 Bancos y finanzas": "JPM BAC WFC GS MS C BLK SCHW AXP USB PNC",
    "💊 Salud": "LLY UNH JNJ MRK ABBV PFE TMO ABT AMGN BMY GILD CVS",
    "⛽ Energía": "XOM CVX COP EOG SLB OXY PSX MPC VLO",
    "🛒 Consumo": "AMZN TSLA HD MCD NKE SBUX COST WMT PG KO PEP",
    "🌎 ADRs de Latinoamérica": "EC CIB AVAL TGLS ITUB BBD PBR VALE NU MELI BAP SQM AMX FMX CX",
}
# Si Yahoo no tiene datos de la acción local, se usa otro listado de la misma empresa
RESPALDO = {"MINEROS.CL": "MSA.TO", "CIBEST.CL": "CIB", "ECOPETROL.CL": "EC", "PFAVAL.CL": "AVAL"}
SCREENERS = {
    "Large caps infravaloradas": "undervalued_large_caps",
    "Crecimiento a buen precio": "undervalued_growth_stocks",
    "Tecnología en crecimiento": "growth_technology_stocks",
    "Las más negociadas hoy": "most_actives",
    "Las que más suben hoy": "day_gainers",
}
ESTILOS = {
    "⚖️ Equilibrado": {"Valor": 2, "Calidad": 2, "Crecimiento": 2, "Momentum": 1},
    "🏷️ Buscar gangas": {"Valor": 4, "Calidad": 2, "Crecimiento": 1, "Momentum": 0},
    "🏛️ Negocios sólidos": {"Valor": 1, "Calidad": 4, "Crecimiento": 1, "Momentum": 0},
    "🚀 Crecimiento": {"Valor": 1, "Calidad": 1, "Crecimiento": 4, "Momentum": 1},
    "📈 Seguir la tendencia": {"Valor": 1, "Calidad": 1, "Crecimiento": 1, "Momentum": 4},
}
NOMBRE_BLOQUE = {"Valor": "Precio atractivo", "Calidad": "Solidez del negocio",
                 "Crecimiento": "Crecimiento", "Momentum": "Tendencia del precio"}
FRASE_BUENA = {"Valor": "Barata frente a empresas similares", "Calidad": "Negocio rentable y con finanzas sanas",
               "Crecimiento": "Ventas y utilidades creciendo", "Momentum": "El precio viene subiendo"}
FRASE_MALA = {"Valor": "cara frente a sus pares", "Calidad": "rentabilidad o deuda débiles",
              "Crecimiento": "crecimiento bajo", "Momentum": "precio en caída"}
BENCH = "^GSPC"
MONEDAS_SUB = {"GBp": ("GBP", 100), "ZAc": ("ZAR", 100), "ILA": ("ILS", 100)}

# ---------------------------------------------------------------------------
# Datos (caché de 1 hora)
# ---------------------------------------------------------------------------

def _es_limite(e: Exception) -> bool:
    txt = f"{type(e).__name__} {e}".lower()
    return "ratelimit" in txt or "rate limit" in txt or "too many requests" in txt or "429" in txt


def _info(t: str, bloqueo: threading.Event | None = None) -> tuple[str, dict, str]:
    """Pide los fundamentales a Yahoo con reintentos. Devuelve (símbolo usado, info, error).
    Si Yahoo ya bloqueó varias consultas (`bloqueo`), no insiste para no alargar la espera."""
    error = ""
    for simbolo in (t, RESPALDO.get(t)):
        if not simbolo:
            continue
        for intento in range(2):
            if bloqueo is not None and bloqueo.is_set():
                return t, {}, "limite"
            try:
                i = yf.Ticker(simbolo).info or {}
                if i.get("regularMarketPrice") or i.get("currentPrice"):
                    return simbolo, i, ""
                error = error or "sin datos"
                break
            except Exception as e:  # noqa: BLE001
                error = "limite" if _es_limite(e) else f"{type(e).__name__}: {e}"[:200]
                if error == "limite":
                    if bloqueo is not None and intento == 1:
                        bloqueo.set()
                    time.sleep(1.5)
                    continue
                break
    return t, {}, error


@st.cache_resource
def _cache_info() -> dict:
    """Caché compartida que SOLO guarda respuestas exitosas (los errores no quedan guardados)."""
    return {}


def _moneda_probable(t: str) -> str:
    return {".CL": "COP", ".TO": "CAD", ".SA": "BRL", ".MX": "MXN", ".DE": "EUR", ".L": "GBp"}.get(
        "." + t.rsplit(".", 1)[-1] if "." in t else "", "USD")


def cargar_fundamentales(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    cache, ahora = _cache_info(), time.time()
    pendientes = [t for t in tickers if t not in cache or ahora - cache[t][0] > 3600]
    errores = {}
    if pendientes:
        bloqueo = threading.Event()
        with ThreadPoolExecutor(max_workers=4) as ex:  # pocas consultas a la vez para no activar el límite de Yahoo
            for t, (sim, info, err) in zip(pendientes, ex.map(lambda x: _info(x, bloqueo), pendientes)):
                if info:
                    cache[t] = (ahora, sim, info)
                else:
                    errores[t] = err
    filas = [M.metricas_desde_info(cache[t][1], cache[t][2]) for t in tickers if t in cache]
    for f in filas:
        f["Con fundamentales"] = True
    # Plan B: si Yahoo no entregó fundamentales, al menos traemos el precio (otro servicio de Yahoo, menos restringido)
    sin_info = [t for t in tickers if t not in cache]
    if sin_info:
        hist = cargar_precios(tuple(sin_info), "5d")
        for t in sin_info:
            if t in hist and hist[t].notna().any():
                f = M.metricas_desde_info(t, {"regularMarketPrice": float(hist[t].dropna().iloc[-1]), "shortName": t,
                                              "currency": _moneda_probable(t),
                                              "quoteType": "CRYPTOCURRENCY" if t.endswith("-USD") else "EQUITY"})
                f["Con fundamentales"] = False
                filas.append(f)
    df = pd.DataFrame(filas)
    if len(df):
        df = df[df["Precio"].notna()].drop_duplicates("Ticker")
    return df.reset_index(drop=True), errores


class SinDatos(Exception):
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def _precios_cache(tickers: tuple[str, ...], periodo: str) -> pd.DataFrame:
    data = yf.download(list(tickers), period=periodo, auto_adjust=True, progress=False, threads=True)
    if data is None or data.empty:
        raise SinDatos("Yahoo no devolvió precios")  # las excepciones no se guardan en caché
    close = data["Close"] if "Close" in data else data
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.dropna(how="all", axis=1)
    if close.empty:
        raise SinDatos("Yahoo no devolvió precios")
    return close


def cargar_precios(tickers: tuple[str, ...], periodo: str) -> pd.DataFrame:
    for intento in range(2):
        try:
            return _precios_cache(tickers, periodo)
        except SinDatos:
            if intento == 0:
                time.sleep(2)
        except Exception as e:  # noqa: BLE001
            if not _es_limite(e) or intento == 1:
                break
            time.sleep(3)
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def tasa_libre_riesgo() -> float | None:
    try:
        return float(cargar_precios(("^IRX",), "1mo").iloc[:, 0].dropna().iloc[-1]) / 100
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def tipo_cambio(de: str, a: str) -> float:
    if de == a:
        return 1.0
    p = cargar_precios((f"{de}{a}=X",), "5d")
    if p.empty:
        raise ValueError(f"No encontré el tipo de cambio {de}/{a} en Yahoo")
    return float(p.iloc[:, 0].dropna().iloc[-1])


@st.cache_data(ttl=1800, show_spinner=False)
def screener_yahoo(clave: str, n: int) -> list[str]:
    return [q["symbol"] for q in yf.screen(clave, count=n).get("quotes", []) if q.get("symbol")]


def parse_tickers(txt: str) -> list[str]:
    return [t.strip().upper() for t in txt.replace(",", " ").replace(";", " ").split() if t.strip()]


def esc(x) -> str:
    return html.escape(str(x))


def etiqueta(p: float) -> str:
    if p >= 65:
        return '<span class="badge b-alta">Oportunidad alta</span>'
    if p >= 45:
        return '<span class="badge b-media">Neutral</span>'
    return '<span class="badge b-baja">Poco atractiva</span>'


def barras_bloques(fila) -> str:
    return "".join(
        f'<div class="fila-b"><span>{NOMBRE_BLOQUE[b]}</span><span>{fila[b]:.0f}</span></div>'
        f'<div class="barra"><div style="width:{max(fila[b], 2):.0f}%"></div></div>' for b in NOMBRE_BLOQUE)


def por_que(fila) -> str:
    buenas = [FRASE_BUENA[b] for b in NOMBRE_BLOQUE if fila[b] >= 70]
    malas = [FRASE_MALA[b] for b in NOMBRE_BLOQUE if fila[b] <= 30]
    txt = " · ".join(buenas[:2]) or "Sin extremos: equilibrada en todos los frentes"
    return txt + (f". Ojo: {', '.join(malas[:2])}." if malas else ".")


def dinero(x: float, moneda: str) -> str:
    return f"{x:,.0f} {moneda}" if moneda == "COP" else f"{x:,.2f} {moneda}"


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="titulo">Radar de <span>inversión</span></div>
  <p>Encuentra acciones y criptomonedas interesantes, entiende si su precio tiene sentido y descubre
  cómo repartir tu dinero entre ellas. Sin necesidad de ser experto.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Encuesta de perfil de inversionista
# ---------------------------------------------------------------------------
PREGUNTAS = [
    ("¿Para qué es el dinero que quieres invertir?",
     ["Quiero protegerlo y que no pierda valor", "Quiero que crezca de forma estable", "Quiero que crezca lo más posible"]),
    ("¿Cuándo crees que vas a necesitar ese dinero?",
     ["En menos de 2 años", "Entre 2 y 5 años", "En más de 5 años"]),
    ("Imagina que tu inversión cae 20% en un mes. ¿Qué harías?",
     ["Vendería para no perder más", "Esperaría a que se recupere", "Aprovecharía para comprar más"]),
    ("¿Cuánta experiencia tienes invirtiendo?",
     ["Ninguna, estoy empezando", "Algo: CDT, fondos o pensiones voluntarias", "Bastante: acciones, ETF o cripto"]),
]
ATRAE = {"Que esté a buen precio": "🏷️ Buscar gangas", "Que sea sólida y rentable": "🏛️ Negocios sólidos",
         "Que esté creciendo rápido": "🚀 Crecimiento", "Que su acción venga subiendo": "📈 Seguir la tendencia",
         "Un poco de todo": "⚖️ Equilibrado"}
PERFILES = {
    "Conservador": dict(desc="Para ti lo más importante es cuidar tu dinero. Prefieres ganar un poco menos a cambio de "
                             "evitar sustos, y eso está perfecto.",
                        recomendado="Mínima varianza", cap=0.25),
    "Moderado": dict(desc="Buscas que tu dinero crezca y aceptas algunos altibajos en el camino, siempre que no sean "
                          "demasiado bruscos. Es el punto medio entre seguridad y rendimiento.",
                     recomendado="Máximo Sharpe", cap=0.30),
    "Arriesgado": dict(desc="Piensas en el largo plazo y las caídas temporales no te quitan el sueño. Estás dispuesto a "
                            "asumir más riesgo a cambio de mayor potencial de crecimiento.",
                       recomendado="Máximo Sharpe", cap=0.40),
}


def calcular_perfil(respuestas: list[int], atrae: str, cripto: str, nombre: str) -> dict:
    puntos = sum(respuestas) + len(respuestas)  # cada respuesta vale 1, 2 o 3
    tipo = "Conservador" if puntos <= 6 else "Moderado" if puntos <= 9 else "Arriesgado"
    return dict(nombre=nombre.strip(), tipo=tipo, puntos=puntos, estilo=ATRAE[atrae],
                cripto=cripto.startswith("Sí") and tipo != "Conservador", quiere_cripto=cripto.startswith("Sí"),
                **PERFILES[tipo])


perfil = st.session_state.get("perfil")
if perfil is None and not st.session_state.get("sin_encuesta"):
    st.markdown("""<div class="encuesta"><div class="kicker">Antes de empezar</div>
        <h3>Conozcámonos un poco</h3>
        <p>Responde 6 preguntas rápidas (toma menos de un minuto) y ajustamos la app a tu forma de invertir:
        el estilo de búsqueda, el tipo de portafolio que más te conviene y cuánto concentrar en cada activo.
        No hay respuestas buenas ni malas.</p></div>""", unsafe_allow_html=True)
    with st.form("encuesta"):
        nombre = st.text_input("¿Cómo te llamas? (opcional)", placeholder="Tu nombre")
        resp = []
        for fila_q in range(2):
            cols = st.columns(2, gap="large")
            for j in range(2):
                i = fila_q * 2 + j
                preg, ops = PREGUNTAS[i]
                with cols[j]:
                    r = st.radio(f"{i + 1}. {preg}", ops, index=None, key=f"q{i}")
                    resp.append(None if r is None else ops.index(r))
        cols = st.columns(2, gap="large")
        with cols[0]:
            atrae = st.radio("5. Cuando miras una empresa, ¿qué es lo que más te llama la atención?", list(ATRAE), index=None)
        with cols[1]:
            cripto = st.radio("6. ¿Te gustaría incluir criptomonedas?",
                              ["Sí, una parte pequeña", "No, prefiero dejarlas por fuera"], index=None)
        c1, c2 = st.columns([1, 4])
        listo = c1.form_submit_button("Ver mi perfil", type="primary")
        saltar = c2.form_submit_button("Prefiero saltarla por ahora")
    if saltar:
        st.session_state["sin_encuesta"] = True
        st.rerun()
    if listo:
        if None in resp or atrae is None or cripto is None:
            st.warning("Te faltan algunas preguntas. Respóndelas todas para darte un perfil acertado.")
        else:
            st.session_state["perfil"] = calcular_perfil(resp, atrae, cripto, nombre)
            st.session_state.pop("opt", None)
            st.rerun()
    st.stop()

if perfil:
    saludo = f"{esc(perfil['nombre'])}, tu" if perfil["nombre"] else "Tu"
    extra_cripto = ("Incluiremos criptomonedas en una proporción pequeña." if perfil["cripto"] else
                    "Por tu perfil, dejamos las criptomonedas por fuera de la sugerencia inicial (igual puedes agregarlas)."
                    if perfil["quiere_cripto"] else "Dejamos las criptomonedas por fuera, como prefieres.")
    p1, p2 = st.columns([5, 1], vertical_alignment="center")
    p1.markdown(f"""<div class="perfil-card"><div class="kicker">{saludo} perfil de inversionista</div>
        <div class="perfil-tipo">{perfil['tipo']}</div><p>{perfil['desc']}</p>
        <div class="chips"><span>Estilo sugerido: <b>{esc(perfil['estilo'].split(' ', 1)[1])}</b></span>
        <span>Portafolio sugerido: <b>{perfil['recomendado']}</b></span>
        <span>Máximo por activo: <b>{perfil['cap']:.0%}</b></span></div>
        <p class="nota">{extra_cripto} Es una orientación para empezar, no una asesoría profesional.</p></div>""",
                unsafe_allow_html=True)
    if p2.button("Repetir encuesta", width="stretch"):
        for k in ["perfil", "opt"] + [f"q{i}" for i in range(len(PREGUNTAS))]:
            st.session_state.pop(k, None)
        st.rerun()
else:
    if st.button("Hacer la encuesta de perfil"):
        st.session_state.pop("sin_encuesta", None)
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["1 · Explora", "2 · Analiza", "3 · Tu portafolio", "¿Cómo funciona?"])

# ---------------------------------------------------------------------------
# Paso 1: Explorar
# ---------------------------------------------------------------------------
with tab1:
    c_a, c_b = st.columns([3, 2], gap="large")
    with c_a:
        st.markdown("#### ¿Qué mercados quieres revisar?")
        grupos = st.pills("Mercados", list(GRUPOS), selection_mode="multi", label_visibility="collapsed",
                          default=["🇨🇴 Bolsa de Colombia", "₿ Criptomonedas", "🇺🇸 Gigantes de EE.UU."])
        with st.expander("➕ Agregar otros activos o usar un buscador de Yahoo"):
            manual = st.text_input("Símbolos de Yahoo Finance separados por coma",
                                   placeholder="Ej: KO, NFLX, ADA-USD, PETR4.SA",
                                   help="Busca el símbolo en finance.yahoo.com. Colombia termina en .CL, criptos en -USD.")
            scr = st.selectbox("Buscador automático de Yahoo (opcional)", ["Ninguno"] + list(SCREENERS))
    with c_b:
        st.markdown("#### ¿Qué buscas en una inversión?")
        opciones_estilo = list(ESTILOS) + ["🎛️ Personalizado"]
        idx = opciones_estilo.index(perfil["estilo"]) if perfil else 0
        estilo = st.radio("Estilo", opciones_estilo, index=idx, label_visibility="collapsed",
                          help="Viene preseleccionado según tu encuesta; puedes cambiarlo cuando quieras.")
        if estilo == "🎛️ Personalizado":
            pesos = {b: st.slider(NOMBRE_BLOQUE[b], 0, 5, 2) for b in NOMBRE_BLOQUE}
        else:
            pesos = ESTILOS[estilo]
    buscar = st.button("Buscar oportunidades", type="primary", icon="📡")

    if buscar:
        tickers = [t for g in (grupos or []) for t in GRUPOS[g].split()] + parse_tickers(manual)
        if scr != "Ninguno":
            try:
                tickers += screener_yahoo(SCREENERS[scr], 25)
            except Exception:
                st.warning("El buscador de Yahoo no respondió; sigo con tu selección.")
        tickers = tuple(dict.fromkeys(tickers))
        if not tickers:
            st.error("Elige al menos un mercado o escribe un símbolo.")
        else:
            with st.spinner(f"Leyendo {len(tickers)} activos en Yahoo Finance…"):
                fund, errores = cargar_fundamentales(tickers)
                hist = cargar_precios(tuple(fund["Ticker"]), "13mo") if len(fund) else pd.DataFrame()
            limite = any(e == "limite" for e in errores.values())
            if fund.empty:
                if limite:
                    st.error("Yahoo está limitando las consultas en este momento (pasa a menudo cuando la app corre en la nube, "
                             "porque muchas apps comparten el mismo servidor). Espera uno o dos minutos y vuelve a intentar, "
                             "o elige menos mercados a la vez.")
                else:
                    st.error("Yahoo no devolvió datos. Espera un momento y vuelve a intentar; si sigue pasando, revisa el detalle abajo.")
                with st.expander("Detalle técnico del error"):
                    st.write({t: ("límite de consultas de Yahoo" if e == "limite" else e) for t, e in list(errores.items())[:15]})
            else:
                sin_fund = int((~fund["Con fundamentales"]).sum())
                if sin_fund:
                    st.warning(f"Yahoo solo entregó el precio de {sin_fund} activo(s), sin sus estados financieros"
                               + (" (límite de consultas)" if limite else "")
                               + ". Su puntaje de empresa será neutral; vuelve a buscar en unos minutos para completarlos.")
                st.session_state.update(
                    fund=fund.join(M.momentum(hist), on="Ticker"), fecha=datetime.now().strftime("%d/%m/%Y %H:%M"),
                    faltan=[t for t in tickers if t not in set(fund["Ticker"]) and RESPALDO.get(t) not in set(fund["Ticker"])])
                st.session_state.pop("opt", None)

    if "fund" not in st.session_state:
        st.markdown('<div class="explica azul">Elige los mercados, tu estilo y pulsa <b>Buscar oportunidades</b>. '
                    'La app califica cada acción de 0 a 100 según qué tan buena oportunidad parece hoy.</div>',
                    unsafe_allow_html=True)
    else:
        base = st.session_state["fund"]
        acc = base[base["Tipo"] == "Acción"]
        cri = base[base["Tipo"] == "Cripto"]
        df = M.puntajes(acc, pesos) if len(acc) else acc.assign(**{b: np.nan for b in [*NOMBRE_BLOQUE, "Puntaje"]})
        st.session_state["df"] = df
        st.caption(f"Datos de Yahoo Finance · {st.session_state['fecha']} · {len(acc)} acciones y {len(cri)} criptos"
                   + (f" · sin datos: {', '.join(st.session_state['faltan'])}" if st.session_state["faltan"] else ""))

        if len(df):
            st.markdown("### Las 5 mejores oportunidades según tu estilo")
            cols = st.columns(min(5, len(df)))
            for col, (_, f) in zip(cols, df.head(5).iterrows()):
                col.markdown(f"""<div class="card"><div class="sub">{esc(f['Ticker'])} · {esc(f['Sector'])}</div>
                    <h4>{esc(f['Nombre'])[:28]}</h4><div class="big">{f['Puntaje']:.0f}<span style="font-size:15px;color:{MUTED}">/100</span></div>
                    {etiqueta(f['Puntaje'])}<div style="margin-top:12px">{barras_bloques(f)}</div>
                    <div class="sub" style="margin:6px 0 0">{esc(por_que(f))}</div></div>""", unsafe_allow_html=True)

            st.markdown("### Mapa: precio vs. solidez")
            st.markdown('<div class="explica">Cada burbuja es una empresa (más grande = empresa más grande). '
                        'Arriba a la derecha están las <b>baratas y sólidas</b>, la zona más interesante.</div>', unsafe_allow_html=True)
            tam = df["Cap. mercado"].fillna(df["Cap. mercado"].median()).fillna(1.0).clip(lower=1)
            fig = go.Figure(go.Scatter(
                x=df["Valor"], y=df["Calidad"], mode="markers+text", text=df["Ticker"].str.replace(".CL", "", regex=False),
                textposition="top center", customdata=np.stack([df["Nombre"], df["Puntaje"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b><br>Precio atractivo %{x:.0f} · Solidez %{y:.0f}<br>Puntaje %{customdata[1]:.0f}<extra></extra>",
                marker=dict(size=np.sqrt(tam / tam.max()) * 46 + 8, color=df["Puntaje"], colorscale=ESCALA, cmin=20, cmax=80,
                            line=dict(color=NAVY, width=1), opacity=0.9, colorbar=dict(title="Puntaje"))))
            fig.add_shape(type="rect", x0=50, x1=100, y0=50, y1=100, fillcolor="rgba(57,255,136,0.06)", line_width=0, layer="below")
            fig.add_annotation(x=98, y=98, text="Baratas y sólidas", showarrow=False, xanchor="right", font=dict(color=VERDE))
            fig.update_layout(height=460, xaxis=dict(title="Precio atractivo →", range=[0, 102]),
                              yaxis=dict(title="Solidez del negocio →", range=[0, 102]))
            st.plotly_chart(fig, width="stretch", theme=None)

            st.markdown("### Todas las acciones")
            tecnico = st.toggle("Mostrar métricas técnicas (múltiplos y rentabilidades)", False)
            simple = ["Ticker", "Nombre", "Sector", "Puntaje", "Valor", "Calidad", "Crecimiento", "Momentum",
                      "Precio", "Moneda", "Ret. 6m", "Volatilidad", "Upside analistas"]
            tec = ["Cap. mercado", "P/E", "P/E fwd", "PEG", "P/B", "P/S", "EV/EBITDA", "FCF yield", "Div. yield", "ROE",
                   "Margen operativo", "Margen neto", "Crec. ventas", "Crec. utilidades", "Deuda/Patrimonio", "Beta", "Mom. 12-1"]
            cfg = {
                "Puntaje": st.column_config.ProgressColumn("Puntaje", min_value=0, max_value=100, format="%.0f"),
                "Valor": st.column_config.NumberColumn("Precio atractivo", format="%.0f"),
                "Calidad": st.column_config.NumberColumn("Solidez", format="%.0f"),
                "Crecimiento": st.column_config.NumberColumn("Crecimiento", format="%.0f"),
                "Momentum": st.column_config.NumberColumn("Tendencia", format="%.0f"),
                "Precio": st.column_config.NumberColumn(format="localized"),
                "Cap. mercado": st.column_config.NumberColumn(format="compact"),
                "Ret. 6m": st.column_config.NumberColumn("Rentab. 6 meses", format="percent"),
                "Volatilidad": st.column_config.NumberColumn("Altibajos (vol.)", format="percent",
                                                             help="Cuánto sube y baja el precio en un año típico."),
                "Upside analistas": st.column_config.NumberColumn("Potencial según analistas", format="percent"),
                **{c: st.column_config.NumberColumn(format="%.1f") for c in ["P/E", "P/E fwd", "PEG", "P/B", "P/S", "EV/EBITDA", "Deuda/Patrimonio", "Beta"]},
                **{c: st.column_config.NumberColumn(format="percent") for c in ["FCF yield", "Div. yield", "ROE", "Margen operativo",
                                                                                  "Margen neto", "Crec. ventas", "Crec. utilidades", "Mom. 12-1"]},
            }
            st.dataframe(df[simple + (tec if tecnico else [])], hide_index=True, width="stretch", column_config=cfg, height=420)
            st.download_button("Descargar tabla (CSV)", df[simple + tec].to_csv(index=False).encode("utf-8"), "oportunidades.csv", "text/csv")

        if len(cri):
            st.markdown("### Criptomonedas")
            st.markdown('<div class="explica azul">Las criptos no tienen ventas ni utilidades, así que no reciben puntaje de empresa. '
                        'Aquí ves su precio, su tendencia y qué tan bruscos son sus movimientos. Puedes incluirlas en tu portafolio en el paso 3.</div>',
                        unsafe_allow_html=True)
            cols = st.columns(min(4, len(cri)))
            for i, (_, f) in enumerate(cri.iterrows()):
                r6 = "—" if pd.isna(f["Ret. 6m"]) else f"{f['Ret. 6m']:+.1%}"
                vol = "—" if pd.isna(f["Volatilidad"]) else f"{f['Volatilidad']:.0%}"
                dd = "—" if pd.isna(f["Caída máx. 1a"]) else f"{f['Caída máx. 1a']:.0%}"
                cols[i % len(cols)].markdown(f"""<div class="card"><div class="sub">{esc(f['Ticker'])}</div><h4>{esc(f['Nombre'])}</h4>
                    <div class="big">{f['Precio']:,.2f}<span style="font-size:14px;color:{MUTED}"> USD</span></div>
                    <ul><li>Últimos 6 meses: <b>{r6}</b></li><li>Altibajos al año: <b>{vol}</b></li>
                    <li>Peor caída del último año: <b>{dd}</b></li></ul></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paso 2: Analizar una empresa
# ---------------------------------------------------------------------------
with tab2:
    df = st.session_state.get("df")
    if df is None or not len(df):
        st.info("Primero busca oportunidades en el paso 1 (con al menos una acción).")
    else:
        nombres = df.set_index("Ticker")["Nombre"]
        sel = st.selectbox("¿Qué empresa quieres analizar?", df["Ticker"], format_func=lambda t: f"{nombres[t]}  ·  {t}")
        fila = df.set_index("Ticker").loc[sel]
        otros = [t for t in df["Ticker"] if t != sel]
        mismos = [t for t in otros if df.set_index("Ticker").loc[t, "Sector"] == fila["Sector"]]
        with st.expander(f"Empresas con las que se compara ({len(mismos) or min(8, len(otros))})"):
            pares = st.multiselect("Comparables", otros, default=mismos or otros[:8], label_visibility="collapsed",
                                   help="Por defecto: empresas del mismo sector que cargaste en el paso 1.")
        val = M.valoracion_multiplos(df, sel, pares) if pares else pd.DataFrame()

        izq, der = st.columns([2, 3], gap="large")
        with izq:
            if len(val):
                med = val["Precio implícito"].median()
                gap = med / fila["Precio"] - 1
                veredicto = ("parece <b>barata</b>" if gap > 0.10 else "parece <b>cara</b>" if gap < -0.10
                             else "parece estar en un <b>precio justo</b>")
                st.markdown(f"""<div class="card"><div class="sub">{esc(sel)} · {esc(fila['Sector'])}</div><h4>{esc(fila['Nombre'])}</h4>
                    <div class="big">{gap:+.0%}</div>{etiqueta(fila['Puntaje'])}
                    <p style="margin:10px 0 0;color:#C9D4EE">Frente a empresas similares, esta acción {veredicto}.
                    Si se valorara igual que ellas, costaría cerca de <b>{med:,.2f} {esc(fila['Moneda'])}</b>
                    (hoy: {fila['Precio']:,.2f}).</p>
                    <div style="margin-top:14px">{barras_bloques(fila)}</div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card"><h4>{esc(fila["Nombre"])}</h4><p>No hay suficientes datos positivos '
                            '(por ejemplo, la empresa tiene pérdidas) para compararla por múltiplos.</p>'
                            f'<div style="margin-top:14px">{barras_bloques(fila)}</div></div>', unsafe_allow_html=True)
            if pd.notna(fila["Upside analistas"]):
                st.metric("Precio objetivo de los analistas", f"{fila['Precio'] * (1 + fila['Upside analistas']):,.2f}",
                          f"{fila['Upside analistas']:+.1%} vs. hoy")
        with der:
            if len(val):
                st.markdown("##### ¿Cuánto valdría según cada forma de medir?")
                fig = go.Figure(go.Bar(y=val["Método"], x=val["Precio implícito"], orientation="h",
                                       marker_color=np.where(val["Upside"] >= 0, VERDE, GRIS3),
                                       text=[f"{v:,.2f} ({u:+.0%})" for v, u in zip(val["Precio implícito"], val["Upside"])],
                                       textposition="outside", cliponaxis=False))
                fig.add_vline(x=fila["Precio"], line_color=BEIGE, line_width=2, line_dash="dot",
                              annotation_text="precio hoy", annotation_font_color=BEIGE)
                fig.update_layout(height=330, xaxis_title=f"Precio por acción ({fila['Moneda']})", margin=dict(r=120))
                st.plotly_chart(fig, width="stretch", theme=None)
                st.caption("Verde: valdría más que hoy. Gris: valdría menos. Cada barra usa un múltiplo distinto "
                           "(P/E = precio/utilidad, P/B = precio/patrimonio, P/S = precio/ventas, EV/EBITDA = valor empresa/flujo operativo).")

        st.markdown("##### Sus números frente a los comparables, en palabras simples")
        grupo = df[df["Ticker"].isin(pares)] if pares else df.iloc[0:0]
        filas = [
            ("P/E", "Años de utilidades que pagas al comprar la acción", "menor", False),
            ("P/E fwd", "Lo mismo, con las utilidades esperadas del próximo año", "menor", False),
            ("P/B", "Cuántas veces pagas su patrimonio contable", "menor", False),
            ("EV/EBITDA", "Valor total de la empresa frente a su flujo operativo", "menor", False),
            ("Div. yield", "Dividendo anual que recibes por cada 100 invertidos", "mayor", True),
            ("ROE", "Ganancia que genera por cada 100 de patrimonio", "mayor", True),
            ("Margen neto", "De cada 100 que vende, cuánto le queda de utilidad", "mayor", True),
            ("Crec. ventas", "Crecimiento de las ventas frente al año anterior", "mayor", True),
            ("Deuda/Patrimonio", "Cuánto debe por cada peso de patrimonio", "menor", False),
            ("Beta", "Qué tanto se mueve frente al mercado (1 = igual)", "—", False),
        ]
        fmt = lambda v, pct: "—" if pd.isna(v) else (f"{v:.1%}" if pct else f"{v:.2f}")
        tabla = []
        for c, desc, mejor, pct in filas:
            v, m = fila[c], grupo[c].median() if len(grupo) else np.nan
            if pd.isna(v) or pd.isna(m) or mejor == "—":
                lectura = "—"
            else:
                bien = v < m if mejor == "menor" else v > m
                if c in ("P/E", "P/E fwd", "EV/EBITDA") and v <= 0:
                    bien = False
                lectura = "✅ Mejor que sus pares" if bien else "⚠️ Peor que sus pares"
            tabla.append({"Indicador": c, "Qué significa": desc, sel: fmt(v, pct), "Pares (mediana)": fmt(m, pct), "Lectura": lectura})
        st.dataframe(pd.DataFrame(tabla), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Paso 3: Portafolio
# ---------------------------------------------------------------------------
with tab3:
    base = st.session_state.get("fund")
    if base is None:
        st.info("Primero busca oportunidades en el paso 1.")
    else:
        df = st.session_state["df"]
        rec = perfil["recomendado"] if perfil else None
        sello = lambda n: '<span class="badge b-alta" style="margin-left:8px">Sugerido para tu perfil</span>' if rec == n else ""
        k1, k2 = st.columns(2, gap="large")
        k1.markdown(f"""<div class="card sharpe"><h4>🎯 Portafolio de máximo Sharpe {sello("Máximo Sharpe")}</h4><div class="sub">El más eficiente</div>
            Busca la combinación que da <b>el mayor rendimiento por cada unidad de riesgo</b> que asumes.
            Es como elegir el carro que más kilómetros recorre por galón: no el más rápido, sino el que mejor aprovecha cada gota.
            <ul><li>Ideal si aceptas altibajos a cambio de mejor rendimiento.</li>
            <li>Suele concentrarse en pocos activos.</li></ul></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="card minvar"><h4>🛡️ Portafolio de mínima varianza {sello("Mínima varianza")}</h4><div class="sub">El más estable</div>
            Combina los activos para que <b>el valor total de tu inversión se mueva lo menos posible</b>,
            aprovechando que unos suben cuando otros bajan. No persigue el mayor rendimiento, persigue la tranquilidad.
            <ul><li>Ideal si priorizas dormir tranquilo.</li><li>Suele estar más diversificado.</li></ul></div>""", unsafe_allow_html=True)
        st.write("")

        orden = df["Ticker"].tolist() + base.loc[base["Tipo"] == "Cripto", "Ticker"].tolist()
        con_cripto = perfil["cripto"] if perfil else True
        defecto = df["Ticker"].head(5).tolist() + [t for t in ["BTC-USD"] if t in orden and con_cripto]
        if len(defecto) < 2:
            defecto = orden[:6]
        nombres_all = base.set_index("Ticker")["Nombre"]
        moneda = st.segmented_control("¿En qué moneda vas a invertir?", ["COP", "USD", "EUR"], default="COP")
        moneda = moneda or "COP"
        with st.form("port"):
            activos = st.multiselect("¿Entre cuáles activos quieres repartir tu dinero?", orden, default=defecto,
                                     format_func=lambda t: f"{nombres_all.get(t, t)} · {t}",
                                     help="Están ordenados por puntaje; las criptos aparecen al final.")
            monto = st.number_input(f"¿Cuánto quieres invertir? ({moneda})", min_value=0.0, step=100_000.0 if moneda == "COP" else 100.0,
                                    value=10_000_000.0 if moneda == "COP" else 2_500.0, key=f"monto_{moneda}", format="%.0f")
            with st.expander("⚙️ Opciones avanzadas"):
                a1, a2, a3 = st.columns(3)
                periodo = a1.selectbox("Historia usada para estimar", ["1y", "2y", "3y", "5y"], index=2,
                                       format_func=lambda p: {"1y": "1 año", "2y": "2 años", "3y": "3 años", "5y": "5 años"}[p])
                rf_live = tasa_libre_riesgo()
                rf = a2.number_input("Tasa libre de riesgo anual (%)", value=round((rf_live or 0.04) * 100, 2), step=0.1,
                                     help="Lo que ganarías sin riesgo. Por defecto: T-Bill de EE.UU. a 3 meses (^IRX).") / 100
                cap = a3.slider("Máximo por activo", 0.1, 1.0, perfil["cap"] if perfil else 0.35, 0.05, format="%.2f",
                                help="Evita poner demasiado dinero en un solo activo. 0.35 = máximo 35%.")
                b1, b2, b3 = st.columns(3)
                metodo_mu = b1.selectbox("Rendimiento esperado", ["Histórico", "CAPM"],
                                         help="Histórico: lo que ha rendido en promedio. CAPM: rf + β × prima de mercado.")
                prima = b2.number_input("Prima de mercado (%) para CAPM", value=5.5, step=0.25) / 100
                metodo_cov = b3.selectbox("Estimación del riesgo", ["Ledoit-Wolf", "Muestral"],
                                          help="Ledoit-Wolf suaviza las estimaciones y da pesos más estables.")
                fracc = st.checkbox("Permitir fracciones de acción (las criptos siempre se compran en fracciones)", False)
            enviar = st.form_submit_button("Calcular mi portafolio", type="primary", icon="🧩")

        if enviar:
            if len(activos) < 2:
                st.error("Elige al menos 2 activos.")
            elif monto <= 0:
                st.error("Escribe un monto mayor que cero.")
            else:
                with st.spinner("Descargando precios históricos y calculando…"):
                    try:
                        cripto = set(base.loc[base["Tipo"] == "Cripto", "Ticker"])
                        hist = cargar_precios(tuple(activos) + (BENCH,), periodo)
                        usables = [a for a in activos if a in hist and hist[a].notna().sum() > 120]
                        if len(usables) < 2:
                            raise ValueError("No hay suficiente historia de precios para estos activos.")
                        ret = M.retornos_diarios(hist[usables], cripto)
                        per = M.periodos_por_anio(ret)
                        mkt = None
                        if BENCH in hist:
                            mkt = hist[BENCH].ffill().pct_change(fill_method=None).reindex(ret.index).fillna(0)
                        mu, cov, delta, betas = M.estimar_parametros(ret, metodo_cov, "CAPM" if metodo_mu == "CAPM" else "Histórica",
                                                                     rf, mkt, prima, per)
                        cap_ef = max(cap, 1 / len(usables))
                        info = base.set_index("Ticker").loc[usables]
                        precios_act, fx = [], []
                        for t in usables:
                            cur, p = info.loc[t, "Moneda"], info.loc[t, "Precio"]
                            if cur in MONEDAS_SUB:
                                cur, d = MONEDAS_SUB[cur]
                                p /= d
                            precios_act.append(p)
                            fx.append(tipo_cambio(cur, moneda))
                        st.session_state["opt"] = dict(
                            mu=mu, cov=cov, rf=rf, delta=delta, betas=betas, ret=ret, mkt=mkt, cap=cap_ef, moneda=moneda, monto=monto,
                            w={"Máximo Sharpe": pd.Series(M.max_sharpe(mu, cov, rf, cap_ef), mu.index),
                               "Mínima varianza": pd.Series(M.min_varianza(mu, cov, cap_ef), mu.index)},
                            precios=pd.Series(precios_act, usables), fx=pd.Series(fx, usables),
                            fracc=pd.Series([fracc or t in cripto for t in usables], usables),
                            excluidos=sorted(set(activos) - set(usables)), metodo_cov=metodo_cov, metodo_mu=metodo_mu)
                    except Exception as e:
                        st.error(f"No se pudo calcular: {e}")

        o = st.session_state.get("opt")
        if o:
            mu, cov, rf, mon = o["mu"], o["cov"], o["rf"], o["moneda"]
            if o["excluidos"]:
                st.warning(f"Dejé por fuera (poca historia de precios): {', '.join(o['excluidos'])}")
            if (mu < rf).all():
                st.warning("Todos los activos rindieron menos que la tasa libre de riesgo en este periodo; el portafolio de Sharpe será 'el menos malo'.")
            colores = {"Máximo Sharpe": CIAN, "Mínima varianza": GRIS1}
            columnas = st.columns(2, gap="large")
            for col, (nombre, w) in zip(columnas, o["w"].items()):
                r, v, s = M._stats(w.values, mu.values, cov.values, rf)
                asig = M.asignar_monto(w, o["precios"], o["fx"], o["monto"], o["fracc"])
                asig = asig[asig["Peso objetivo"] > 0].sort_values("Peso objetivo", ascending=False)
                with col:
                    st.markdown(f"<h3>{'🎯' if nombre == 'Máximo Sharpe' else '🛡️'} {nombre} {sello(nombre)}</h3>", unsafe_allow_html=True)
                    pal = ([CIAN, GRIS1, VERDE, GRIS3, "#7CF5FF", GRIS2, "#1F8FA3"] if nombre == "Máximo Sharpe"
                           else [GRIS1, CIAN, GRIS3, VERDE, GRIS2, "#7CF5FF", "#3B4457"])
                    fig = go.Figure(go.Pie(labels=[t.replace(".CL", "") for t in asig.index], values=asig["Peso objetivo"], hole=0.62, sort=False,
                                           marker=dict(colors=pal * 5, line=dict(color=NAVY, width=2)),
                                           textinfo="label+percent", textposition="outside",
                                           hovertemplate="%{label}: %{percent}<extra></extra>"))
                    fig.update_layout(height=320, showlegend=False, margin=dict(t=30, b=30, l=40, r=40),
                                      annotations=[dict(text=f"<b>{dinero(o['monto'], mon)}</b>", showarrow=False, font=dict(size=15, color=BEIGE))])
                    st.plotly_chart(fig, width="stretch", theme=None, key=f"pie_{nombre}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rinde al año", f"{r:.1%}", help="Rendimiento anual esperado según los datos históricos.")
                    m2.metric("Altibajos", f"{v:.1%}", help="Volatilidad anual: cuánto puede subir o bajar en un año típico.")
                    m3.metric("Sharpe", f"{s:.2f}", help="Rendimiento extra sobre la tasa libre de riesgo por cada unidad de riesgo. Más alto = más eficiente.")
                    lo, hi = o["monto"] * (1 + r - v), o["monto"] * (1 + r + v)
                    st.markdown(f'<div class="explica{" azul" if nombre != "Máximo Sharpe" else ""}">En un año normal (2 de cada 3 años), '
                                f'tus <b>{dinero(o["monto"], mon)}</b> terminarían aproximadamente entre <b>{dinero(lo, mon)}</b> y '
                                f'<b>{dinero(hi, mon)}</b>, según cómo se comportaron estos activos en el pasado.</div>', unsafe_allow_html=True)
                    tabla = pd.DataFrame({
                        "Activo": [f"{nombres_all.get(t, t)} · {t}" for t in asig.index],
                        "Peso": asig["Peso objetivo"].values,
                        f"Invertir ({mon})": asig["Monto invertido"].round(0 if mon == "COP" else 2).values,
                        "Qué comprar": [f"{u:,.4f}" if o["fracc"][t] else (f"{u:,.0f}" if u >= 1 else "0 · no alcanza para 1")
                                        for t, u in zip(asig.index, asig["Unidades"])],
                        f"Precio unidad ({mon})": asig["Precio (moneda inversión)"].round(0 if mon == "COP" else 2).values,
                    })
                    st.dataframe(tabla, hide_index=True, width="stretch", column_config={
                        "Peso": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                        f"Invertir ({mon})": st.column_config.NumberColumn(format="localized"),
                        f"Precio unidad ({mon})": st.column_config.NumberColumn(format="localized")})
                    sobra = o["monto"] - asig["Monto invertido"].sum()
                    if sobra > o["monto"] * 0.002:
                        st.caption(f"Quedan {dinero(sobra, mon)} sin invertir porque las acciones se compran en unidades enteras.")

            st.markdown("### ¿Dónde quedan tus portafolios frente a todas las combinaciones posibles?")
            st.markdown('<div class="explica">Cada punto pequeño es una forma distinta de repartir tu dinero. Más a la derecha = más altibajos; '
                        'más arriba = más rendimiento. La <b>línea verde</b> es la frontera eficiente: las mejores combinaciones posibles '
                        'para cada nivel de riesgo (respetando el máximo por activo). Tus dos portafolios están sobre ella.</div>', unsafe_allow_html=True)
            fr = M.frontera(mu, cov, o["cap"])
            rnd = M.portafolios_aleatorios(mu, cov, rf, cap=o["cap"])
            fig = go.Figure()
            fig.add_scatter(x=rnd["Volatilidad"], y=rnd["Retorno"], mode="markers", name="Combinaciones posibles", hoverinfo="skip",
                            marker=dict(size=4, color=rnd["Sharpe"], colorscale=ESCALA, opacity=0.35, showscale=False))
            fig.add_scatter(x=fr["Volatilidad"], y=fr["Retorno"], mode="lines", name="Frontera eficiente", line=dict(color=VERDE, width=3))
            vols = np.sqrt(np.diag(cov.values))
            fig.add_scatter(x=vols, y=mu.values, mode="markers+text", text=[t.replace(".CL", "") for t in mu.index],
                            textposition="top center", name="Cada activo solo", marker=dict(size=9, color=MUTED, symbol="diamond"))
            for nombre, w in o["w"].items():
                r, v, s = M._stats(w.values, mu.values, cov.values, rf)
                fig.add_scatter(x=[v], y=[r], mode="markers", name=nombre,
                                marker=dict(size=20, color=colores[nombre], symbol="star" if nombre == "Máximo Sharpe" else "circle",
                                            line=dict(color=NAVY, width=2)))
            fig.update_layout(height=500, xaxis=dict(title="Altibajos al año (volatilidad)", tickformat=".0%"),
                              yaxis=dict(title="Rendimiento esperado al año", tickformat=".0%"), legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig, width="stretch", theme=None)

            st.markdown("### ¿Cómo le habría ido a cada portafolio?")
            cum = pd.DataFrame({n: (1 + o["ret"] @ w).cumprod() * 100 for n, w in o["w"].items()})
            cum["Repartir en partes iguales"] = (1 + o["ret"].mean(axis=1)).cumprod() * 100
            if o["mkt"] is not None:
                cum["S&P 500"] = (1 + o["mkt"]).cumprod() * 100
            fig = go.Figure()
            for c, color in zip(cum, [CIAN, GRIS1, GRIS3, VERDE]):
                fig.add_scatter(x=cum.index, y=cum[c], name=c, mode="lines",
                                line=dict(color=color, width=3 if c in o["w"] else 1.6, dash="solid" if c in o["w"] else "dot"))
            fig.update_layout(height=380, yaxis_title="Valor de cada 100 invertidos", legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig, width="stretch", theme=None)
            st.caption("Ojo: es una mirada al pasado con los mismos datos usados para calcular los pesos, así que se ve mejor de lo que "
                       "probablemente será en el futuro. Rentabilidades pasadas no garantizan rentabilidades futuras.")

            with st.expander("🔗 ¿Qué activos se mueven juntos? (correlación)"):
                st.markdown("1 = se mueven igual, 0 = no tienen relación, negativo = cuando uno sube el otro tiende a bajar. "
                            "Mezclar activos poco correlacionados es lo que reduce el riesgo.")
                corr = o["ret"].corr()
                etiquetas = [t.replace(".CL", "") for t in corr.index]
                fig = go.Figure(go.Heatmap(z=corr.values, x=etiquetas, y=etiquetas, zmin=-1, zmax=1, colorscale=ESCALA,
                                           text=np.round(corr.values, 2), texttemplate="%{text}"))
                fig.update_layout(height=420)
                st.plotly_chart(fig, width="stretch", theme=None)
            st.caption(f"Riesgo estimado con {o['metodo_cov']}" + (f" (suavizado δ = {o['delta']:.2f})" if o["metodo_cov"] == "Ledoit-Wolf" else "")
                       + f" · rendimiento {o['metodo_mu'].lower()} · máximo {o['cap']:.0%} por activo · tasa libre de riesgo {rf:.2%}.")

# ---------------------------------------------------------------------------
# ¿Cómo funciona?
# ---------------------------------------------------------------------------
with tab4:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### El puntaje de oportunidad")
        st.markdown("Cada acción se compara contra empresas de su mismo sector y recibe una nota de 0 a 100 en cuatro frentes:")
        st.markdown("- **Precio atractivo:** ¿está barata frente a lo que gana, vende y vale en libros?\n"
                    "- **Solidez del negocio:** ¿es rentable, tiene buenos márgenes y poca deuda?\n"
                    "- **Crecimiento:** ¿sus ventas y utilidades están aumentando?\n"
                    "- **Tendencia del precio:** ¿el precio viene subiendo en el último año?")
        st.markdown("Tu estilo de inversionista define cuánto pesa cada frente en la nota final.")
        st.markdown("### Valoración por múltiplos")
        st.markdown("Si empresas parecidas se venden, por ejemplo, a 10 veces sus utilidades y esta se vende a 8, "
                    "podría estar barata. La app hace esa comparación con varios múltiplos y te muestra el precio que tendría "
                    "si se valorara igual que sus pares.")
    with c2:
        st.markdown("### Los dos portafolios")
        st.markdown("**🎯 Máximo Sharpe.** El índice de Sharpe mide cuánto rendimiento extra obtienes por cada unidad de riesgo. "
                    "Este portafolio es la combinación con el mejor índice: la más eficiente.")
        st.markdown("**🛡️ Mínima varianza.** La varianza mide qué tanto se mueve el valor de tu inversión. Este portafolio "
                    "es la combinación que menos se mueve, aprovechando que algunos activos suben cuando otros bajan.")
        st.markdown("### Ten en cuenta")
        st.markdown("- Los cálculos usan el pasado, y el futuro puede ser distinto.\n"
                    "- Las criptomonedas pueden caer más de 50% en pocos meses.\n"
                    "- Yahoo puede tener datos incompletos para algunas acciones colombianas.\n"
                    "- Esta es una herramienta educativa, no una recomendación de inversión.")

    with st.expander("📐 Fórmulas para los curiosos"):
        st.markdown("**Puntaje:** cada métrica se convierte en percentil $p\\in[0,1]$ dentro del sector.")
        st.latex(r"\text{Puntaje}=100\cdot\frac{\sum_b \omega_b\,\bar p_b}{\sum_b \omega_b},\qquad \text{E/P}=\frac{1}{\text{P/E}},\quad \text{FCF yield}=\frac{\text{FCF}}{\text{Cap. mercado}}")
        st.markdown("**Precio implícito por múltiplos** (en múltiplos de EV se descuenta la deuda neta):")
        st.latex(r"P^{\text{impl}}=P\cdot\frac{\tilde m_{\text{pares}}}{m_{\text{propio}}},\qquad P^{\text{impl}}_{EV}=P\cdot\frac{EV\,\tilde m_{\text{pares}}/m_{\text{propio}}-(EV-\text{Cap})}{\text{Cap}}")
        st.markdown("**Parámetros:** $k$ = periodos por año (≈252 con acciones, ≈365 solo con criptos).")
        st.latex(r"\mu=k\,\bar r\ \ \text{o}\ \ \mu_i=r_f+\beta_i\,(E[R_m]-r_f),\qquad \Sigma=k\,\big(\delta F+(1-\delta)S\big),\ \ F=\tfrac{\operatorname{tr}(S)}{N}I")
        st.markdown("**Optimización** (sin ventas en corto, con tope $\\bar w$ por activo, resuelta con SLSQP):")
        st.latex(r"\text{Máx. Sharpe: }\max_w \frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}} \qquad \text{Mín. varianza: }\min_w\ w^\top\Sigma w \qquad \text{s.a. } \mathbf 1^\top w=1,\ 0\le w_i\le\bar w")

st.markdown('<div class="pie">Datos: Yahoo Finance (pueden tener retraso de 15–20 minutos). Herramienta educativa; no constituye asesoría de inversión.</div>',
            unsafe_allow_html=True)
