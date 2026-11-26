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
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf

import fuentes as F
import motor as M
import seguimiento as S
import montecarlo as MC
try:  # módulo de day trading: si falta noise_area.py, el resto de la app sigue funcionando
    import noise_area as NA
except ModuleNotFoundError:
    NA = None
from dataclasses import replace
import grafica_mc as G
import streamlit.components.v1 as components

st.set_page_config(page_title="Radar · Inversión inteligente", page_icon="📡", layout="wide")

# ---------------------------------------------------------------------------
# Identidad visual: azul marino + crema + coral
# ---------------------------------------------------------------------------
NAVY_BASE, NAVY_OSC, NAVY_PANEL = "#14213D", "#0E1830", "#1D2E52"
CREMA, CREMA_SUAVE, CREMA_TENUE = "#F4EDE0", "#D9D2C4", "#9AA3B8"
CORAL, CORAL_SUAVE, CORAL_OSC, ACERO = "#FF6B5B", "#FF9C8F", "#E04A3A", "#7D9BD1"
LINEA = "rgba(244,237,224,0.12)"
# nombres usados en el resto del código
NAVY, PANEL, MUTED, BEIGE = NAVY_OSC, NAVY_PANEL, CREMA_TENUE, CREMA

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Inter Tight', sans-serif !important; font-weight: 500 !important; letter-spacing: -0.025em; }
.stApp { background:
  repeating-linear-gradient(90deg, rgba(244,237,224,0.035) 0 2px, transparent 2px 10px),
  radial-gradient(520px 620px at 72% 22%, rgba(255,107,91,0.32), transparent 70%),
  radial-gradient(700px 520px at 12% 30%, rgba(125,155,209,0.22), transparent 70%),
  linear-gradient(180deg, #1A2B4F 0%, #14213D 45%, #0E1830 100%); background-attachment: fixed; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }

/* encabezado */
.hero { position: relative; padding: 26px 30px 0; margin-bottom: 12px; border-radius: 26px; overflow: hidden;
  border: 1px solid rgba(244,237,224,0.14);
  background: repeating-linear-gradient(90deg, rgba(255,255,255,0.045) 0 3px, transparent 3px 12px),
    radial-gradient(420px 320px at 70% 45%, rgba(255,107,91,0.55), transparent 70%),
    radial-gradient(500px 360px at 20% 30%, rgba(125,155,209,0.35), transparent 70%), #16254A; }
.hero .rutas { display: inline-flex; gap: 4px; padding: 5px; border-radius: 999px; background: rgba(14,24,48,0.55);
  border: 1px solid rgba(244,237,224,0.16); font-size: 13px; margin-bottom: 70px; }
.hero .rutas span { padding: 5px 12px; border-radius: 999px; }
.hero .rutas span:first-child { background: rgba(244,237,224,0.14); }
.hero .rutas span:first-child:before { content: "●"; color: #FF6B5B; font-size: 9px; margin-right: 6px; vertical-align: 2px; }
.titulo { font-family: 'Inter Tight', sans-serif; font-weight: 300; font-size: clamp(40px, 6vw, 76px); line-height: 1.0;
  color: #F4EDE0; margin: 0 0 14px; letter-spacing: -0.045em; max-width: 820px; }
.titulo span { font-weight: 500; color: #FF6B5B; }
.hero p { color: #F4EDE0; opacity: .85; font-size: 16px; max-width: 560px; margin: 0 0 22px; }
.sello { position: absolute; right: 28px; top: 24px; background: #FF6B5B; color: #fff; border-radius: 999px; padding: 6px 14px;
  font: 600 12px 'Inter', sans-serif; box-shadow: 0 8px 22px rgba(255,107,91,0.45); }
.beneficios { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 18px; border-top: 1px solid rgba(244,237,224,0.18);
  margin: 0 -30px; padding: 16px 30px 18px; background: rgba(14,24,48,0.35); }
.beneficios div { font-size: 13px; line-height: 1.35; position: relative; padding-top: 4px; }
.beneficios div:first-child:before { content: ""; position: absolute; top: -17px; left: 0; width: 60%; height: 2px; background: #FF6B5B; }
.beneficios i { display: block; font-style: normal; font-size: 15px; margin-bottom: 4px; }
@media (max-width: 760px) { .hero .rutas { flex-wrap: wrap; border-radius: 18px; } .beneficios { grid-template-columns: repeat(2, minmax(0,1fr)); } .sello { display: none; } .hero .rutas { margin-bottom: 36px; } }

/* tarjetas de vidrio */
.card { background: rgba(29,46,82,0.55); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(244,237,224,0.16); border-radius: 20px; padding: 18px 20px; height: 100%;
  box-shadow: 0 18px 40px rgba(5,10,25,0.35); transition: transform .2s, box-shadow .2s, border-color .2s; }
.card:hover { transform: translateY(-3px); border-color: rgba(255,107,91,0.55); }
.stApp .card, .stApp .card p, .stApp .card li, .stApp .card span, .stApp .card h4, .stApp .card b { color: #F4EDE0; }
.card h4 { margin: 0 0 4px; font-size: 20px; font-weight: 500; }
.stApp .card .sub { color: #9AA3B8; font-size: 13px; margin-bottom: 10px; }
.card .big { font-family: 'Inter Tight', sans-serif; font-size: 34px; font-weight: 400; letter-spacing: -0.03em; }
.stApp .card .de { color: #9AA3B8; font-size: 15px; }
.card ul { margin: 6px 0 0; padding-left: 18px; font-size: 14px; }
.card.destacada { border: 1px solid rgba(255,107,91,0.8);
  background: radial-gradient(260px 200px at 80% 0%, rgba(255,107,91,0.35), transparent 70%), rgba(29,46,82,0.7);
  box-shadow: 0 18px 44px rgba(255,107,91,0.25); }
.card.sharpe { box-shadow: 0 18px 40px rgba(5,10,25,0.35), inset 0 3px 0 #FF6B5B; }
.card.minvar { box-shadow: 0 18px 40px rgba(5,10,25,0.35), inset 0 3px 0 #7D9BD1; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.stApp .badge.b-alta { background: #FF6B5B; color: #FFFFFF; border: 1px solid #FF6B5B; }
.stApp .badge.b-media { background: rgba(244,237,224,0.10); color: #F4EDE0; border: 1px solid rgba(244,237,224,0.35); }
.stApp .badge.b-baja { background: rgba(125,155,209,0.12); color: #AFC2E6; border: 1px solid rgba(125,155,209,0.40); }
.barra { height: 6px; border-radius: 6px; background: rgba(244,237,224,0.12); margin: 3px 0 9px; overflow: hidden; }
.barra > div { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #FF6B5B, #FF9C8F); }
.fila-b { display: flex; justify-content: space-between; font-size: 13px; }

/* textos generales */
.explica { border-left: 2px solid #FF6B5B; background: rgba(255,107,91,0.08); padding: 14px 18px;
  border-radius: 0 14px 14px 0; font-size: 15px; margin: 6px 0 14px; }
.explica.azul { border-left-color: #7D9BD1; background: rgba(125,155,209,0.10); }
.stApp .explica, .stApp .explica * { color: #F4EDE0; } .stApp .explica b { color: #FFFFFF; }
.pie { font-size: 13px; margin-top: 24px; }
.stApp .pie { color: #9AA3B8; }
div[data-testid="stMetric"] { background: rgba(29,46,82,0.55); backdrop-filter: blur(12px); border: 1px solid rgba(244,237,224,0.14);
  border-radius: 18px; padding: 14px 16px; }
div[data-testid="stMetricValue"] { font-family: 'Inter Tight', sans-serif; font-weight: 400; }

/* ---- Colores forzados: la app se ve igual aunque falte .streamlit/config.toml ---- */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp summary, .stApp small,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #F4EDE0; }
.stApp .titulo span { color: #FF6B5B; } .stApp .sello { color: #FFFFFF; }
.stApp .hero .rutas span:first-child:before { color: #FF6B5B; }
.stApp .beneficios div:first-child, .stApp .beneficios div:first-child i { color: #FF6B5B; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: #9AA3B8; }
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricLabel"] * { color: #D9D2C4; }
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricValue"] * { color: #FFFFFF; }
.stApp [data-testid="stMetricDelta"] { background: rgba(255,107,91,0.16) !important; }
.stApp [data-testid="stMetricDelta"] * { color: #FF9C8F !important; fill: #FF9C8F !important; }
div[data-testid="stButtonGroup"] > div { flex-wrap: wrap !important; overflow: visible !important; row-gap: 8px; }
/* botones */
.stApp button[data-testid^="stBaseButton-primary"], .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: #FF6B5B !important; border: 0 !important; border-radius: 999px !important; padding: 0.55rem 1.4rem !important;
  box-shadow: 0 8px 24px rgba(255,107,91,0.40); }
.stApp button[data-testid^="stBaseButton-primary"]:hover { background: #F25848 !important; box-shadow: 0 10px 30px rgba(255,107,91,0.55); }
.stApp button[data-testid^="stBaseButton-primary"] * { color: #FFFFFF !important; font-weight: 600; }
.stApp button[data-testid="stBaseButton-secondary"] { background: rgba(244,237,224,0.06) !important; border: 1px solid rgba(244,237,224,0.30) !important;
  border-radius: 999px !important; }
.stApp button[data-testid="stBaseButton-secondary"] * { color: #F4EDE0 !important; }
/* pills y control segmentado */
.stApp button[data-variant="pills"], .stApp button[data-variant="segmented_control"],
.stApp button[data-testid="stBaseButton-pills"], .stApp button[data-testid="stBaseButton-segmented_control"] {
  background: rgba(14,24,48,0.55) !important; border: 1px solid rgba(244,237,224,0.22) !important; }
.stApp button[data-variant="pills"] *, .stApp button[data-variant="segmented_control"] *,
.stApp button[data-testid="stBaseButton-pills"] *, .stApp button[data-testid="stBaseButton-segmented_control"] * { color: #F4EDE0 !important; }
.stApp button[data-variant="pills"][aria-pressed="true"], .stApp button[data-variant="segmented_control"][aria-checked="true"],
.stApp button[data-testid="stBaseButton-pillsActive"], .stApp button[data-testid="stBaseButton-segmented_controlActive"] {
  background: #FF6B5B !important; border: 1px solid #FF6B5B !important; box-shadow: 0 6px 18px rgba(255,107,91,0.35); }
.stApp button[data-variant="pills"][aria-pressed="true"] *, .stApp button[data-variant="segmented_control"][aria-checked="true"] *,
.stApp button[data-testid="stBaseButton-pillsActive"] *, .stApp button[data-testid="stBaseButton-segmented_controlActive"] * { color: #FFFFFF !important; }
/* radio */
.stApp [data-testid="stRadioOption"] > div > div:first-child { background-color: #0E1830 !important; border: 2px solid rgba(244,237,224,0.45) !important; }
.stApp [data-testid="stRadioOption"] > div > div:first-child > div { background-color: transparent !important; }
.stApp [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child { border-color: #FF6B5B !important; }
.stApp [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div { background-color: #FF6B5B !important; }
/* campos de texto, número y listas */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="select"] > div, .stApp textarea, .stApp input,
.stApp [data-testid="stMultiSelect"] [role="group"], .stApp [data-testid="stSelectbox"] [role="group"], .stApp [data-testid="stSelectbox"] button {
  background-color: #0E1830 !important; color: #F4EDE0 !important; border-color: rgba(244,237,224,0.20) !important; }
.stApp input::placeholder, .stApp textarea::placeholder { color: #7C869C !important; }
.stApp [data-baseweb="select"] svg, .stApp [data-testid="stNumberInput"] button svg { fill: #F4EDE0 !important; color: #F4EDE0 !important; }
.stApp [data-testid="stNumberInput"] button { background-color: #1D2E52 !important; }
.stApp [data-testid="stMultiSelectTagsContainer"] [role="group"] { background: transparent !important; }
.stApp [data-baseweb="tag"], .stApp [data-testid="stMultiSelectTagsContainer"] > span > span {
  background-color: rgba(255,107,91,0.18) !important; border: 1px solid rgba(255,107,91,0.55) !important; border-radius: 999px !important; }
.stApp [data-baseweb="tag"] *, .stApp [data-testid="stMultiSelectTagsContainer"] > span > span * { color: #FFE3DF !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"], [role="listbox"], [role="option"] {
  background-color: #0E1830 !important; color: #F4EDE0 !important; }
div[data-baseweb="popover"] li:hover, [role="option"]:hover, [role="option"][aria-selected="true"], [role="option"][data-focused="true"] {
  background-color: rgba(255,107,91,0.22) !important; }
div[data-baseweb="popover"] *, [role="listbox"] * { color: #F4EDE0 !important; }
/* expanders, formularios, slider, avisos */
.stApp [data-testid="stExpander"] details { background: rgba(14,24,48,0.5); border: 1px solid rgba(244,237,224,0.16) !important; border-radius: 16px; }
.stApp [data-testid="stExpander"] summary:hover * { color: #FF9C8F; }
.stApp [data-testid="stForm"] { border: 1px solid rgba(244,237,224,0.16); background: rgba(14,24,48,0.45); backdrop-filter: blur(10px); border-radius: 20px; }
.stApp [data-testid="stSlider"] [role="slider"] { background-color: #FF6B5B !important; box-shadow: 0 0 0 4px rgba(255,107,91,0.25); }
.stApp [data-testid="stTooltipIcon"] svg { stroke: #9AA3B8 !important; }
.stApp [data-testid="stAlert"] * { color: #F4EDE0; }
.stApp [data-testid="stCheckbox"] label span:first-child { border-color: #FF9C8F !important; }
/* navegación principal: botones grandes */
.stApp [data-testid="stTabs"] [role="tablist"] { gap: 10px; flex-wrap: wrap; border: 0 !important; padding: 8px; margin-bottom: 14px;
  background: rgba(14,24,48,0.55); border-radius: 999px; border: 1px solid rgba(244,237,224,0.14) !important; }
.stApp [data-testid="stTabs"] [role="tab"] { flex: 1 1 120px; min-height: 54px; padding: 10px 12px !important; justify-content: center;
  background: transparent !important; border: 1px solid transparent !important; border-radius: 999px !important; transition: background .2s, box-shadow .2s; }
.stApp [data-testid="stTabs"] [role="tab"] p { font-family: 'Inter Tight', sans-serif !important; font-size: 16.5px !important;
  font-weight: 500 !important; color: #D9D2C4 !important; white-space: nowrap; }
.stApp [data-testid="stTabs"] [role="tab"]:hover { background: rgba(244,237,224,0.08) !important; }
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"], .stApp [data-testid="stTabs"] [role="tab"][data-selected="true"] {
  background: #FF6B5B !important; box-shadow: 0 8px 24px rgba(255,107,91,0.40); }
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] p, .stApp [data-testid="stTabs"] [role="tab"][data-selected="true"] p { color: #FFFFFF !important; }
.stApp [data-testid="stTabs"] .react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
/* encuesta y tarjeta de perfil */
.encuesta, .perfil-card { border: 1px solid rgba(244,237,224,0.16); border-radius: 24px; padding: 22px 26px; margin: 6px 0 14px;
  background: radial-gradient(500px 220px at 100% 0%, rgba(255,107,91,0.22), transparent 70%), rgba(29,46,82,0.55); backdrop-filter: blur(12px); }
.stApp .kicker { color: #FF9C8F; font-size: 13px; font-weight: 600; letter-spacing: .02em; }
.encuesta h3 { margin: 4px 0 6px; font-size: 30px; }
.stApp .encuesta p, .stApp .perfil-card p { color: #F4EDE0; opacity: .88; margin: 0; max-width: 820px; }
.perfil-tipo { font-family: 'Inter Tight', sans-serif; font-size: 42px; font-weight: 400; letter-spacing: -0.03em; margin: 2px 0 6px; }
.stApp .perfil-tipo { color: #FF6B5B; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 8px; }
.stApp .chips span { background: rgba(244,237,224,0.08); border: 1px solid rgba(244,237,224,0.22); border-radius: 999px;
  padding: 5px 12px; font-size: 13px; color: #F4EDE0; }
.stApp .chips span b { color: #FFFFFF; }
.stApp .perfil-card .nota { color: #9AA3B8; font-size: 13px; opacity: 1; }
.stApp [data-testid="stForm"] [data-testid="stRadio"] { background: rgba(29,46,82,0.5); border: 1px solid rgba(244,237,224,0.12);
  border-radius: 16px; padding: 12px 14px; margin-bottom: 6px; width: 100%; box-sizing: border-box; }
.stApp [data-testid="stForm"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) { width: 100% !important; }
.card.reco { height: auto; margin-bottom: 14px; padding: 16px 18px; }
.card.reco.alerta { border-left: 3px solid #FF6B5B; }
.card.reco.sugerencia { border-left: 3px solid #F4EDE0; }
.card.reco.bien { border-left: 3px solid #7D9BD1; }
.stApp .card.reco p { opacity: .9; font-size: 14.5px; }
/* ---- animaciones ---- */
.hero > *:not(.hero-graf):not(.sello) { position: relative; z-index: 1; }
.sello { z-index: 2; }
.hero-graf { position: absolute; right: 26px; top: 64px; width: 47%; height: 230px; z-index: 0; pointer-events: none; overflow: visible; }
.hero-graf .guia { stroke: rgba(244,237,224,0.12); stroke-dasharray: 3 6; }
.hero-graf .linea { fill: none; stroke: #FF6B5B; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: 1000; stroke-dashoffset: 1000; filter: drop-shadow(0 0 8px rgba(255,107,91,0.65));
  animation: trazo 2.6s cubic-bezier(.65,0,.35,1) .35s forwards; }
.hero-graf .linea2 { fill: none; stroke: #7D9BD1; stroke-width: 2; stroke-dasharray: 1000; stroke-dashoffset: 1000; opacity: .7;
  animation: trazo 3.2s cubic-bezier(.65,0,.35,1) .6s forwards; }
.hero-graf .area { opacity: 0; animation: aparecer 1.4s ease 1.7s forwards; }
.hero-graf .punto { opacity: 0; transform-origin: 594px 34px; animation: aparecer .4s ease 2.8s forwards, latido 2.2s ease-in-out 3.2s infinite; }
.hero-graf .halo { opacity: 0; transform-origin: 594px 34px; animation: onda 2.2s ease-out 3.2s infinite; }
.titulo, .hero p { animation: subir .9s cubic-bezier(.2,.7,.2,1) both; }
.hero p { animation-delay: .15s; }
.beneficios div:first-child:before { animation: crecer 1.2s ease 1s both; }
.sello { animation: latido 2.6s ease-in-out 1s infinite; }
.card { animation: subir .7s cubic-bezier(.2,.7,.2,1) both; }
div[data-testid="stMetric"] { animation: subir .6s cubic-bezier(.2,.7,.2,1) both; }
.encuesta, .perfil-card { animation: subir .7s cubic-bezier(.2,.7,.2,1) both; }
.barra > div { transform-origin: left; animation: llenar 1.1s cubic-bezier(.2,.7,.2,1) .3s both; }
@keyframes trazo { to { stroke-dashoffset: 0; } }
@keyframes aparecer { from { opacity: 0; } to { opacity: 1; } }
@keyframes latido { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.08); } }
@keyframes onda { 0% { opacity: .55; transform: scale(1); } 100% { opacity: 0; transform: scale(3.2); } }
@keyframes subir { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
@keyframes crecer { from { width: 0; } to { width: 60%; } }
@keyframes llenar { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (max-width: 1100px) { .hero-graf { width: 70%; opacity: .35; } }
@media (prefers-reduced-motion: reduce) {
  .hero-graf .linea, .hero-graf .linea2 { animation: none; stroke-dashoffset: 0; }
  .hero-graf .area, .hero-graf .punto { animation: none; opacity: 1; } .hero-graf .halo { display: none; }
  .titulo, .hero p, .card, div[data-testid="stMetric"], .encuesta, .perfil-card, .barra > div, .sello,
  .beneficios div:first-child:before { animation: none !important; }
}
/* animación Noise Area */
.na-anim { border: 1px solid rgba(244,237,224,0.14); border-radius: 20px; padding: 10px 14px 4px; margin: 4px 0 18px;
  background: rgba(14,24,48,0.55); }
.na-anim svg { width: 100%; height: auto; max-height: 280px; display: block; }
.na-ejes line { stroke: rgba(244,237,224,0.2); } .na-ejes text { fill: #9AA3B8; font: 12px Inter, sans-serif; }
.na-zona { opacity: 0; animation: aparecer 1s ease .2s forwards; }
.na-borde { fill: none; stroke: #7D9BD1; stroke-width: 1.6; stroke-dasharray: 1; stroke-dashoffset: 1; animation: trazo 1.4s ease .2s forwards; }
.na-tz { fill: #AFC2E6; font: 500 13px Inter, sans-serif; opacity: 0; animation: aparecer .8s ease 1.2s forwards; }
.na-precio { fill: none; stroke: #FF6B5B; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; stroke-dasharray: 1; stroke-dashoffset: 1;
  filter: drop-shadow(0 0 6px rgba(255,107,91,.6)); animation: trazo 3.2s cubic-bezier(.45,0,.3,1) 1.4s forwards; }
.na-stop { fill: none; stroke: #F4EDE0; stroke-width: 1.6; stroke-dasharray: 1; stroke-dashoffset: 1; opacity: .8;
  animation: trazo 2.2s ease 2.8s forwards; }
.na-in, .na-out, .na-st { opacity: 0; }
.na-in { animation: aparecer .5s ease 2.4s forwards; } .na-out { animation: aparecer .5s ease 4.5s forwards; }
.na-st { fill: #F4EDE0; font: 12px Inter, sans-serif; animation: aparecer .6s ease 3.6s forwards; }
.na-in circle { fill: #F4EDE0; stroke: #FF6B5B; stroke-width: 3; } .na-out circle { fill: #FF6B5B; stroke: #F4EDE0; stroke-width: 2; }
.na-in text, .na-out text { fill: #F4EDE0; font: 600 13px Inter, sans-serif; }
@media (prefers-reduced-motion: reduce) { .na-zona, .na-borde, .na-tz, .na-precio, .na-stop, .na-in, .na-out, .na-st { animation: none !important; opacity: 1; stroke-dashoffset: 0; } }
@media (prefers-reduced-motion: reduce) { .card { transition: none; } .card:hover { transform: none; } }
</style>
""", unsafe_allow_html=True)

pio.templates["radar"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=CREMA, size=13),
    colorway=[CORAL, CREMA, ACERO, CORAL_SUAVE, CREMA_TENUE, CORAL_OSC],
    xaxis=dict(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA, automargin=True),
    yaxis=dict(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA, automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(bgcolor=NAVY_OSC, font_color=CREMA),
    margin=dict(l=10, r=10, t=30, b=10)))
pio.templates.default = "radar"
ESCALA = [[0, "#3A4E7A"], [0.5, CREMA], [1, CORAL]]  # bajo → alto

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
# Sector de cada activo de las listas (se usa cuando Yahoo no lo entrega)
_S = {
    "Financial Services": "CIBEST.CL PFCIBEST.CL PFAVAL.CL BOGOTA.CL PFDAVVNDA.CL CORFICOLCF.CL GRUPOSURA.CL BVC.CL BRK-B JPM V MA "
                          "BAC WFC GS MS C BLK SCHW AXP USB PNC CIB AVAL ITUB BBD NU BAP",
    "Energy": "ECOPETROL.CL TERPEL.CL XOM CVX COP EOG SLB OXY PSX MPC VLO EC PBR",
    "Utilities": "GEB.CL CELSIA.CL ISA.CL PROMIGAS.CL",
    "Basic Materials": "MINEROS.CL MSA.TO CEMARGOS.CL VALE SQM CX",
    "Industrials": "GRUPOARGOS.CL TGLS",
    "Technology": "AAPL MSFT NVDA AVGO ORCL CRM ADBE AMD INTC QCOM TXN IBM CSCO NOW INTU MU AMAT",
    "Communication Services": "GOOGL META AMX",
    "Consumer Cyclical": "AMZN TSLA HD MCD NKE SBUX MELI",
    "Consumer Defensive": "PG COST WMT KO PEP FMX",
    "Healthcare": "LLY UNH JNJ MRK ABBV PFE TMO ABT AMGN BMY GILD CVS",
}
SECTORES = {t: s for s, ts in _S.items() for t in ts.split()}
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


_FX: dict = {}


def _fx(de: str, a: str) -> float:
    """Unidades de `a` por 1 unidad de `de` (tasa de Yahoo). Maneja subunidades como GBp."""
    base, factor = MONEDAS_SUB.get(a, (a, 1))
    if de == base:
        return float(factor)
    if (de, base) not in _FX:
        _FX[(de, base)] = F.parse_chart(F.chart(f"{de}{base}=X", "5d"))["precio"]
    return _FX[(de, base)] * factor


def _estados_financieros(t: str) -> tuple[dict | None, str]:
    """Plan B: fundamentales calculados desde los estados financieros de Yahoo (no exige el permiso del resumen)."""
    error = ""
    for simbolo in (t, RESPALDO.get(t)):
        if not simbolo:
            continue
        try:
            ch = F.parse_chart(F.chart(simbolo))
            if not ch["precio"]:
                continue
            if ch["tipo"] == "CRYPTOCURRENCY" or simbolo.endswith("-USD"):
                fila = M.metricas_desde_info(simbolo, {"regularMarketPrice": ch["precio"], "currency": ch["moneda"],
                                                       "shortName": ch["nombre"], "quoteType": "CRYPTOCURRENCY",
                                                       "fiftyTwoWeekHigh": ch["hi52"], "fiftyTwoWeekLow": ch["lo52"]})
                return fila, ""
            ts = F.parse_timeseries(F.timeseries(simbolo))
            if not ts:
                error = "sin estados financieros"
                continue
            return F.calcular_metricas(simbolo, ch, ts, lambda mon: _fx(mon, ch["moneda"])), ""
        except F.LimiteYahoo:
            return None, "limite"
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"[:200]
    return None, error


def _obtener(t: str, estado: dict) -> tuple[str, dict | None, str]:
    # 1) Resumen de Yahoo (más completo: P/E forward, analistas, sector, beta)
    err = ""
    if not estado["bloqueo"].is_set():
        sim, info, err = _info(t, estado["bloqueo"])
        with estado["lock"]:
            estado["ok" if info else "fallas"] += 1
            if estado["fallas"] >= 3 and estado["ok"] == 0:
                estado["bloqueo"].set()  # el resumen no está disponible: pasar directo al plan B
        if info:
            fila = M.metricas_desde_info(sim, info)
            fila["Fuente"] = "Resumen de Yahoo"
            return t, fila, ""
    # 2) Estados financieros de Yahoo
    fila, err2 = _estados_financieros(t)
    if fila:
        fila["Fuente"] = "Precio de Yahoo" if fila["Tipo"] == "Cripto" else "Estados financieros de Yahoo"
        return t, fila, ""
    return t, None, err2 or err


def cargar_fundamentales(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    cache, ahora = _cache_info(), time.time()
    pendientes = [t for t in tickers if t not in cache or ahora - cache[t][0] > 3600]
    errores = {}
    if pendientes:
        estado = {"bloqueo": threading.Event(), "lock": threading.Lock(), "ok": 0, "fallas": 0}
        with ThreadPoolExecutor(max_workers=4) as ex:  # pocas consultas a la vez para no activar el límite de Yahoo
            for t, fila, err in ex.map(lambda x: _obtener(x, estado), pendientes):
                if fila:
                    cache[t] = (ahora, fila)  # solo se guardan los éxitos
                else:
                    errores[t] = err
    filas = [dict(cache[t][1], **{"Con fundamentales": cache[t][1]["Tipo"] == "Acción" or None})
             for t in tickers if t in cache]
    # Plan C: si tampoco hubo estados financieros, al menos el precio
    sin_info = [t for t in tickers if t not in cache]
    if sin_info:
        hist = cargar_precios(tuple(sin_info), "5d")
        for t in sin_info:
            if t in hist and hist[t].notna().any():
                cripto = t.endswith("-USD")
                f = M.metricas_desde_info(t, {"regularMarketPrice": float(hist[t].dropna().iloc[-1]), "shortName": t,
                                              "currency": _moneda_probable(t),
                                              "quoteType": "CRYPTOCURRENCY" if cripto else "EQUITY"})
                f["Con fundamentales"] = None if cripto else False
                f["Fuente"] = "Precio de Yahoo"
                filas.append(f)
    df = pd.DataFrame(filas)
    if len(df):
        df = df[df["Precio"].notna()].drop_duplicates("Ticker")
        conocido = df["Ticker"].map(SECTORES)
        df["Sector"] = np.where((df["Sector"] == "Sin sector") & conocido.notna(), conocido, df["Sector"])
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


def dinero_corto(x: float, moneda: str) -> str:
    """Cifras grandes legibles en tarjetas: 549,8 M COP · 1,2 mil M COP."""
    s = "-" if x < 0 else ""
    v = abs(x)
    if v >= 1e9:
        return f"{s}{v / 1e9:,.2f} mil M {moneda}".replace(",", "X").replace(".", ",").replace("X", ".")
    if v >= 1e6:
        return f"{s}{v / 1e6:,.1f} M {moneda}".replace(",", "X").replace(".", ",").replace("X", ".")
    return dinero(x, moneda)


def dinero(x: float, moneda: str) -> str:
    return f"{x:,.0f} {moneda}" if moneda == "COP" else f"{x:,.2f} {moneda}"


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <svg class="hero-graf" viewBox="0 0 600 240" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="relleno-hero" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FF6B5B" stop-opacity="0.35"/><stop offset="1" stop-color="#FF6B5B" stop-opacity="0"/></linearGradient></defs>
    <line class="guia" x1="0" y1="60" x2="600" y2="60"/><line class="guia" x1="0" y1="130" x2="600" y2="130"/><line class="guia" x1="0" y1="200" x2="600" y2="200"/>
    <polygon class="area" points="0,206 38,198 76,203 114,182 152,188 190,164 228,171 266,146 304,152 342,124 380,131 418,102 456,108 494,78 532,84 570,52 594,34 594,240 0,240" fill="url(#relleno-hero)"/>
    <polyline class="linea2" pathLength="1000" points="0,214 50,208 100,210 150,196 200,199 250,185 300,186 350,170 400,172 450,158 500,152 550,140 594,128"/>
    <polyline class="linea" pathLength="1000" points="0,206 38,198 76,203 114,182 152,188 190,164 228,171 266,146 304,152 342,124 380,131 418,102 456,108 494,78 532,84 570,52 594,34"/>
    <circle class="halo" cx="594" cy="34" r="7" fill="none" stroke="#FF6B5B" stroke-width="2"/>
    <circle class="punto" cx="594" cy="34" r="6.5" fill="#FF6B5B" stroke="#F4EDE0" stroke-width="2"/>
  </svg>
  <div class="sello">Datos en vivo</div>
  <div class="rutas"><span>Oportunidades</span><span>Valoración</span><span>Portafolio</span><span>Seguimiento</span><span>Proyecciones</span><span>Day trading</span></div>
  <div class="titulo">Radar de <span>inversión</span>.</div>
  <p>Encuentra acciones y criptomonedas interesantes, entiende si su precio tiene sentido y descubre
  cómo repartir tu dinero entre ellas. Sin necesidad de ser experto.</p>
  <div class="beneficios">
    <div><i>◎</i>Acciones de Colombia,<br>EE.UU. y Latinoamérica</div>
    <div><i>₿</i>Bitcoin, Ethereum,<br>Solana y XRP</div>
    <div><i>⚖</i>Valoración por<br>múltiplos</div>
    <div><i>◔</i>Portafolio óptimo<br>para tu perfil</div>
  </div>
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

tab1, tab2, tab3, tab_seg, tab_mc, tab_dt, tab4 = st.tabs(["1 · Explora", "2 · Analiza", "3 · Tu portafolio", "4 · Mi seguimiento", "5 · Proyecciones", "6 · Day trading", "¿Cómo funciona?"])

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
                sin_fund = int((fund["Con fundamentales"] == False).sum())  # noqa: E712
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
        n_ef = int((acc.get("Fuente") == "Estados financieros de Yahoo").sum()) if "Fuente" in acc else 0
        st.caption(f"Datos de Yahoo Finance · {st.session_state['fecha']} · {len(acc)} acciones y {len(cri)} criptos"
                   + (f" · {n_ef} acciones valoradas con sus estados financieros (P/E y demás múltiplos calculados por la app)" if n_ef else "")
                   + (f" · sin datos: {', '.join(st.session_state['faltan'])}" if st.session_state["faltan"] else ""))

        if len(df):
            st.markdown("### Las 5 mejores oportunidades según tu estilo")
            cols = st.columns(min(5, len(df)))
            for k, (col, (_, f)) in enumerate(zip(cols, df.head(5).iterrows())):
                col.markdown(f"""<div class="card{' destacada' if k == 0 else ''}" style="animation-delay:{k * 0.09:.2f}s"><div class="sub">{esc(f['Ticker'])} · {esc(f['Sector'])}</div>
                    <h4>{esc(f['Nombre'])[:28]}</h4><div class="big">{f['Puntaje']:.0f}<span class="de">/100</span></div>
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
            fig.add_shape(type="rect", x0=50, x1=100, y0=50, y1=100, fillcolor="rgba(139,92,246,0.10)", line_width=0, layer="below")
            fig.add_annotation(x=98, y=98, text="Baratas y sólidas", showarrow=False, xanchor="right", font=dict(color=CORAL_SUAVE))
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
                    <div class="big">{f['Precio']:,.2f}<span class="de"> USD</span></div>
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
                    <p style="margin:10px 0 0">Frente a empresas similares, esta acción {veredicto}.
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
                                       marker_color=np.where(val["Upside"] >= 0, CORAL, ACERO),
                                       text=[f"{v:,.2f} ({u:+.0%})" for v, u in zip(val["Precio implícito"], val["Upside"])],
                                       textposition="outside", cliponaxis=False))
                fig.add_vline(x=fila["Precio"], line_color=BEIGE, line_width=2, line_dash="dot",
                              annotation_text="precio hoy", annotation_font_color=BEIGE)
                fig.update_layout(height=330, xaxis_title=f"Precio por acción ({fila['Moneda']})", margin=dict(r=120))
                st.plotly_chart(fig, width="stretch", theme=None)
                st.caption("Coral: valdría más que hoy. Azul: valdría menos. Cada barra usa un múltiplo distinto "
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
            colores = {"Máximo Sharpe": CORAL, "Mínima varianza": CREMA}
            columnas = st.columns(2, gap="large")
            for col, (nombre, w) in zip(columnas, o["w"].items()):
                r, v, s = M._stats(w.values, mu.values, cov.values, rf)
                asig = M.asignar_monto(w, o["precios"], o["fx"], o["monto"], o["fracc"])
                asig = asig[asig["Peso objetivo"] > 0].sort_values("Peso objetivo", ascending=False)
                with col:
                    st.markdown(f"<h3>{'🎯' if nombre == 'Máximo Sharpe' else '🛡️'} {nombre} {sello(nombre)}</h3>", unsafe_allow_html=True)
                    pal = ([CORAL, CORAL_SUAVE, CREMA, ACERO, CORAL_OSC, CREMA_TENUE, "#3A4E7A"] if nombre == "Máximo Sharpe"
                           else [CREMA, ACERO, CORAL, CREMA_TENUE, CORAL_SUAVE, "#3A4E7A", CORAL_OSC])
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
                        'más arriba = más rendimiento. La <b>línea crema</b> es la frontera eficiente: las mejores combinaciones posibles '
                        'para cada nivel de riesgo (respetando el máximo por activo). Tus dos portafolios están sobre ella.</div>', unsafe_allow_html=True)
            fr = M.frontera(mu, cov, o["cap"])
            rnd = M.portafolios_aleatorios(mu, cov, rf, cap=o["cap"])
            fig = go.Figure()
            fig.add_scatter(x=rnd["Volatilidad"], y=rnd["Retorno"], mode="markers", name="Combinaciones posibles", hoverinfo="skip",
                            marker=dict(size=4, color=rnd["Sharpe"], colorscale=ESCALA, opacity=0.35, showscale=False))
            fig.add_scatter(x=fr["Volatilidad"], y=fr["Retorno"], mode="lines", name="Frontera eficiente", line=dict(color=CREMA, width=3))
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
            for c, color in zip(cum, [CORAL, CREMA, CREMA_TENUE, ACERO]):
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
# Paso 4: Mi seguimiento (portafolio real)
# ---------------------------------------------------------------------------
EJEMPLO = pd.DataFrame([
    {"Activo": "AAPL", "Fecha": "2024-06-03", "Cantidad": 8, "Precio compra": 193.0, "Comisión": 0.0},
    {"Activo": "ECOPETROL.CL", "Fecha": "2024-09-02", "Cantidad": 1500, "Precio compra": 2350.0, "Comisión": 9000.0},
    {"Activo": "GEB.CL", "Fecha": "2025-02-03", "Cantidad": 2000, "Precio compra": 2300.0, "Comisión": 9000.0},
    {"Activo": "MSFT", "Fecha": "2025-04-07", "Cantidad": 4, "Precio compra": 360.0, "Comisión": 0.0},
    {"Activo": "BTC-USD", "Fecha": "2025-08-11", "Cantidad": 0.01, "Precio compra": 118000.0, "Comisión": 0.0},
])
ICONO = {"alerta": "⚠️", "sugerencia": "💡", "bien": "✅"}


def _serie_fx(de: str, a: str, periodo: str, idx: pd.DatetimeIndex) -> pd.Series:
    """Tasa histórica: unidades de `a` por 1 `de` (con la tasa inversa como respaldo)."""
    if de == a:
        return pd.Series(1.0, index=idx)
    for par, inv in ((f"{de}{a}=X", False), (f"{a}{de}=X", True)):
        h = cargar_precios((par,), periodo)
        if not h.empty:
            s = h.iloc[:, 0].dropna()
            s = 1 / s if inv else s
            return s.reindex(s.index.union(idx)).ffill().bfill().reindex(idx)
    raise ValueError(f"No encontré la tasa de cambio {de}/{a} en Yahoo")


def _registro() -> pd.DataFrame:
    if "trans" not in st.session_state:
        st.session_state["trans"] = S.registro_vacio()
        st.session_state["trans_v"] = 0
    return st.session_state["trans"]


def _guardar_registro(df_nuevo: pd.DataFrame):
    st.session_state["trans"] = df_nuevo.reset_index(drop=True)
    st.session_state["trans_v"] = st.session_state.get("trans_v", 0) + 1  # reinicia la tabla editable


with tab_seg:
    st.markdown('<div class="explica">Registra las inversiones que <b>ya tienes</b>: qué compraste, cuándo, cuánto y a qué precio. '
                'La app calcula cuánto has ganado o perdido, qué tan riesgoso es tu portafolio, lo compara contra el S&P 500 y te '
                'da recomendaciones según tu perfil.</div>', unsafe_allow_html=True)
    base_reg = _registro()

    c_izq, c_der = st.columns([3, 2], gap="large")
    with c_izq:
        st.markdown("#### Agregar una compra")
        conocidos = sorted({t for g in GRUPOS.values() for t in g.split()})
        with st.form("nueva_compra", clear_on_submit=True):
            f1, f2 = st.columns(2)
            activo = f1.selectbox("Activo", conocidos, index=None, accept_new_options=True,
                                  placeholder="Busca o escribe el símbolo (ej. GEB.CL, AAPL, BTC-USD)",
                                  help="Si no está en la lista, escribe el símbolo de Yahoo Finance y presiona Enter.")
            fecha = f2.date_input("Fecha de compra", value=date.today(), max_value=date.today(), format="DD/MM/YYYY")
            f3, f4, f5 = st.columns(3)
            cantidad = f3.number_input("Cantidad (acciones o unidades)", min_value=0.0, value=None, step=1.0, format="%.6g",
                                       placeholder="Ej. 100")
            precio = f4.number_input("Precio por unidad", min_value=0.0, value=None, format="%.6g", placeholder="Vacío = cierre de ese día",
                                     help="En la moneda en que cotiza el activo: COP para acciones colombianas (.CL), USD para EE.UU. y cripto. "
                                          "Si lo dejas vacío, usamos el precio de cierre de ese día.")
            comision = f5.number_input("Comisión (opcional)", min_value=0.0, value=0.0, format="%.6g",
                                       help="Lo que pagaste al comisionista, en la misma moneda del precio.")
            agregar = st.form_submit_button("Agregar al registro", type="primary")
        if agregar:
            if not activo or not cantidad:
                st.warning("Indica al menos el activo y la cantidad.")
            else:
                t = activo.strip().upper()
                p = precio
                if not p:
                    h = cargar_precios((t,), S.periodo_desde(pd.Timestamp(fecha)))
                    if t in h and h[t].notna().any():
                        s = h[t].dropna()
                        p = float(s.asof(pd.Timestamp(fecha))) if pd.Timestamp(fecha) >= s.index[0] else float(s.iloc[0])
                    else:
                        p = None
                if p is None or p != p:
                    st.error(f"No encontré el precio de {t} en Yahoo. Revisa el símbolo o escribe el precio a mano.")
                else:
                    fila = pd.DataFrame([{"Activo": t, "Fecha": pd.Timestamp(fecha), "Cantidad": float(cantidad),
                                          "Precio compra": float(p), "Comisión": float(comision or 0)}])
                    _guardar_registro(pd.concat([st.session_state.get("trans_editado", base_reg), fila], ignore_index=True))
                    st.toast(f"Agregué {cantidad:g} de {t} a tu registro.", icon="✅")
                    st.rerun()
    with c_der:
        st.markdown("#### Guarda o carga tu registro")
        st.markdown('<div class="explica azul">Por privacidad, la app <b>no guarda</b> tus datos en ningún servidor. '
                    'Descarga tu registro al terminar y cárgalo la próxima vez para continuar donde quedaste.</div>',
                    unsafe_allow_html=True)
        archivo = st.file_uploader("Cargar mi registro (CSV)", type=["csv"], label_visibility="collapsed")
        if archivo is not None and st.session_state.get("trans_archivo") != archivo.file_id:
            try:
                cargado = pd.read_csv(archivo)
                faltan = [c for c in ["Activo", "Fecha", "Cantidad", "Precio compra"] if c not in cargado]
                if faltan:
                    st.error(f"Al archivo le faltan columnas: {', '.join(faltan)}.")
                else:
                    _guardar_registro(S.limpiar(cargado))
                    st.session_state["trans_archivo"] = archivo.file_id
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No pude leer el archivo: {e}")
        b1, b2 = st.columns(2)
        if b1.button("Probar con un ejemplo", width="stretch"):
            _guardar_registro(S.limpiar(EJEMPLO))
            st.rerun()
        if b2.button("Borrar todo", width="stretch"):
            _guardar_registro(S.registro_vacio())
            st.rerun()

    st.markdown("#### Tu registro de compras")
    st.caption("Puedes corregir cualquier celda, agregar filas al final o borrar filas (selecciónalas y presiona Supr).")
    editado = st.data_editor(
        st.session_state["trans"], key=f"editor_trans_{st.session_state.get('trans_v', 0)}", num_rows="dynamic",
        hide_index=True, width="stretch",
        column_config={
            "Activo": st.column_config.TextColumn("Activo", help="Símbolo de Yahoo Finance", required=True),
            "Fecha": st.column_config.DateColumn("Fecha de compra", format="DD/MM/YYYY", required=True),
            "Cantidad": st.column_config.NumberColumn("Cantidad", min_value=0.0, format="%.6g", required=True),
            "Precio compra": st.column_config.NumberColumn("Precio por unidad", min_value=0.0, format="%.6g"),
            "Comisión": st.column_config.NumberColumn("Comisión", min_value=0.0, format="%.6g", default=0.0),
        })
    st.session_state["trans_editado"] = editado
    trans = S.limpiar(editado)
    if len(trans):
        st.download_button("Descargar mi registro (CSV)", trans.assign(Fecha=trans["Fecha"].dt.strftime("%Y-%m-%d"))
                           .to_csv(index=False).encode("utf-8"), "mi_portafolio.csv", "text/csv", icon="💾")

    if trans.empty:
        st.info("Agrega tu primera compra o pulsa «Probar con un ejemplo» para ver cómo funciona.")
    else:
        mon_rep = st.segmented_control("Ver resultados en", ["COP", "USD", "EUR"], default="COP", key="mon_seg") or "COP"
        try:
            with st.spinner("Calculando el desempeño de tu portafolio…"):
                activos = tuple(sorted(trans["Activo"].unique()))
                periodo = S.periodo_desde(trans["Fecha"].min())
                hist = cargar_precios(activos + (BENCH,), periodo)
                sin_hist = [a for a in activos if a not in hist or hist[a].notna().sum() < 2]
                if sin_hist:
                    st.warning(f"No encontré precios de {', '.join(sin_hist)} en Yahoo; los dejo por fuera del análisis.")
                trans_ok = trans[~trans["Activo"].isin(sin_hist)].copy()
                if trans_ok.empty:
                    raise ValueError("No hay activos con precios disponibles.")
                activos = tuple(sorted(trans_ok["Activo"].unique()))
                # precio de compra vacío -> cierre de ese día
                for i, l in trans_ok[trans_ok["Precio compra"].isna()].iterrows():
                    s = hist[l["Activo"]].dropna()
                    trans_ok.at[i, "Precio compra"] = float(s.asof(l["Fecha"])) if l["Fecha"] >= s.index[0] else float(s.iloc[0])
                fund_h, _ = cargar_fundamentales(activos)
                info_h = fund_h.set_index("Ticker") if len(fund_h) else pd.DataFrame()
                monedas, divisor = {}, {}
                for a in activos:
                    cur = info_h.loc[a, "Moneda"] if a in info_h.index else _moneda_probable(a)
                    base_cur, d = MONEDAS_SUB.get(cur, (cur, 1))
                    monedas[a], divisor[a] = base_cur, d
                precios_h = hist[list(activos)].copy()
                for a in activos:
                    precios_h[a] = precios_h[a] / divisor[a]
                    if divisor[a] != 1:
                        trans_ok.loc[trans_ok["Activo"] == a, ["Precio compra", "Comisión"]] /= divisor[a]
                idx = precios_h.index
                fx_h = pd.DataFrame({a: _serie_fx(monedas[a], mon_rep, periodo, idx) for a in activos})
                ev = S.evolucion(trans_ok, precios_h, fx_h)
                precio_hoy = pd.Series({a: precios_h[a].dropna().iloc[-1] for a in activos})  # mismo dato que el gráfico
                fx_hoy = fx_h.ffill().iloc[-1]
                pos = S.posiciones(ev["lotes"], precio_hoy, fx_hoy)
                pos["Nombre"] = pos["Activo"].map(lambda a: info_h.loc[a, "Nombre"] if a in info_h.index else a)
                pos["Sector"] = pos["Activo"].map(lambda a: info_h.loc[a, "Sector"] if a in info_h.index else SECTORES.get(a, "Sin sector"))
                pos["Tipo"] = pos["Activo"].map(lambda a: "Cripto" if a.endswith("-USD") or (a in info_h.index and info_h.loc[a, "Tipo"] == "Cripto") else "Acción")
                pos["Moneda"] = pos["Activo"].map(monedas)
                dfp = st.session_state.get("df")
                pos["Puntaje"] = pos["Activo"].map(dfp.set_index("Ticker")["Puntaje"]) if dfp is not None and len(dfp) else np.nan
                # índice de comparación en moneda de reporte
                indice = None
                if BENCH in hist:
                    indice = hist[BENCH].reindex(idx).ffill() * _serie_fx("USD", mon_rep, periodo, idx)
                    bench_val = S.simular_indice(ev["lotes"], indice, ev["valor"].index)
                valor_hoy = float(pos["Valor actual"].sum())
                flujos = [(r["Día efectivo"], -r["Costo reporte"]) for _, r in ev["lotes"].iterrows()]
                tir = S.xirr(flujos + [(pd.Timestamp(date.today()), valor_hoy)])
                tir_ind = S.xirr(flujos + [(pd.Timestamp(date.today()), float(bench_val.iloc[-1]))]) if indice is not None else np.nan
                rf_s = tasa_libre_riesgo() or 0.04
                riesgo = S.metricas_riesgo(ev["r"], rf_s, indice.pct_change(fill_method=None) if indice is not None else None)
                ret_act = precios_h[list(activos)].ffill().pct_change(fill_method=None).dropna(how="all").tail(756)
                corr = ret_act.corr() if len(activos) > 1 else None
                corr_media = float(corr.values[np.triu_indices(len(corr), 1)].mean()) if corr is not None else None
        except Exception as e:  # noqa: BLE001
            st.error(f"No pude calcular el desempeño: {e}")
            ev = None

        if ev is not None:
            invertido = float(pos["Invertido"].sum())
            ganancia = valor_hoy - invertido
            dias = (pd.Timestamp(date.today()) - trans_ok["Fecha"].min()).days
            st.markdown(f"### Así va tu portafolio")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Valor actual", dinero_corto(valor_hoy, mon_rep), help=dinero(valor_hoy, mon_rep))
            k2.metric("Ganancia o pérdida", dinero_corto(ganancia, mon_rep), f"{ganancia / invertido:+.1%}", help=f"Invertido: {dinero(invertido, mon_rep)}")
            if dias >= 90 and tir == tir:
                k3.metric("Rentabilidad anual", f"{tir:.1%}", help="Tasa interna de retorno (XIRR): tiene en cuenta cuándo y cuánto invertiste.")
            else:
                k3.metric("Rentabilidad", f"{ganancia / invertido:.1%}", help="Con menos de 3 meses no anualizamos para no exagerar.")
            if tir_ind == tir_ind and dias >= 90:
                k4.metric("S&P 500 con las mismas compras", f"{tir_ind:.1%} anual", f"{(tir - tir_ind) * 100:+.1f} pts a tu favor" if tir >= tir_ind
                          else f"{(tir - tir_ind) * 100:+.1f} pts", delta_color="normal")
            elif indice is not None:
                k4.metric("S&P 500 con las mismas compras", dinero_corto(float(bench_val.iloc[-1]), mon_rep))
            if riesgo:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Altibajos (volatilidad)", f"{riesgo['Volatilidad']:.1%}", help="Qué tanto se mueve el valor de tu portafolio en un año típico.")
                r2.metric("Sharpe", f"{riesgo['Sharpe']:.2f}", help="Rendimiento extra sobre la tasa libre de riesgo por cada unidad de riesgo.")
                r3.metric("Peor caída desde un máximo", f"{riesgo['Caída máxima']:.1%}", help="La mayor pérdida desde un pico hasta un valle.")
                r4.metric("Beta frente al S&P 500", f"{riesgo['Beta']:.2f}" if "Beta" in riesgo else "—",
                          help="1 = se mueve como el mercado; mayor que 1 = más sensible; menor que 1 = más defensivo.")
            efecto_fx = float(pos["Efecto cambiario"].sum())
            if abs(efecto_fx) > 0.005 * invertido:
                st.caption(f"De tu resultado, {dinero(float(pos['Efecto precio'].sum()), mon_rep)} vienen del precio de los activos y "
                           f"{dinero(efecto_fx, mon_rep)} del cambio en la tasa de cambio (al medir en {mon_rep}).")

            st.markdown("#### Evolución de tu dinero")
            fig = go.Figure()
            fig.add_scatter(x=ev["invertido"].index, y=ev["invertido"], name="Lo que has invertido", mode="lines",
                            line=dict(color=CREMA_TENUE, width=2, dash="dot", shape="hv"))
            if indice is not None:
                fig.add_scatter(x=bench_val.index, y=bench_val, name="Si lo hubieras puesto en el S&P 500", mode="lines",
                                line=dict(color=ACERO, width=2))
            fig.add_scatter(x=ev["valor"].index, y=ev["valor"], name="Tu portafolio", mode="lines",
                            line=dict(color=CORAL, width=3), fill="tozeroy", fillcolor="rgba(255,107,91,0.08)")
            fig.update_layout(height=400, yaxis_title=mon_rep, legend=dict(orientation="h", y=-0.15), hovermode="x unified")
            st.plotly_chart(fig, width="stretch", theme=None)

            st.markdown("#### Tus posiciones")
            tabla = pos.assign(**{"Activo": pos["Nombre"].astype(str) + " · " + pos["Activo"]})[
                ["Activo", "Cantidad", "Moneda", "Costo promedio", "Precio actual", "Invertido", "Valor actual", "Ganancia", "Rentabilidad", "Peso"]]
            st.dataframe(tabla, hide_index=True, width="stretch", column_config={
                "Cantidad": st.column_config.NumberColumn(format="%.6g"),
                "Costo promedio": st.column_config.NumberColumn("Costo promedio (por unidad)", format="localized"),
                "Precio actual": st.column_config.NumberColumn(format="localized"),
                "Invertido": st.column_config.NumberColumn(f"Invertido ({mon_rep})", format="localized"),
                "Valor actual": st.column_config.NumberColumn(f"Valor hoy ({mon_rep})", format="localized"),
                "Ganancia": st.column_config.NumberColumn(f"Ganancia ({mon_rep})", format="localized"),
                "Rentabilidad": st.column_config.NumberColumn(format="percent"),
                "Peso": st.column_config.ProgressColumn("Peso", min_value=0, max_value=1, format="percent")})

            g1, g2 = st.columns(2, gap="large")
            pal = [CORAL, CREMA, ACERO, CORAL_SUAVE, CREMA_TENUE, CORAL_OSC, "#3A4E7A", "#C9B8A0"]
            for col, campo, titulo in ((g1, "Activo", "Por activo"), (g2, "Sector", "Por sector")):
                comp = pos.groupby(campo)["Valor actual"].sum().sort_values(ascending=False)
                fig = go.Figure(go.Pie(labels=[str(x).replace(".CL", "") for x in comp.index], values=comp.values, hole=0.6, sort=False,
                                       marker=dict(colors=pal * 3, line=dict(color=NAVY, width=2)), textinfo="label+percent",
                                       textposition="outside", hovertemplate="%{label}: %{percent}<extra></extra>"))
                fig.update_layout(height=320, showlegend=False, margin=dict(t=40, b=30, l=40, r=40),
                                  title=dict(text=titulo, x=0.5, font=dict(size=15)))
                col.plotly_chart(fig, width="stretch", theme=None, key=f"pie_seg_{campo}")

            # Recomendaciones
            st.markdown("### Recomendaciones para mejorar tu portafolio")
            if not perfil:
                st.caption("Haz la encuesta de perfil (arriba) para recibir recomendaciones ajustadas a ti; por ahora usamos un perfil moderado.")
            recs = S.recomendaciones(pos, riesgo, perfil, corr_media, tir if dias >= 90 else None,
                                     tir_ind if dias >= 90 else None, mon_rep)
            cols_r = st.columns(2, gap="large")
            for i, r in enumerate(recs):
                cols_r[i % 2].markdown(f'<div class="card reco {r["nivel"]}"><h4>{ICONO[r["nivel"]]} {esc(r["titulo"])}</h4>'
                                       f'<p style="margin:6px 0 0">{esc(r["texto"])}</p></div>', unsafe_allow_html=True)

            # Rebalanceo sugerido
            if len(activos) >= 2 and len(ret_act) > 120:
                tipo_p = perfil["recomendado"] if perfil else "Máximo Sharpe"
                cap_p = max(perfil["cap"] if perfil else 0.30, 1 / len(activos))
                try:
                    ret_r = M.retornos_diarios(precios_h[list(activos)].tail(800),
                                               {a for a in activos if a.endswith("-USD")})
                    mu_r, cov_r, _, _ = M.estimar_parametros(ret_r, "Ledoit-Wolf", "Histórica", rf_s, None, 0.055, M.periodos_por_anio(ret_r))
                    w_obj = pd.Series(M.max_sharpe(mu_r, cov_r, rf_s, cap_p) if tipo_p == "Máximo Sharpe"
                                      else M.min_varianza(mu_r, cov_r, cap_p), mu_r.index)
                    reb = pos.set_index("Activo")[["Peso", "Valor actual"]].reindex(w_obj.index)
                    reb["Peso sugerido"] = w_obj
                    reb["Ajuste"] = (reb["Peso sugerido"] - reb["Peso"]) * valor_hoy
                    reb["Qué hacer"] = np.where(reb["Ajuste"] > 0.01 * valor_hoy, "Comprar más",
                                                np.where(reb["Ajuste"] < -0.01 * valor_hoy, "Reducir", "Mantener"))
                    st.markdown(f"#### Rebalanceo hacia el portafolio de {tipo_p.lower()} con tus mismos activos")
                    st.markdown('<div class="explica azul">Así quedaría tu portafolio si redistribuyeras el dinero entre los mismos activos '
                                f'buscando el portafolio de <b>{tipo_p.lower()}</b> (máximo {cap_p:.0%} por activo). Una forma sencilla de '
                                'acercarte sin vender es dirigir tus próximas compras a los activos marcados como «Comprar más».</div>',
                                unsafe_allow_html=True)
                    st.dataframe(reb.reset_index(names="Activo")[["Activo", "Peso", "Peso sugerido", "Ajuste", "Qué hacer"]],
                                 hide_index=True, width="stretch", column_config={
                                     "Peso": st.column_config.NumberColumn("Peso actual", format="percent"),
                                     "Peso sugerido": st.column_config.NumberColumn(format="percent"),
                                     "Ajuste": st.column_config.NumberColumn(f"Ajuste aproximado ({mon_rep})", format="localized")})
                except Exception:  # noqa: BLE001
                    pass
            st.caption("Estas recomendaciones son orientativas y se basan en reglas generales y datos históricos; no son asesoría "
                       "financiera profesional. Rentabilidades pasadas no garantizan rentabilidades futuras.")

# ---------------------------------------------------------------------------
# Paso 5: Proyecciones Monte Carlo
# ---------------------------------------------------------------------------
HORIZONTES = {"1 mes": 1 / 12, "3 meses": 0.25, "6 meses": 0.5, "1 año": 1.0, "2 años": 2.0}

with tab_mc:
    st.markdown('<div class="explica"><b>¿Cuánto podría valer un activo en el futuro?</b> Nadie lo sabe con certeza, pero sí podemos '
                'estimar qué tan probable es cada escenario. La simulación Monte Carlo genera <b>miles de caminos posibles</b> del precio '
                'usando cómo se ha movido en el pasado (su tendencia y sus altibajos) y luego cuenta cuántos terminan en cada nivel. '
                'El <b>precio objetivo</b> es el valor del escenario del medio: la mitad de los caminos termina por encima y la mitad por debajo.</div>',
                unsafe_allow_html=True)
    base_mc = st.session_state.get("fund")
    df_mc = st.session_state.get("df")
    opciones_mc = list(dict.fromkeys(([] if df_mc is None else df_mc["Ticker"].tolist())
                                     + ([] if base_mc is None else base_mc["Ticker"].tolist())
                                     + sorted({t for g in GRUPOS.values() for t in g.split()})))
    with st.form("form_mc"):
        m1, m2, m3 = st.columns([2, 2, 1])
        activo_mc = m1.selectbox("Activo", opciones_mc, index=0, accept_new_options=True,
                                 help="Acciones o criptomonedas. Si no está en la lista, escribe su símbolo de Yahoo Finance.")
        horizonte = m2.select_slider("¿A cuánto tiempo?", list(HORIZONTES), value="1 año")
        n_sim = m3.selectbox("Simulaciones", [1000, 5000, 10000], index=1, format_func=lambda n: f"{n:,}".replace(",", "."))
        with st.expander("⚙️ Opciones del modelo"):
            o1, o2, o3 = st.columns(3)
            modelo = o1.radio("Modelo", ["Bootstrap", "GBM"], format_func=lambda m: {
                "Bootstrap": "Bootstrap histórico (recomendado)", "GBM": "Movimiento browniano geométrico"}[m],
                help="Bootstrap reutiliza días reales del pasado, incluidos los días extremos. GBM supone que los retornos siguen una curva normal.")
            tendencia = o2.radio("Tendencia de precio", ["Moderada", "Histórica", "Sin tendencia"], help=(
                "Histórica: supone que seguirá subiendo/bajando al mismo ritmo del pasado (suele exagerar). "
                "Moderada: mitad histórica y mitad tasa libre de riesgo. Sin tendencia: solo la tasa libre de riesgo."))
            hist_mc = o3.selectbox("Historia usada", ["1y", "3y", "5y"], index=1,
                                   format_func=lambda p: {"1y": "1 año", "3y": "3 años", "5y": "5 años"}[p])
            objetivo_mc = st.number_input("¿Tienes un precio en mente? (opcional)", min_value=0.0, value=None, format="%.6g",
                                          placeholder="Ej. el precio al que te gustaría vender",
                                          help="Calculamos la probabilidad de que el precio lo alcance.")
        correr = st.form_submit_button("Simular escenarios", type="primary", icon="🎲")

    if correr:
        t_mc = (activo_mc or "").strip().upper()
        with st.spinner(f"Simulando {n_sim:,} escenarios para {t_mc}…".replace(",", ".")):
            try:
                h = cargar_precios((t_mc,), hist_mc)
                if t_mc not in h or h[t_mc].notna().sum() < 120:
                    raise ValueError(f"No hay suficiente historia de precios de {t_mc} en Yahoo (se necesitan al menos 6 meses).")
                par = MC.parametros(h[t_mc])
                rf_mc = tasa_libre_riesgo() or 0.04
                sim_mc = MC.simular(par, HORIZONTES[horizonte], int(n_sim), modelo, tendencia, rf_mc)
                res_mc = MC.resumen(sim_mc, par["s0"], objetivo_mc or None)
                info_mc = base_mc.set_index("Ticker").loc[t_mc] if base_mc is not None and t_mc in set(base_mc["Ticker"]) else None
                st.session_state["mc"] = dict(
                    t=t_mc, horizonte=horizonte, par={k: v for k, v in par.items() if k != "r"}, res=res_mc, modelo=modelo,
                    tendencia=tendencia, n=int(n_sim), rf=rf_mc, anim=MC.datos_animacion(sim_mc, res_mc),
                    moneda=(info_mc["Moneda"] if info_mc is not None else _moneda_probable(t_mc)),
                    nombre=(info_mc["Nombre"] if info_mc is not None else t_mc))
            except Exception as e:  # noqa: BLE001
                st.error(f"No pude simular: {e}")

    mc = st.session_state.get("mc")
    if not mc:
        st.info("Elige un activo y pulsa «Simular escenarios». La gráfica se anima mostrando cómo se generan los escenarios.")
    else:
        res, par, mon = mc["res"], mc["par"], mc["moneda"]
        fmtp = lambda v: f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:,.2f}" if abs(v) >= 1 else f"{v:,.5f}")
        st.markdown(f"### {esc(mc['nombre'])} · en {mc['horizonte']}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Precio hoy", f"{fmtp(res['s0'])} {mon}")
        k2.metric("Precio objetivo (mediana)", f"{fmtp(res['mediana'])} {mon}", f"{res['ret_mediano']:+.1%}",
                  help="La mitad de los escenarios termina por encima de este precio y la mitad por debajo.")
        k3.metric("Escenario pesimista (P5)", f"{fmtp(res['pct'][5])} {mon}", f"{res['pct'][5] / res['s0'] - 1:+.1%}",
                  help="Solo 5 de cada 100 escenarios terminan por debajo de este precio.")
        k4.metric("Escenario optimista (P95)", f"{fmtp(res['pct'][95])} {mon}", f"{res['pct'][95] / res['s0'] - 1:+.1%}",
                  help="Solo 5 de cada 100 escenarios terminan por encima de este precio. Entre P5 y P95 queda el 90% de los casos.")

        html_mc = G.html_simulacion(mc["anim"], f"{esc(mc['t'])} · <b>{mc['horizonte']}</b>", mon, mc["n"], 470)
        if hasattr(st, "iframe"):  # Streamlit reciente
            st.iframe(html_mc, height=476)
        else:  # versiones anteriores
            components.html(html_mc, height=476)

        probs = [("Que termine más alto que hoy", res["prob_sube"]), ("Que suba más de 20%", res["prob_sube20"]),
                 ("Que caiga más de 20%", res["prob_cae20"])]
        if res.get("objetivo"):
            probs += [(f"Que termine {'por encima' if res['objetivo'] >= res['s0'] else 'por debajo'} de tu precio ({fmtp(res['objetivo'])})",
                       res["prob_objetivo"]),
                      ("Que lo toque en algún momento del periodo", res["prob_toca_objetivo"])]
        st.markdown("#### Probabilidades")
        cols_p = st.columns(len(probs))
        for k, (col, (txt, pr)) in enumerate(zip(cols_p, probs)):
            col.markdown(f'<div class="card" style="animation-delay:{k * 0.08:.2f}s"><div class="sub">{esc(txt)}</div>'
                         f'<div class="big">{pr:.0%}</div><div class="barra"><div style="width:{max(pr * 100, 1.5):.1f}%"></div></div></div>',
                         unsafe_allow_html=True)

        # Comparar con otras formas de estimar el precio objetivo
        filas_po = [{"Método": "Monte Carlo (mediana)", "Precio objetivo": res["mediana"], "Horizonte": mc["horizonte"]}]
        info_po = st.session_state.get("fund")
        if info_po is not None and mc["t"] in set(info_po["Ticker"]):
            fi = info_po.set_index("Ticker").loc[mc["t"]]
            if pd.notna(fi.get("Upside analistas", np.nan)):
                filas_po.append({"Método": "Consenso de analistas", "Precio objetivo": fi["Precio"] * (1 + fi["Upside analistas"]),
                                 "Horizonte": "12 meses"})
        if df_mc is not None and mc["t"] in set(df_mc["Ticker"]):
            sect = df_mc.set_index("Ticker").loc[mc["t"], "Sector"]
            pares_mc = df_mc.loc[(df_mc["Sector"] == sect) & (df_mc["Ticker"] != mc["t"]), "Ticker"].tolist()
            if pares_mc:
                vm = M.valoracion_multiplos(df_mc, mc["t"], pares_mc)
                if len(vm):
                    p_hoy_df = df_mc.set_index("Ticker").loc[mc["t"], "Precio"]
                    filas_po.append({"Método": "Valoración por múltiplos (pares)",
                                     "Precio objetivo": vm["Precio implícito"].median() * res["s0"] / p_hoy_df, "Horizonte": "Valor justo hoy"})
        if len(filas_po) > 1:
            st.markdown("#### Tres formas de estimar el precio objetivo")
            po = pd.DataFrame(filas_po)
            po["Frente a hoy"] = po["Precio objetivo"] / res["s0"] - 1
            st.dataframe(po, hide_index=True, width="stretch", column_config={
                "Precio objetivo": st.column_config.NumberColumn(f"Precio objetivo ({mon})", format="localized"),
                "Frente a hoy": st.column_config.NumberColumn(format="percent")})
            st.caption("Si los tres métodos apuntan en la misma dirección, la señal es más sólida. Si se contradicen, conviene más cautela.")

        st.caption(f"Modelo: {'bootstrap histórico' if mc['modelo'] == 'Bootstrap' else 'movimiento browniano geométrico'} · "
                   f"tendencia {mc['tendencia'].lower()} · volatilidad histórica {par['sigma']:.1%} anual · rendimiento histórico "
                   f"{par['mu']:.1%} anual · {par['n_obs']} días de datos · en el 5% de los peores escenarios la pérdida supera "
                   f"{res['var95']:.0%} (VaR 95%).")
        with st.expander("📐 ¿Cómo funciona la simulación?"):
            st.markdown("Se estiman la tendencia $\\mu$ y la volatilidad $\\sigma$ con los retornos diarios del pasado. Luego, para cada "
                        "escenario, el precio avanza día a día:")
            st.latex(r"S_{t+\Delta t}=S_t\,\exp\!\Big[\big(\mu-\tfrac{\sigma^2}{2}\big)\Delta t+\sigma\sqrt{\Delta t}\;Z_t\Big],\qquad Z_t\sim N(0,1)")
            st.markdown("En el **bootstrap**, en lugar de $Z_t$ se usa un retorno real tomado al azar de la historia, lo que conserva los "
                        "días extremos. La **tendencia moderada** usa $\\mu^{*}=\\tfrac12\\mu_{\\text{hist}}+\\tfrac12 r_f$. Con miles de "
                        "escenarios se calculan los percentiles del precio final:")
            st.latex(r"\text{Precio objetivo}=P_{50}(S_T),\qquad \Pr(S_T>S_0)=\frac{\#\{\,i:\,S_T^{(i)}>S_0\,\}}{N}")
            st.markdown("**Limitación importante:** el modelo supone que el futuro se parecerá estadísticamente al pasado. No anticipa "
                        "noticias, crisis ni cambios en la empresa. Úsalo como un mapa de escenarios posibles, no como una predicción exacta.")

# ---------------------------------------------------------------------------
# Paso 6: Day trading · Backtest Noise Area en MES
# ---------------------------------------------------------------------------
HORAS_NA = [f"{(570 + 30 * (k + 1)) // 60:02d}:{(570 + 30 * (k + 1)) % 60:02d}" for k in range(13)]

ANIM_NA = """
<div class="na-anim">
<svg viewBox="0 0 720 260" preserveAspectRatio="xMidYMid meet" aria-label="Cómo funciona la zona de ruido">
  <defs><linearGradient id="na-zona" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#7D9BD1" stop-opacity=".30"/><stop offset=".5" stop-color="#7D9BD1" stop-opacity=".12"/>
    <stop offset="1" stop-color="#7D9BD1" stop-opacity=".30"/></linearGradient></defs>
  <g class="na-ejes"><line x1="40" y1="225" x2="700" y2="225"/>
    <text x="40" y="246">9:30</text><text x="200" y="246">11:00</text><text x="360" y="246">12:30</text>
    <text x="520" y="246">14:00</text><text x="672" y="246">16:00</text></g>
  <path class="na-zona" d="M40,130 C160,100 320,82 700,62 L700,198 C320,178 160,160 40,130 Z" fill="url(#na-zona)"/>
  <path class="na-borde" pathLength="1" d="M40,130 C160,100 320,82 700,62"/>
  <path class="na-borde" pathLength="1" d="M40,130 C160,160 320,178 700,198"/>
  <text class="na-tz" x="470" y="128">Zona de ruido: aquí no se opera</text>
  <polyline class="na-precio" pathLength="1" points="40,130 70,122 95,138 120,118 150,134 175,112 200,126 228,104 250,92 275,84 300,78 330,70 360,74 390,60 420,66 450,52 480,58 510,46 540,52 570,40 600,46 630,36 660,42 690,34"/>
  <polyline class="na-stop" pathLength="1" points="250,104 300,96 360,90 420,82 480,76 540,70 600,64 660,58 690,56"/>
  <g class="na-in"><circle cx="250" cy="92" r="7"/><text x="210" y="74">Rompe la banda: compra</text></g>
  <g class="na-out"><circle cx="690" cy="34" r="7"/><text x="560" y="20">16:00 · cierra todo</text></g>
  <text class="na-st" x="430" y="104">Stop que sube con el VWAP</text>
</svg></div>
"""


def _barras_csv_na(contenido: bytes, zona: str):
    return NA.barras_rth_30m(NA.cargar_csv(contenido, zona))


_barras_csv_na = st.cache_data(show_spinner=False)(_barras_csv_na)


@st.cache_data(show_spinner=False, ttl=3600)
def _barras_yahoo_na():
    return NA.barras_rth_30m(NA.cargar_yahoo("MES=F", "30m", "60d"))


@st.cache_data(show_spinner=False)
def _barras_ejemplo_na():
    return NA.barras_rth_30m(NA.datos_ejemplo())


with tab_dt:
  if NA is None:
    st.error("Falta el archivo **noise_area.py** en el repositorio de GitHub. Súbelo al mismo nivel que app.py "
             "(junto a motor.py y fuentes.py) y reinicia la app desde Manage app → Reboot app.")
  else:
      st.markdown('<div class="explica"><b>Momentum intradía "Noise Area"</b> (Zarattini, Aziz y Barbon, 2024, <i>Beat the Market</i>) '
                  'aplicado a los micro-futuros del S&amp;P 500 (<b>MES</b>, 5 USD por punto). Cada día se dibuja una '
                  '<b>zona de ruido</b> alrededor del precio de apertura, con el tamaño de los movimientos normales de los últimos 14 días. '
                  'Mientras el precio se mueve dentro de esa zona, es ruido y no se opera. Cuando la rompe, hay <b>momentum</b>: '
                  'se entra a favor del movimiento, con un stop que sigue al VWAP, y todo se cierra antes de las 4:00 p. m. de Nueva York. '
                  'Aquí puedes probarla con datos históricos y simular si pasaría la evaluación de una prop firm.</div>',
                  unsafe_allow_html=True)
      st.markdown(ANIM_NA, unsafe_allow_html=True)

      with st.form("form_na"):
          st.markdown("#### 1 · Datos")
          d1, d2 = st.columns([2, 3], gap="large")
          fuente_na = d1.radio("Fuente", ["Ejemplo", "CSV", "Yahoo"], format_func=lambda f: {
              "Ejemplo": "Datos de ejemplo (simulados)", "CSV": "Mi archivo CSV (recomendado)",
              "Yahoo": "Yahoo Finance (MES=F, ~60 días)"}[f])
          with d2:
              archivo_na = st.file_uploader("CSV de MES o ES en barras de 1, 5, 15 o 30 min", type=["csv", "txt"],
                                            help="Columnas: datetime/timestamp (o date + time), open, high, low, close, volume. "
                                                 "Puedes exportarlo de NinjaTrader, TradingView, Databento o FirstRate Data.")
              zona_na = st.selectbox("Zona horaria de los datos del CSV", ["America/New_York", "UTC", "America/Chicago", "America/Bogota"])
          st.caption("Los datos de ejemplo son inventados: sirven para ver cómo funciona la herramienta, no para evaluar la estrategia. "
                     "Yahoo solo entrega unos 60 días de barras de 30 minutos. Para sacar conclusiones usa un CSV de 2 años o más.")

          st.markdown("#### 2 · Prop firm")
          p1, p2 = st.columns([1, 3], gap="large")
          preset_na = p1.selectbox("Reglas de evaluación", list(NA.PRESETS), index=1)
          base_na = NA.PRESETS[preset_na]
          with p2.expander("Ajustar reglas (verifícalas en la página oficial de la firma)", expanded=preset_na == "Personalizada"):
              r1, r2, r3 = st.columns(3)
              meta_na = r1.number_input("Meta de ganancia (USD)", 0.0, value=float(base_na.meta), step=250.0)
              mll_na = r2.number_input("Pérdida máxima trailing EOD (USD)", 0.0, value=float(base_na.mll), step=250.0)
              bloq_na = r3.number_input("El trailing se congela en inicial + (USD)", value=float(base_na.bloqueo), step=50.0)
              r4, r5, r6 = st.columns(3)
              dll_na = r4.number_input("Límite de pérdida diaria (0 = no tiene)", 0.0, value=float(base_na.dll), step=100.0)
              cons_na = r5.slider("Consistencia: mejor día ≤ % de la ganancia", 0, 100, int(base_na.consistencia * 100), 5)
              micros_na = r6.number_input("Máximo de micros", 1, 200, int(base_na.max_contratos_micro))

          st.markdown("#### 3 · Estrategia, tamaño y costos")
          e1, e2, e3, e4 = st.columns(4)
          riesgo_na = e1.number_input("Riesgo por trade (USD)", 10.0, value=150.0, step=10.0,
                                      help="Lo máximo que pierdes si salta el stop. Con 50K de cuenta y 2.000 de pérdida máxima, 150 USD son unas 13 pérdidas seguidas.")
          mult_na = e2.slider("Ancho de la zona (multiplicador)", 0.5, 2.0, 1.0, 0.05, help="1,0 = el del estudio. Más ancho = menos operaciones, más selectivas.")
          slip_na = e3.number_input("Slippage por lado (ticks)", 0.0, value=1.0, step=0.5)
          com_na = e4.number_input("Comisión ida y vuelta (USD/contrato)", 0.0, value=1.50, step=0.10)
          with st.expander("⚙️ Más opciones de la estrategia"):
              o1, o2, o3, o4 = st.columns(4)
              lookback_na = o1.number_input("Días para medir el ruido", 5, 60, 14)
              k_ini_na = o2.selectbox("Primera decisión (NY)", HORAS_NA[:-1], index=0)
              k_fin_na = o3.selectbox("Última entrada nueva (NY)", HORAS_NA[:-1], index=11)
              perd_na = o4.number_input("Parar el día tras N pérdidas (0 = sin límite)", 0, 10, 2)
              o5, o6, o7, o8 = st.columns(4)
              vwap_na = o5.checkbox("Stop = máx(banda, VWAP)", True)
              cortos_na = o6.checkbox("Permitir cortos", True)
              duro_na = o7.checkbox("Stop como orden real", True, help="Recomendado en prop firm: limita la pérdida dentro de cada media hora.")
              forzar_na = o8.checkbox("Si da 0 contratos, entrar con 1", False)
              o9, o10, o11, o12 = st.columns(4)
              modo_na = o9.radio("Tamaño", ["riesgo", "volatilidad"], format_func=lambda m: {
                  "riesgo": "Riesgo fijo (prop firm)", "volatilidad": "Volatilidad objetivo (estudio)"}[m])
              stop_min_na = o10.number_input("Stop mínimo para dimensionar (pts)", 0.25, value=3.0, step=0.25)
              stop_max_na = o11.number_input("No entrar si el stop está a más de (pts, 0 = sin filtro)", 0.0, value=0.0, step=1.0)
              capital_na = o12.number_input("Capital (modo volatilidad)", 1000.0, value=50_000.0, step=5000.0)
              n_boot_na = st.select_slider("Trayectorias del bootstrap", [1000, 2000, 5000, 10000], 2000)
          correr_na = st.form_submit_button("Correr backtest", type="primary", icon="⚡")

      if correr_na:
          try:
              with st.spinner("Preparando las barras de 30 minutos…"):
                  if fuente_na == "CSV":
                      if archivo_na is None:
                          raise ValueError("Sube un archivo CSV o elige los datos de ejemplo.")
                      barras_na = _barras_csv_na(archivo_na.getvalue(), zona_na)
                  elif fuente_na == "Yahoo":
                      barras_na = _barras_yahoo_na()
                  else:
                      barras_na = _barras_ejemplo_na()
              reglas_na = replace(base_na, meta=meta_na, mll=mll_na, bloqueo=bloq_na, dll=dll_na,
                                  consistencia=cons_na / 100, max_contratos_micro=int(micros_na))
              par_na = NA.Parametros(
                  lookback=int(lookback_na), multiplicador=float(mult_na), usar_vwap_stop=vwap_na, permitir_cortos=cortos_na,
                  primera_k=HORAS_NA.index(k_ini_na), ultima_k_entrada=HORAS_NA.index(k_fin_na), max_perdidas_dia=int(perd_na),
                  slippage_ticks=float(slip_na), comision_rt=float(com_na), modo_tamano=modo_na, riesgo_usd=float(riesgo_na),
                  stop_min_puntos=float(stop_min_na), stop_max_puntos=float(stop_max_na), forzar_un_contrato=forzar_na,
                  max_contratos=int(reglas_na.max_contratos_micro), capital=float(capital_na), stop_duro=duro_na)
              with st.spinner("Corriendo el backtest y simulando evaluaciones…"):
                  tr_na, di_na, ba_na = NA.backtest(barras_na, par_na)
                  vent_na = NA.ventanas_historicas(di_na, reglas_na) if not di_na.empty else pd.DataFrame()
                  boot_na = NA.bootstrap(di_na, reglas_na, n=int(n_boot_na)) if not di_na.empty else pd.DataFrame()
              st.session_state["na_res"] = dict(trades=tr_na, diario=di_na, bandas=ba_na, vent=vent_na, boot=boot_na, reglas=reglas_na,
                                                saltados=di_na.attrs.get("saltados", {}), fuente=fuente_na, par=par_na)
          except Exception as e:  # noqa: BLE001
              st.error(f"No pude correr el backtest: {e}")

      na = st.session_state.get("na_res")
      if not na:
          st.info("Elige la fuente de datos y pulsa «Correr backtest». Puedes empezar con los datos de ejemplo.")
      elif na["trades"].empty:
          st.warning("No hubo operaciones. Revisa que haya al menos «días para medir el ruido» + 1 días completos de datos.")
      else:
          trades_na, diario_na, reglas_na = na["trades"], na["diario"], na["reglas"]
          rs = NA.resumen(trades_na, diario_na)
          if na["fuente"] == "Ejemplo":
              st.warning("Estás viendo **datos simulados**. Sirven para entender la herramienta; para evaluar la estrategia sube un CSV real del MES o ES.")
          elif len(diario_na) < 250:
              st.warning(f"La muestra tiene solo {len(diario_na)} días hábiles. Con menos de ~1 año los resultados son casi pura suerte.")
          st.caption(f"Muestra: {diario_na['fecha'].min():%d/%m/%Y} a {diario_na['fecha'].max():%d/%m/%Y} · {len(diario_na)} días · "
                     f"riesgo {na['par'].riesgo_usd:,.0f} USD por trade · slippage {na['par'].slippage_ticks:g} tick(s) por lado.")

          c = st.columns(6)
          usd = lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
          c[0].metric("P&L neto", usd(rs['P&L neto (USD)']))
          c[1].metric("Trades", f"{rs['Trades']:,}", help=f"{rs['Trades por día operado']} por día operado; se operó el {rs['% de días con trade']}% de los días.")
          c[2].metric("Acierto", f"{rs['Tasa de acierto (%)']}%", help="En momentum lo normal es acertar poco y ganar mucho más de lo que se pierde.")
          c[3].metric("Profit factor", f"{rs['Profit factor']}", help="Ganancias brutas ÷ pérdidas brutas. Mayor que 1 = gana.")
          c[4].metric("Sharpe", f"{rs['Sharpe (diario anualizado)']}")
          c[5].metric("Drawdown máx.", usd(rs['Drawdown máximo (USD)']))

          vista_na = st.segmented_control("Ver", ["Curva", f"Prop firm · {reglas_na.nombre}", "Por periodo y hora", "Ver un día", "Trades"],
                                          default="Curva", key="vista_na") or "Curva"

          if vista_na == "Curva":
              fig = go.Figure()
              eq = diario_na["equity"]
              fig.add_scatter(x=diario_na["fecha"], y=eq, mode="lines", name="P&L acumulado", line=dict(color=CORAL, width=2.6),
                              fill="tozeroy", fillcolor="rgba(255,107,91,0.10)",
                              hovertemplate="%{x|%d %b %Y}<br>P&L acumulado: $%{y:,.0f}<extra></extra>")
              fig.update_layout(height=380, title=dict(text="P&L acumulado (USD)", font=dict(size=15)), showlegend=False, hovermode="x unified")
              st.plotly_chart(fig, width="stretch", theme=None)
              dd = NA.drawdown(diario_na)
              fig2 = go.Figure(go.Scatter(x=diario_na["fecha"], y=dd, mode="lines", fill="tozeroy", line=dict(color=ACERO, width=2),
                                          fillcolor="rgba(125,155,209,0.18)", hovertemplate="%{x|%d %b %Y}<br>Drawdown: $%{y:,.0f}<extra></extra>"))
              fig2.add_hline(y=-reglas_na.mll, line_dash="dash", line_color=CORAL,
                             annotation_text=f"Pérdida máxima de {reglas_na.nombre}", annotation_font_color=CORAL)
              fig2.update_layout(height=250, title=dict(text="Caída desde el máximo (USD)", font=dict(size=15)), showlegend=False, hovermode="x unified")
              st.plotly_chart(fig2, width="stretch", theme=None)
              with st.expander("Todas las estadísticas"):
                  st.dataframe(pd.DataFrame(rs.items(), columns=["Métrica", "Valor"]).astype({"Valor": str}), hide_index=True, width="stretch")
              sl = na["saltados"]
              if sl:
                  st.caption(f"Señales no tomadas: {sl.get('sin_contratos', 0)} porque el stop costaba más que el riesgo permitido "
                             f"(daba 0 contratos) y {sl.get('stop_muy_lejos', 0)} por stop demasiado lejano.")

          elif vista_na.startswith("Prop firm"):
              rv, rb = NA.resumir(na["vent"]), NA.resumir(na["boot"])
              a, b = st.columns(2, gap="large")
              for col, titulo, sub, rr in ((a, "Ventanas históricas", "Una evaluación que empieza cada día de la muestra", rv),
                                           (b, "Bootstrap", f"{len(na['boot']):,} trayectorias de 250 días remuestreando días reales".replace(",", "."), rb)):
                  pasa = rr.get("Pasan (%)", 0) or 0
                  falla = rr.get("Fallan (%)", 0) or 0
                  sin = rr.get("Sin terminar (%)", 0) or 0
                  dias_p = rr.get("Días medianos para pasar")
                  col.markdown(f"""<div class="card"><div class="sub">{titulo} · {sub}</div>
                      <div class="big">{pasa:.0f}%</div><div class="sub" style="margin:0 0 8px">pasan la evaluación</div>
                      <div class="barra"><div style="width:{max(pasa, 1):.1f}%"></div></div>
                      <ul><li>Fallan: <b>{falla:.0f}%</b></li><li>Sin terminar: <b>{sin:.0f}%</b></li>
                      <li>Días medianos para pasar: <b>{'—' if dias_p is None or dias_p != dias_p else f'{dias_p:.0f}'}</b></li></ul></div>""",
                               unsafe_allow_html=True)
              st.caption(f"Reglas: meta ${reglas_na.meta:,.0f}, pérdida máxima trailing EOD ${reglas_na.mll:,.0f} (se congela en inicial + "
                         f"${reglas_na.bloqueo:,.0f}), límite diario {'no tiene' if reglas_na.dll == 0 else f'${reglas_na.dll:,.0f}'}, consistencia "
                         f"{'no tiene' if reglas_na.consistencia == 0 else f'{reglas_na.consistencia:.0%}'}. La pérdida intradía se mide en los extremos "
                         "de cada barra de 30 min. Las ventanas históricas se solapan y el bootstrap supone días independientes. "
                         "Verifica siempre las reglas vigentes de la firma.")
              mejor = rs.get("Mejor día / P&L total (%)")
              if mejor == mejor and mejor is not None and mejor > reglas_na.consistencia * 100 > 0:
                  st.warning(f"Tu mejor día es el {mejor:.0f}% del P&L total: la regla de consistencia te va a frenar.")
              if len(na["boot"]):
                  dist = na["boot"][na["boot"]["resultado"] == "Pasa"]["dias"]
                  if len(dist):
                      fig = go.Figure(go.Histogram(x=dist, nbinsx=30, marker_color=CORAL, opacity=0.85,
                                                   hovertemplate="%{x} días: %{y} trayectorias<extra></extra>"))
                      fig.update_layout(height=280, title=dict(text="¿En cuántos días se pasa? (bootstrap)", font=dict(size=15)),
                                        xaxis_title="Días hábiles hasta pasar", yaxis_title="Trayectorias", bargap=0.05)
                      st.plotly_chart(fig, width="stretch", theme=None)

          elif vista_na == "Por periodo y hora":
              mens = NA.por_periodo(trades_na, "M")
              fig = go.Figure(go.Bar(x=mens.index.astype(str), y=mens["P&L (USD)"],
                                     marker_color=np.where(mens["P&L (USD)"] >= 0, CORAL, ACERO),
                                     hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
              fig.update_layout(height=300, title=dict(text="P&L por mes (USD)", font=dict(size=15)))
              st.plotly_chart(fig, width="stretch", theme=None)
              a, b = st.columns(2, gap="large")
              a.markdown("**Por año**")
              a.dataframe(NA.por_periodo(trades_na, "Y"), width="stretch")
              hora = NA.por_hora_entrada(trades_na)
              b.markdown("**Por hora de entrada (Nueva York)**")
              b.dataframe(hora, width="stretch")

          elif vista_na == "Ver un día":
              dias_tr = sorted(trades_na["fecha"].dt.date.unique(), reverse=True)
              dia = st.selectbox("Día", dias_tr, format_func=lambda d: d.strftime("%d/%m/%Y"))
              bd = na["bandas"]
              d = bd[bd["fecha"] == dia].sort_values("k")
              t = trades_na[trades_na["fecha"].dt.date == dia]
              fig = go.Figure()
              fig.add_scatter(x=d["hora_cierre"], y=d["UB"], mode="lines", line=dict(color=ACERO, width=1.5, dash="dash"), name="Banda superior")
              fig.add_scatter(x=d["hora_cierre"], y=d["LB"], mode="lines", line=dict(color=ACERO, width=1.5, dash="dash"), name="Banda inferior",
                              fill="tonexty", fillcolor="rgba(125,155,209,0.14)")
              fig.add_scatter(x=d["hora_cierre"], y=d["vwap"], mode="lines", line=dict(color=CREMA_TENUE, width=1.6, dash="dot"), name="VWAP")
              fig.add_scatter(x=d["hora_cierre"], y=d["close"], mode="lines+markers", line=dict(color=CORAL, width=2.6),
                              marker=dict(size=7), name="Precio (cierre 30 min)")
              for _, tr in t.iterrows():
                  color = CORAL if tr["pnl_usd"] >= 0 else ACERO
                  fig.add_scatter(x=[tr["hora_entrada"]], y=[tr["precio_entrada"]], mode="markers", showlegend=False,
                                  marker=dict(symbol="triangle-up" if tr["lado"] == "Largo" else "triangle-down", size=16, color=CREMA,
                                              line=dict(color=NAVY, width=1)),
                                  hovertemplate=f"Entrada {tr['lado'].lower()} ×{tr['contratos']}<br>%{{y:,.2f}}<extra></extra>")
                  fig.add_scatter(x=[tr["hora_salida"]], y=[tr["precio_salida"]], mode="markers", showlegend=False,
                                  marker=dict(symbol="x", size=13, color=color),
                                  hovertemplate=f"Salida ({tr['motivo'].lower()})<br>%{{y:,.2f}} · {tr['pnl_usd']:+,.0f} USD<extra></extra>")
              fig.update_layout(height=440, hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                                title=dict(text="La zona sombreada es el ruido: solo se entra cuando el precio la rompe", font=dict(size=14)))
              st.plotly_chart(fig, width="stretch", theme=None)
              st.dataframe(t, hide_index=True, width="stretch")

          else:
              st.dataframe(trades_na.sort_values(["fecha", "hora_entrada"], ascending=False), hide_index=True, width="stretch",
                           column_config={"fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                                          "pnl_usd": st.column_config.NumberColumn("P&L (USD)", format="%.2f")})
              b1, b2 = st.columns(2)
              b1.download_button("Descargar trades (CSV)", trades_na.to_csv(index=False).encode("utf-8"), "trades_noise_area_mes.csv", "text/csv")
              b2.download_button("Descargar P&L diario (CSV)", diario_na.to_csv(index=False).encode("utf-8"), "diario_noise_area_mes.csv", "text/csv")

          st.caption("Resultados hipotéticos de un backtest: no garantizan resultados futuros. Las decisiones se toman con barras de 30 min, "
                     "el slippage y las comisiones son fijos, y no se modelan los vencimientos (usa una serie continua ajustada). "
                     "No es asesoría financiera.")

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
