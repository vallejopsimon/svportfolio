"""
Perfil de una empresa: descripción, CEO, datos básicos, directivos y noticias recientes.

Fuentes (cada una es opcional: se muestra lo que responda):
  • Yahoo Finance (resumen de la empresa y directivos), si no está bloqueado.
  • Wikipedia en español (descripción) y Wikidata (CEO, fundación, sede, empleados, sitio web).
  • Google Noticias en español (RSS) y el buscador de noticias de Yahoo Finance.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime

import requests

UA = {"User-Agent": "RadarInversion/1.0 (app educativa de finanzas; contacto en el repositorio)"}
TIEMPO = 12

SECTOR_ES = {
    "Technology": "Tecnología", "Financial Services": "Servicios financieros", "Healthcare": "Salud",
    "Energy": "Energía", "Utilities": "Servicios públicos", "Basic Materials": "Materiales básicos",
    "Industrials": "Industria", "Consumer Cyclical": "Consumo cíclico", "Consumer Defensive": "Consumo básico",
    "Communication Services": "Comunicaciones", "Real Estate": "Inmobiliario",
}
PALABRAS_EMPRESA = ("empresa", "compañía", "corporación", "multinacional", "banco", "grupo", "holding",
                    "conglomerado", "sociedad", "fabricante", "minera", "petrolera", "aseguradora", "fondo")
SUFIJOS = r"\b(s\.?a\.?s?\.?|e\.?s\.?p\.?|inc\.?|corp\.?|corporation|incorporated|holdings?|ltd\.?|plc|n\.?v\.?|ag|sab de cv|s\.a\.b\.? de c\.v\.?|co\.?|company)\b"


def nombre_corto(nombre: str) -> str:
    """'Apple Inc.' -> 'Apple'; 'Grupo Energía Bogotá S.A. E.S.P.' -> 'Grupo Energía Bogotá'."""
    n = re.sub(SUFIJOS, "", nombre or "", flags=re.I)
    n = re.sub(r"[,.\s]+$", "", re.sub(r"\s{2,}", " ", n)).strip(" ,.-")
    return n or (nombre or "")


# ---------------------------------------------------------------------------
# Yahoo: resumen y directivos (viene en Ticker.info)
# ---------------------------------------------------------------------------

def desde_yahoo(info: dict) -> dict:
    if not info:
        return {}
    directivos = []
    for o in info.get("companyOfficers") or []:
        if o.get("name"):
            directivos.append({"Nombre": o["name"], "Cargo": o.get("title", ""), "Edad": o.get("age")})
    ceo = next((d["Nombre"] for d in directivos if re.search(r"\bCEO\b|Chief Executive|Presidente", d["Cargo"] or "", re.I)), None)
    ciudad = ", ".join(x for x in (info.get("city"), info.get("country")) if x)
    return {k: v for k, v in {
        "resumen_en": info.get("longBusinessSummary"), "ceo": ceo, "directivos": directivos[:8],
        "sitio": info.get("website"), "empleados": info.get("fullTimeEmployees"), "sede": ciudad or None,
        "sector": SECTOR_ES.get(info.get("sector"), info.get("sector")), "industria": info.get("industry"),
    }.items() if v}


# ---------------------------------------------------------------------------
# Wikipedia + Wikidata
# ---------------------------------------------------------------------------

def wikipedia(nombre: str, idioma: str = "es") -> dict:
    """Busca la página de la empresa y devuelve el resumen introductorio, el enlace y su id de Wikidata."""
    consulta = nombre_corto(nombre)
    r = requests.get(f"https://{idioma}.wikipedia.org/w/api.php", headers=UA, timeout=TIEMPO, params={
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{consulta} empresa" if idioma == "es" else f"{consulta} company",
        "gsrlimit": 5, "prop": "extracts|pageprops|info", "exintro": 1, "explaintext": 1, "inprop": "url", "redirects": 1})
    r.raise_for_status()
    paginas = sorted((r.json().get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    palabras_nombre = {w for w in re.findall(r"\w+", consulta.lower()) if len(w) > 2}
    señales = [w for w in PALABRAS_EMPRESA + ("company", "corporation") if w not in palabras_nombre]
    mejor, mejor_puntaje = None, 0.0
    for p in paginas:
        texto = (p.get("extract") or "").strip()
        bajo, titulo = texto.lower(), (p.get("title") or "").lower()
        if not texto or "(desambiguación)" in titulo or re.search(r"puede referirse a|may refer to|se puede referir a", bajo[:200]):
            continue  # páginas de desambiguación
        # 1) que hable de una empresa (con palabras que no vengan del propio nombre)
        if not any(w in bajo[:600] for w in señales):
            continue
        # 2) que el título se parezca al nombre buscado y lo mencione al inicio
        parecido = SequenceMatcher(None, re.sub(r"\s*\(.*?\)", "", titulo), consulta.lower()).ratio()
        menciona = sum(w in bajo[:300] for w in palabras_nombre) / max(len(palabras_nombre), 1)
        puntaje = parecido + menciona - (0.5 if "(" in titulo and not any(w in titulo for w in ("empresa", "compañía", "banco")) else 0)
        if menciona >= 0.5 and puntaje > mejor_puntaje:
            mejor, mejor_puntaje = p, puntaje
    if mejor is None or mejor_puntaje < 1.0:
        return {}
    return {"resumen_es": mejor["extract"].strip(), "wiki_url": mejor.get("fullurl"), "wiki_titulo": mejor.get("title"),
            "qid": (mejor.get("pageprops") or {}).get("wikibase_item")}


def _etiquetas(ids: list[str]) -> dict:
    if not ids:
        return {}
    r = requests.get("https://www.wikidata.org/w/api.php", headers=UA, timeout=TIEMPO, params={
        "action": "wbgetentities", "ids": "|".join(ids[:40]), "props": "labels", "languages": "es|en", "format": "json"})
    r.raise_for_status()
    out = {}
    for q, e in (r.json().get("entities") or {}).items():
        lab = e.get("labels") or {}
        out[q] = (lab.get("es") or lab.get("en") or {}).get("value")
    return out


def wikidata(qid: str) -> dict:
    """CEO actual (P169), fundación (P571), sede (P159), empleados (P1128) y sitio web (P856)."""
    r = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json", headers=UA, timeout=TIEMPO)
    r.raise_for_status()
    claims = (r.json().get("entities", {}).get(qid) or {}).get("claims", {})

    def valores(p):
        return [c for c in claims.get(p, []) if c.get("mainsnak", {}).get("datavalue")]

    def fecha_q(c, prop):
        q = (c.get("qualifiers") or {}).get(prop)
        return q[0]["datavalue"]["value"]["time"] if q and q[0].get("datavalue") else None

    out = {}
    ceos = [c for c in valores("P169") if not fecha_q(c, "P582")]  # sin fecha de fin = vigente
    ceos.sort(key=lambda c: fecha_q(c, "P580") or "", reverse=True)
    ids_ceo = [c["mainsnak"]["datavalue"]["value"]["id"] for c in ceos[:2]]
    sedes = [c["mainsnak"]["datavalue"]["value"]["id"] for c in valores("P159")[:1]]
    nombres = _etiquetas(ids_ceo + sedes)
    if ids_ceo and nombres.get(ids_ceo[0]):
        out["ceo"] = nombres[ids_ceo[0]]
    if sedes and nombres.get(sedes[0]):
        out["sede"] = nombres[sedes[0]]
    fund = valores("P571")
    if fund:
        t = fund[0]["mainsnak"]["datavalue"]["value"]["time"]
        m = re.match(r"[+-](\d{4})", t)
        if m:
            out["fundacion"] = int(m.group(1))
    emp = valores("P1128")
    if emp:
        emp.sort(key=lambda c: fecha_q(c, "P585") or "", reverse=True)
        try:
            out["empleados"] = int(float(emp[0]["mainsnak"]["datavalue"]["value"]["amount"]))
        except (KeyError, ValueError):
            pass
    web = valores("P856")
    if web:
        out["sitio"] = web[0]["mainsnak"]["datavalue"]["value"]
    return out


# ---------------------------------------------------------------------------
# Noticias
# ---------------------------------------------------------------------------

def noticias_google(nombre: str, n: int = 8) -> list[dict]:
    consulta = f'"{nombre_corto(nombre)}"'
    r = requests.get("https://news.google.com/rss/search", headers=UA, timeout=TIEMPO,
                     params={"q": consulta + " when:60d", "hl": "es-419", "gl": "CO", "ceid": "CO:es-419"})
    r.raise_for_status()
    raiz = ET.fromstring(r.content)
    out = []
    for it in raiz.iter("item"):
        titulo = html.unescape(it.findtext("title") or "")
        fuente = it.findtext("source") or ""
        if fuente and titulo.endswith(" - " + fuente):  # Google agrega " - Medio" al final del título
            titulo = titulo[: -len(" - " + fuente)]
        try:
            fecha = parsedate_to_datetime(it.findtext("pubDate"))
        except (TypeError, ValueError):
            fecha = None
        out.append({"titulo": titulo, "url": it.findtext("link"), "fuente": fuente, "fecha": fecha, "idioma": "es"})
        if len(out) >= n:
            break
    return out


def noticias_yahoo(ticker: str, http_json, n: int = 8) -> list[dict]:
    js = http_json("https://query2.finance.yahoo.com/v1/finance/search",
                   {"q": ticker, "quotesCount": 0, "newsCount": n, "enableFuzzyQuery": "false"})
    out = []
    for x in js.get("news") or []:
        ts = x.get("providerPublishTime")
        out.append({"titulo": x.get("title", ""), "url": x.get("link"), "fuente": x.get("publisher", ""),
                    "fecha": datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None, "idioma": "en"})
    return out


def combinar_noticias(*listas: list[dict], n: int = 10) -> list[dict]:
    vistas, out = set(), []
    for x in sorted((x for l in listas for x in l if x.get("titulo") and x.get("url")),
                    key=lambda x: x["fecha"] or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True):
        clave = re.sub(r"\W+", "", x["titulo"].lower())[:60]
        if clave not in vistas:
            vistas.add(clave)
            out.append(x)
    return out[:n]


def hace(fecha: datetime | None) -> str:
    if not fecha:
        return ""
    s = (datetime.now(timezone.utc) - fecha).total_seconds()
    if s < 3600:
        return f"hace {max(int(s // 60), 1)} min"
    if s < 86400:
        h = int(s // 3600)
        return f"hace {h} hora{'s' if h > 1 else ''}"
    d = int(s // 86400)
    return "ayer" if d == 1 else f"hace {d} días" if d < 30 else fecha.strftime("%d/%m/%Y")
