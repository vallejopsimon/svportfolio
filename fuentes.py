"""
Fuente alternativa de fundamentales: estados financieros de Yahoo Finance.

Cuando el "resumen" de Yahoo (Ticker.info) está bloqueado —pasa a menudo desde servidores en la nube,
porque exige un permiso de acceso llamado crumb— se usan dos servicios públicos de Yahoo que no lo exigen:

  1. chart  -> precio, moneda, nombre, rango de 52 semanas y dividendos del último año.
  2. fundamentals-timeseries -> estados financieros (ingresos, utilidad, EBITDA, patrimonio, deuda,
     caja, acciones en circulación, flujo de caja libre), cada dato con su moneda.

Con eso se calculan los mismos múltiplos y razones que usa la app (P/E, P/B, P/S, EV/EBITDA, ROE, márgenes,
crecimiento, deuda/patrimonio...). Si la empresa reporta en otra moneda (p. ej. Mineros reporta en USD
y cotiza en COP), las cifras se convierten antes de calcular.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import numpy as np

try:  # curl_cffi viene con yfinance e imita un navegador real (Yahoo bloquea menos)
    from curl_cffi import requests as _http
    _KW = {"impersonate": "chrome"}
except Exception:  # pragma: no cover
    import requests as _http
    _KW = {}

BASE = "https://query2.finance.yahoo.com"
_local = threading.local()


class LimiteYahoo(Exception):
    """Yahoo respondió 429 (demasiadas consultas)."""


def _sesion():
    if not hasattr(_local, "s"):
        _local.s = _http.Session(**_KW)
        _local.s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                               "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
                                 "Accept": "application/json"})
    return _local.s


def http_json(url: str, params: dict) -> dict:
    for intento in range(2):
        r = _sesion().get(url, params=params, timeout=20)
        if r.status_code == 429:
            if intento == 0:
                time.sleep(2)
                continue
            raise LimiteYahoo("Yahoo: demasiadas consultas (429)")
        r.raise_for_status()
        return r.json()
    raise LimiteYahoo("Yahoo: demasiadas consultas (429)")


# ---------------------------------------------------------------------------
# Servicio 1: chart (precio, moneda, nombre, dividendos)
# ---------------------------------------------------------------------------

def chart(simbolo: str, rango: str = "1y") -> dict:
    js = http_json(f"{BASE}/v8/finance/chart/{simbolo}",
                   {"range": rango, "interval": "1d", "events": "div", "includePrePost": "false"})
    res = (js.get("chart") or {}).get("result") or []
    if not res:
        raise ValueError(f"{simbolo}: Yahoo no tiene precios")
    return res[0]


def parse_chart(res: dict) -> dict:
    meta = res.get("meta", {})
    divs = ((res.get("events") or {}).get("dividends") or {}).values()
    ahora = time.time()
    div_12m = sum(d.get("amount", 0) for d in divs if ahora - d.get("date", 0) <= 366 * 86400)
    precio = meta.get("regularMarketPrice")
    if precio is None:
        closes = [c for c in (((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []) if c is not None]
        precio = closes[-1] if closes else None
    return {
        "precio": precio, "moneda": meta.get("currency") or "USD",
        "nombre": meta.get("longName") or meta.get("shortName") or meta.get("symbol"),
        "tipo": meta.get("instrumentType", ""), "hi52": meta.get("fiftyTwoWeekHigh"), "lo52": meta.get("fiftyTwoWeekLow"),
        "div_12m": div_12m,
    }


# ---------------------------------------------------------------------------
# Servicio 2: estados financieros
# ---------------------------------------------------------------------------
CONCEPTOS_PYG = ["TotalRevenue", "NetIncomeCommonStockholders", "EBITDA", "OperatingIncome", "GrossProfit", "FreeCashFlow"]
CONCEPTOS_BAL = ["StockholdersEquity", "TotalDebt", "CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments",
                 "OrdinarySharesNumber", "ShareIssued", "TotalAssets", "CurrentAssets", "CurrentLiabilities"]


def tipos_solicitados() -> list[str]:
    t = [f"trailing{c}" for c in CONCEPTOS_PYG] + [f"annual{c}" for c in CONCEPTOS_PYG]
    t += [f"quarterly{c}" for c in CONCEPTOS_BAL] + [f"annual{c}" for c in CONCEPTOS_BAL]
    return t


def timeseries(simbolo: str) -> dict:
    inicio = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp())
    fin = int(time.time()) + 86400
    return http_json(f"{BASE}/ws/fundamentals-timeseries/v1/finance/timeseries/{simbolo}",
                     {"symbol": simbolo, "type": ",".join(tipos_solicitados()), "period1": inicio, "period2": fin})


def parse_timeseries(js: dict) -> dict[str, list[tuple[str, float, str]]]:
    """{tipo: [(fecha, valor, moneda), ...] ordenado de más antiguo a más reciente}."""
    out = {}
    for item in ((js.get("timeseries") or {}).get("result") or []):
        tipo = ((item.get("meta") or {}).get("type") or [None])[0]
        if not tipo:
            continue
        pts = []
        for p in item.get(tipo) or []:
            if not p or not p.get("reportedValue"):
                continue
            v = p["reportedValue"].get("raw")
            if v is None:
                continue
            pts.append((p.get("asOfDate", ""), float(v), p.get("currencyCode") or ""))
        if pts:
            out[tipo] = sorted(pts)
    return out


def _ultimo(ts: dict, *tipos):
    for t in tipos:
        if ts.get(t):
            return ts[t][-1]
    return None


def calcular_metricas(ticker: str, ch: dict, ts: dict, fx_a_moneda_precio) -> dict:
    """
    Arma la misma fila de métricas que motor.metricas_desde_info, pero calculada a partir
    de los estados financieros. fx_a_moneda_precio(moneda_reporte) -> factor para pasar a la moneda del precio.
    """
    nan = np.nan
    precio, mon_p = ch["precio"], ch["moneda"]
    monedas = [v[2] for serie in ts.values() for v in serie if v[2]]
    mon_f = max(set(monedas), key=monedas.count) if monedas else mon_p
    fx = fx_a_moneda_precio(mon_f) if mon_f != mon_p else 1.0

    def val(*tipos):
        u = _ultimo(ts, *tipos)
        return u[1] * fx if u else nan

    def anuales(c):
        return [v[1] * fx for v in ts.get(f"annual{c}", [])]

    acciones = _ultimo(ts, "quarterlyOrdinarySharesNumber", "annualOrdinarySharesNumber",
                       "quarterlyShareIssued", "annualShareIssued")
    acciones = acciones[1] if acciones else nan
    mcap = precio * acciones if precio and acciones == acciones else nan

    ventas = val("trailingTotalRevenue", "annualTotalRevenue")
    utilidad = val("trailingNetIncomeCommonStockholders", "annualNetIncomeCommonStockholders")
    ebitda = val("trailingEBITDA", "annualEBITDA")
    op = val("trailingOperatingIncome", "annualOperatingIncome")
    bruta = val("trailingGrossProfit", "annualGrossProfit")
    fcf = val("trailingFreeCashFlow", "annualFreeCashFlow")
    patrimonio = val("quarterlyStockholdersEquity", "annualStockholdersEquity")
    deuda = val("quarterlyTotalDebt", "annualTotalDebt")
    caja = val("quarterlyCashCashEquivalentsAndShortTermInvestments", "annualCashCashEquivalentsAndShortTermInvestments",
               "quarterlyCashAndCashEquivalents", "annualCashAndCashEquivalents")
    activos = val("quarterlyTotalAssets", "annualTotalAssets")
    act_c = val("quarterlyCurrentAssets", "annualCurrentAssets")
    pas_c = val("quarterlyCurrentLiabilities", "annualCurrentLiabilities")
    ev = mcap + np.nan_to_num(deuda) - np.nan_to_num(caja) if mcap == mcap else nan

    div = lambda a, b: a / b if (a == a and b == b and b not in (0, 0.0)) else nan
    pos = lambda a, b: div(a, b) if (b == b and b > 0) else nan

    def crec(c):
        s = anuales(c)
        return s[-1] / s[-2] - 1 if len(s) >= 2 and s[-2] > 0 else nan

    hi, lo = ch.get("hi52"), ch.get("lo52")
    return {
        "Ticker": ticker, "Tipo": "Acción", "Nombre": ch.get("nombre") or ticker, "Sector": "Sin sector", "Industria": "",
        "Moneda": mon_p, "Precio": precio, "Cap. mercado": mcap,
        "P/E": div(mcap, utilidad), "P/E fwd": nan, "PEG": nan,
        "P/B": pos(mcap, patrimonio), "P/S": pos(mcap, ventas),
        "EV/EBITDA": div(ev, ebitda), "EV/Ventas": pos(ev, ventas), "EV": ev,
        "Div. yield": div(ch.get("div_12m", 0), precio) if ch.get("div_12m") else 0.0,
        "FCF yield": div(fcf, mcap),
        "ROE": pos(utilidad, patrimonio), "ROA": pos(utilidad, activos),
        "Margen bruto": pos(bruta, ventas), "Margen operativo": pos(op, ventas), "Margen neto": pos(utilidad, ventas),
        "Crec. ventas": crec("TotalRevenue"), "Crec. utilidades": crec("NetIncomeCommonStockholders"),
        "Deuda/Patrimonio": pos(deuda, patrimonio), "Razón corriente": pos(act_c, pas_c), "Beta": nan,
        "Pos. 52s": (precio - lo) / (hi - lo) if hi and lo and hi > lo and precio else nan,
        "Upside analistas": nan, "Recomendación": nan, "N analistas": nan,
        "Moneda reporte": mon_f,
    }
