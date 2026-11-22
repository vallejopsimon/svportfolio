"""
Motor analítico: métricas fundamentales, puntajes de oportunidad,
valoración por múltiplos y optimización media-varianza (Markowitz).
No depende de Streamlit ni de Yahoo: recibe DataFrames y devuelve resultados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# 1. Fundamentales
# ---------------------------------------------------------------------------

def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _inv(x):
    """Inverso de un múltiplo (P/E -> rendimiento de utilidades). Negativos se conservan."""
    x = _num(x)
    return np.nan if (np.isnan(x) or x == 0) else 1.0 / x


def metricas_desde_info(ticker: str, info: dict) -> dict:
    """Convierte el diccionario .info de yfinance en una fila de métricas."""
    g = lambda k: _num(info.get(k))
    precio = g("currentPrice")
    if np.isnan(precio):
        precio = g("regularMarketPrice")
    mcap, fcf = g("marketCap"), g("freeCashflow")
    hi, lo = g("fiftyTwoWeekHigh"), g("fiftyTwoWeekLow")
    div_rate = g("dividendRate")
    target = g("targetMeanPrice")
    de = g("debtToEquity")
    return {
        "Ticker": ticker,
        "Nombre": info.get("shortName") or info.get("longName") or ticker,
        "Sector": info.get("sector") or "Sin sector",
        "Industria": info.get("industry") or "",
        "Moneda": info.get("currency") or "USD",
        "Precio": precio,
        "Cap. mercado": mcap,
        "P/E": g("trailingPE"),
        "P/E fwd": g("forwardPE"),
        "PEG": g("trailingPegRatio") if not np.isnan(g("trailingPegRatio")) else g("pegRatio"),
        "P/B": g("priceToBook"),
        "P/S": g("priceToSalesTrailing12Months"),
        "EV/EBITDA": g("enterpriseToEbitda"),
        "EV/Ventas": g("enterpriseToRevenue"),
        "EV": g("enterpriseValue"),
        "Div. yield": div_rate / precio if precio and not np.isnan(div_rate) else np.nan,
        "FCF yield": fcf / mcap if mcap and not np.isnan(fcf) else np.nan,
        "ROE": g("returnOnEquity"),
        "ROA": g("returnOnAssets"),
        "Margen bruto": g("grossMargins"),
        "Margen operativo": g("operatingMargins"),
        "Margen neto": g("profitMargins"),
        "Crec. ventas": g("revenueGrowth"),
        "Crec. utilidades": g("earningsGrowth"),
        "Deuda/Patrimonio": de / 100 if not np.isnan(de) else np.nan,  # Yahoo lo da en %
        "Razón corriente": g("currentRatio"),
        "Beta": g("beta"),
        "Pos. 52s": (precio - lo) / (hi - lo) if hi and lo and hi > lo else np.nan,
        "Upside analistas": target / precio - 1 if precio and not np.isnan(target) else np.nan,
        "Recomendación": g("recommendationMean"),  # 1 = compra fuerte ... 5 = venta
        "N analistas": g("numberOfAnalystOpinions"),
    }


def momentum(precios: pd.DataFrame) -> pd.DataFrame:
    """Momentum 12-1 (excluye el último mes) y retorno 6 meses."""
    out = {}
    for t in precios.columns:
        s = precios[t].dropna()
        if len(s) < 130:
            out[t] = {"Mom. 12-1": np.nan, "Ret. 6m": np.nan}
            continue
        p_1m = s.iloc[-22] if len(s) > 22 else np.nan
        p_12m = s.iloc[-252] if len(s) >= 252 else s.iloc[0]
        p_6m = s.iloc[-126]
        out[t] = {"Mom. 12-1": p_1m / p_12m - 1, "Ret. 6m": s.iloc[-1] / p_6m - 1}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------
# 2. Puntaje de oportunidad (percentiles 0-100)
# ---------------------------------------------------------------------------

# (columna, mayor_es_mejor). Los múltiplos se invierten a "yields" para que
# las utilidades negativas queden naturalmente al fondo del ranking.
BLOQUES = {
    "Valor": [("E/P fwd", True), ("EBITDA/EV", True), ("B/P", True), ("S/P", True), ("FCF yield", True)],
    "Calidad": [("ROE", True), ("Margen operativo", True), ("Margen neto", True),
                ("Deuda/Patrimonio", False), ("Razón corriente", True)],
    "Crecimiento": [("Crec. ventas", True), ("Crec. utilidades", True)],
    "Momentum": [("Mom. 12-1", True), ("Ret. 6m", True)],
}


def _rank(s: pd.Series, mayor_mejor: bool) -> pd.Series:
    r = s.rank(pct=True, ascending=mayor_mejor)
    return r


def puntajes(df: pd.DataFrame, pesos: dict, relativo_sector: bool = True, min_pares: int = 4) -> pd.DataFrame:
    d = df.copy()
    d["E/P fwd"] = d["P/E fwd"].map(_inv)
    d["EBITDA/EV"] = d["EV/EBITDA"].map(_inv)
    d["B/P"] = d["P/B"].map(_inv)
    d["S/P"] = d["P/S"].map(_inv)

    def puntuar(g: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame(index=g.index)
        for bloque, cols in BLOQUES.items():
            ranks = [_rank(g[c], m) for c, m in cols if c in g and g[c].notna().sum() >= 2]
            res[bloque] = pd.concat(ranks, axis=1).mean(axis=1) if ranks else np.nan
        return res

    if relativo_sector:
        tam = d.groupby("Sector")["Ticker"].transform("size")
        grandes = d[tam >= min_pares]
        pequenos = d[tam < min_pares]
        partes = [puntuar(g) for _, g in grandes.groupby("Sector")]
        if len(pequenos):
            partes.append(puntuar(d).loc[pequenos.index])  # vs. universo completo
        sc = pd.concat(partes).reindex(d.index)
    else:
        sc = puntuar(d)

    sc = sc.fillna(0.5)  # sin datos = neutral
    total_peso = sum(pesos.values()) or 1
    sc["Puntaje"] = sum(sc[b] * w for b, w in pesos.items()) / total_peso
    for c in sc:
        d[c] = (sc[c] * 100).round(1)
    return d.sort_values("Puntaje", ascending=False)


# ---------------------------------------------------------------------------
# 3. Valoración relativa por múltiplos
# ---------------------------------------------------------------------------

def valoracion_multiplos(df: pd.DataFrame, ticker: str, pares: list[str]) -> pd.DataFrame:
    """
    Precio implícito usando la mediana de los pares. Se trabaja en razones
    (precio × múltiplo_pares / múltiplo_propio) para que funcione con ADRs
    que reportan en otra moneda.
    """
    fila = df.set_index("Ticker").loc[ticker]
    peers = df[df["Ticker"].isin(pares) & (df["Ticker"] != ticker)]
    p = fila["Precio"]
    filas = []
    for m in ["P/E", "P/E fwd", "P/B", "P/S"]:
        propio = fila[m]
        med = peers.loc[peers[m] > 0, m].median()
        if propio and propio > 0 and pd.notna(med):
            imp = p * med / propio
            filas.append({"Método": m, "Propio": propio, "Mediana pares": med, "Precio implícito": imp})
    for m in ["EV/EBITDA", "EV/Ventas"]:
        propio, ev, mcap = fila[m], fila["EV"], fila["Cap. mercado"]
        med = peers.loc[peers[m] > 0, m].median()
        if propio and propio > 0 and pd.notna(med) and ev and mcap:
            deuda_neta = ev - mcap
            mcap_imp = ev * med / propio - deuda_neta
            filas.append({"Método": m, "Propio": propio, "Mediana pares": med,
                          "Precio implícito": max(p * mcap_imp / mcap, 0)})
    out = pd.DataFrame(filas)
    if len(out):
        out["Upside"] = out["Precio implícito"] / p - 1
    return out


# ---------------------------------------------------------------------------
# 4. Optimización media-varianza
# ---------------------------------------------------------------------------

def retornos_diarios(precios: pd.DataFrame) -> pd.DataFrame:
    # ffill para alinear calendarios de distintas bolsas, sin rellenar el inicio
    return precios.ffill().pct_change(fill_method=None).dropna(how="any")


def ledoit_wolf(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Covarianza con encogimiento de Ledoit-Wolf hacia μ·I. Devuelve (Σ, δ)."""
    T, N = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    d2 = np.sum((S - F) ** 2)
    sq = np.sum(X ** 2, axis=1) ** 2
    b2_bar = (sq.sum() - T * np.sum(S ** 2)) / T ** 2
    b2 = min(max(b2_bar, 0), d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * F + (1 - delta) * S, delta


def estimar_parametros(ret: pd.DataFrame, metodo_cov: str = "Ledoit-Wolf",
                       metodo_mu: str = "Histórica", rf: float = 0.04,
                       ret_mercado: pd.Series | None = None, prima: float = 0.055):
    R = ret.values
    if metodo_cov == "Ledoit-Wolf":
        cov_d, delta = ledoit_wolf(R)
    else:
        cov_d, delta = np.cov(R, rowvar=False, ddof=1), 0.0
    cov = pd.DataFrame(np.atleast_2d(cov_d) * TRADING_DAYS, index=ret.columns, columns=ret.columns)
    betas = None
    if ret_mercado is not None:
        m = ret_mercado.reindex(ret.index).ffill().fillna(0)
        betas = ret.apply(lambda c: np.cov(c, m)[0, 1] / np.var(m, ddof=1))
    if metodo_mu == "CAPM" and betas is not None:
        mu = rf + betas * prima
    else:
        mu = ret.mean() * TRADING_DAYS
    return mu, cov, delta, betas


def _stats(w, mu, cov, rf):
    r = float(w @ mu)
    v = float(np.sqrt(max(w @ cov @ w, 1e-16)))
    return r, v, (r - rf) / v


def _resolver(obj, n, cap, extra_cons=(), x0s=None):
    bounds = [(0.0, cap)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_cons]
    x0s = x0s or [np.full(n, 1.0 / n)]
    mejor = None
    for x0 in x0s:
        r = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                     options={"maxiter": 1000, "ftol": 1e-12})
        if r.success and (mejor is None or r.fun < mejor.fun):
            mejor = r
    if mejor is None:
        raise ValueError("El optimizador no convergió. Prueba subir el peso máximo o cambiar activos.")
    w = np.clip(mejor.x, 0, None)
    w[w < 1e-6] = 0
    return w / w.sum()


def min_varianza(mu, cov, cap=1.0):
    C = cov.values
    n = len(mu)
    return _resolver(lambda w: w @ C @ w, n, cap)


def max_sharpe(mu, cov, rf, cap=1.0):
    m, C = mu.values, cov.values
    n = len(m)
    w_mv = min_varianza(mu, cov, cap)
    mejor_activo = np.zeros(n)
    orden = np.argsort(-(m - rf) / np.sqrt(np.diag(C)))
    resto = 1.0
    for i in orden:  # llenado codicioso respetando el tope
        mejor_activo[i] = min(cap, resto)
        resto -= mejor_activo[i]
        if resto <= 1e-12:
            break
    obj = lambda w: -(w @ m - rf) / np.sqrt(max(w @ C @ w, 1e-16))
    return _resolver(obj, n, cap, x0s=[np.full(n, 1 / n), w_mv, mejor_activo])


def frontera(mu, cov, cap=1.0, puntos=40):
    m, C = mu.values, cov.values
    n = len(m)
    w_mv = min_varianza(mu, cov, cap)
    r_min = float(w_mv @ m)
    # retorno máximo factible con el tope de peso
    w_max, resto = np.zeros(n), 1.0
    for i in np.argsort(-m):
        w_max[i] = min(cap, resto)
        resto -= w_max[i]
        if resto <= 1e-12:
            break
    r_max = float(w_max @ m)
    out = []
    for tgt in np.linspace(r_min, r_max, puntos):
        cons = ({"type": "eq", "fun": lambda w, t=tgt: w @ m - t},)
        try:
            w = _resolver(lambda w: w @ C @ w, n, cap, cons, x0s=[w_mv, w_max])
            out.append((float(np.sqrt(w @ C @ w)), float(w @ m)))
        except ValueError:
            continue
    return pd.DataFrame(out, columns=["Volatilidad", "Retorno"])


def portafolios_aleatorios(mu, cov, rf, n=3000, seed=7):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(mu)) * 0.6, n)
    r = W @ mu.values
    v = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
    return pd.DataFrame({"Volatilidad": v, "Retorno": r, "Sharpe": (r - rf) / v})


def asignar_monto(pesos: pd.Series, precios: pd.Series, fx: pd.Series, monto: float,
                  fraccionadas: bool = False) -> pd.DataFrame:
    """Convierte pesos a dinero y número de acciones. fx = moneda inversión por 1 unidad de la moneda del activo."""
    precio_inv = precios * fx
    valor_obj = pesos * monto
    acciones = valor_obj / precio_inv
    if not fraccionadas:
        acciones = np.floor(acciones)
    valor_real = acciones * precio_inv
    return pd.DataFrame({
        "Peso objetivo": pesos,
        "Monto objetivo": valor_obj,
        "Precio (moneda inversión)": precio_inv,
        "Acciones": acciones,
        "Monto invertido": valor_real,
        "Peso real": valor_real / monto,
    })
