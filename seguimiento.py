"""
Seguimiento de un portafolio real: a partir de las compras registradas (activo, fecha, cantidad, precio, comisión)
calcula la evolución diaria, la rentabilidad, el riesgo, la comparación contra un índice y recomendaciones.

Todo se expresa en la "moneda de reporte" que elija la persona (COP, USD o EUR). El costo de cada compra se
convierte con la tasa de cambio DEL DÍA DE LA COMPRA y el valor actual con la tasa de HOY, así la ganancia
incluye el efecto cambiario (y se muestra por separado).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

COLUMNAS = ["Activo", "Fecha", "Cantidad", "Precio compra", "Comisión"]


# ---------------------------------------------------------------------------
# Registro de compras
# ---------------------------------------------------------------------------

def registro_vacio() -> pd.DataFrame:
    return pd.DataFrame({"Activo": pd.Series(dtype="str"), "Fecha": pd.Series(dtype="datetime64[ns]"),
                         "Cantidad": pd.Series(dtype="float"), "Precio compra": pd.Series(dtype="float"),
                         "Comisión": pd.Series(dtype="float")})


def limpiar(trans: pd.DataFrame) -> pd.DataFrame:
    """Deja solo filas válidas: activo, fecha pasada y cantidad positiva."""
    if trans is None or trans.empty:
        return registro_vacio()
    t = trans.copy()
    for c in COLUMNAS:
        if c not in t:
            t[c] = np.nan
    t = t[COLUMNAS]
    t["Activo"] = t["Activo"].astype("string").str.strip().str.upper()
    t["Fecha"] = pd.to_datetime(t["Fecha"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ["Cantidad", "Precio compra", "Comisión"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["Comisión"] = t["Comisión"].fillna(0.0)
    hoy = pd.Timestamp(date.today())
    t = t[t["Activo"].notna() & (t["Activo"] != "") & t["Fecha"].notna() & (t["Fecha"] <= hoy) & (t["Cantidad"] > 0)]
    return t.sort_values("Fecha").reset_index(drop=True)


def periodo_desde(fecha: pd.Timestamp) -> str:
    """Periodo de Yahoo que cubre desde `fecha` hasta hoy."""
    dias = (pd.Timestamp(date.today()) - pd.Timestamp(fecha)).days
    for p, d in [("1y", 355), ("2y", 720), ("5y", 1815), ("10y", 3635)]:
        if dias <= d:
            return p
    return "max"


# ---------------------------------------------------------------------------
# Evolución diaria
# ---------------------------------------------------------------------------

def _primer_dia_habil(idx: pd.DatetimeIndex, fecha: pd.Timestamp) -> pd.Timestamp | None:
    pos = idx.searchsorted(fecha)
    return idx[pos] if pos < len(idx) else None


def evolucion(trans: pd.DataFrame, precios: pd.DataFrame, fx: pd.DataFrame) -> dict:
    """
    trans: registro limpio y con precio de compra completo.
    precios: precio diario de cada activo en SU moneda (columnas = activos).
    fx: unidades de moneda de reporte por 1 unidad de la moneda de cada activo (mismas columnas).
    """
    activos = sorted(trans["Activo"].unique())
    inicio = trans["Fecha"].min()
    idx = precios.index[precios.index >= inicio - pd.Timedelta(days=7)]
    P = precios.reindex(columns=activos).loc[idx].ffill().bfill()
    FX = fx.reindex(columns=activos).reindex(idx).ffill().bfill()
    unidades = pd.DataFrame(0.0, index=idx, columns=activos)
    nuevas = pd.DataFrame(0.0, index=idx, columns=activos)
    costo_dia = pd.Series(0.0, index=idx)
    lotes = []
    for _, l in trans.iterrows():
        d = _primer_dia_habil(idx, l["Fecha"])
        if d is None:
            continue
        a = l["Activo"]
        fx_compra = float(FX.at[d, a])
        costo_local = l["Cantidad"] * l["Precio compra"] + l["Comisión"]
        costo_rep = costo_local * fx_compra
        unidades.loc[idx >= d, a] += l["Cantidad"]
        nuevas.at[d, a] += l["Cantidad"]
        costo_dia[d] += costo_rep
        lotes.append({**l.to_dict(), "Día efectivo": d, "FX compra": fx_compra, "Costo local": costo_local, "Costo reporte": costo_rep})
    valor_activo = unidades * P * FX
    valor = valor_activo.sum(axis=1)
    invertido = costo_dia.cumsum()
    flujo_mercado = (nuevas * P * FX).sum(axis=1)  # valor de mercado de lo comprado ese día
    previo = valor.shift(1)
    r = (valor - flujo_mercado) / previo - 1
    r[(previo <= 0) | previo.isna()] = np.nan
    desde = invertido[invertido > 0].index.min()
    return dict(valor=valor.loc[desde:], invertido=invertido.loc[desde:], r=r.loc[desde:].dropna(),
                valor_activo=valor_activo.loc[desde:], P=P, FX=FX, lotes=pd.DataFrame(lotes))


def simular_indice(lotes: pd.DataFrame, indice_rep: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Qué valdría hoy si cada compra se hubiera invertido en el índice (en moneda de reporte)."""
    s = indice_rep.reindex(idx).ffill().bfill()
    unidades = pd.Series(0.0, index=idx)
    for _, l in lotes.iterrows():
        d = l["Día efectivo"]
        if d in s.index and s[d] > 0:
            unidades[idx >= d] += l["Costo reporte"] / s[d]
    return unidades * s


# ---------------------------------------------------------------------------
# Posiciones y métricas
# ---------------------------------------------------------------------------

def posiciones(lotes: pd.DataFrame, precio_actual: pd.Series, fx_hoy: pd.Series) -> pd.DataFrame:
    filas = []
    for a, g in lotes.groupby("Activo"):
        cant = g["Cantidad"].sum()
        p, f = float(precio_actual[a]), float(fx_hoy[a])
        costo = g["Costo reporte"].sum()
        valor = cant * p * f
        efecto_precio = float(((g["Cantidad"] * p - g["Costo local"]) * g["FX compra"]).sum())
        filas.append({
            "Activo": a, "Cantidad": cant, "Costo promedio": g["Costo local"].sum() / cant, "Precio actual": p,
            "Invertido": costo, "Valor actual": valor, "Ganancia": valor - costo,
            "Rentabilidad": valor / costo - 1 if costo else np.nan,
            "Efecto precio": efecto_precio, "Efecto cambiario": (valor - costo) - efecto_precio,
            "Primera compra": g["Fecha"].min(), "Días": (pd.Timestamp(date.today()) - g["Fecha"].min()).days,
        })
    pos = pd.DataFrame(filas)
    if len(pos):
        pos["Peso"] = pos["Valor actual"] / pos["Valor actual"].sum()
        pos = pos.sort_values("Valor actual", ascending=False).reset_index(drop=True)
    return pos


def xirr(flujos: list[tuple[pd.Timestamp, float]]) -> float:
    """Tasa interna de retorno anual con fechas (rentabilidad ponderada por dinero)."""
    if len(flujos) < 2 or not any(v > 0 for _, v in flujos) or not any(v < 0 for _, v in flujos):
        return np.nan
    d0 = min(d for d, _ in flujos)
    t = np.array([(d - d0).days / 365.25 for d, _ in flujos])
    v = np.array([v for _, v in flujos])
    f = lambda r: float(np.sum(v / (1 + r) ** t))
    try:
        return brentq(f, -0.9999, 100)
    except ValueError:
        return np.nan


def metricas_riesgo(r: pd.Series, rf: float, r_indice: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    anios = max((r.index[-1] - r.index[0]).days / 365.25, 1 / 365)
    per = len(r) / anios
    acum = (1 + r).cumprod()
    twr = acum.iloc[-1] ** (1 / anios) - 1 if anios >= 1 else acum.iloc[-1] - 1
    vol = r.std() * np.sqrt(per)
    out = {
        "TWR anual": twr, "TWR total": acum.iloc[-1] - 1, "Volatilidad": vol,
        "Sharpe": ((acum.iloc[-1] ** (1 / anios) - 1) - rf) / vol if vol > 0 else np.nan,
        "Caída máxima": float((acum / acum.cummax() - 1).min()),
        "Mejor día": float(r.max()), "Peor día": float(r.min()), "Años": anios,
    }
    if r_indice is not None:
        ri = r_indice.reindex(r.index).dropna()
        comun = r.loc[ri.index]
        if len(ri) > 20 and ri.var() > 0:
            out["Beta"] = float(np.cov(comun, ri)[0, 1] / ri.var())
    return out


# ---------------------------------------------------------------------------
# Recomendaciones
# ---------------------------------------------------------------------------
LIMITES = {
    "Conservador": dict(vol=0.15, cripto=0.0, cap=0.25),
    "Moderado": dict(vol=0.25, cripto=0.10, cap=0.30),
    "Arriesgado": dict(vol=0.40, cripto=0.25, cap=0.40),
}


def recomendaciones(pos: pd.DataFrame, riesgo: dict, perfil: dict | None, corr_media: float | None,
                    rent_port: float | None, rent_indice: float | None, moneda: str) -> list[dict]:
    """Reglas sencillas y explicables. nivel: 'alerta' | 'sugerencia' | 'bien'."""
    tipo = perfil["tipo"] if perfil else "Moderado"
    lim = LIMITES.get(tipo, LIMITES["Moderado"])
    cap = perfil["cap"] if perfil else lim["cap"]
    rec = []
    n = len(pos)

    # 1. Concentración por activo
    grandes = pos[pos["Peso"] > cap + 0.02]
    for _, f in grandes.iterrows():
        rec.append(dict(nivel="alerta", titulo=f"{f['Activo']} pesa {f['Peso']:.0%} de tu portafolio",
                        texto=f"Para un perfil {tipo.lower()} sugerimos no superar {cap:.0%} en un solo activo. Si a esa empresa "
                              f"le va mal, arrastra todo tu portafolio. Puedes llevar tus próximas compras a otros activos en lugar de vender."))
    if grandes.empty and n >= 3:
        rec.append(dict(nivel="bien", titulo="Ningún activo domina tu portafolio",
                        texto=f"Todos tus activos están por debajo de {cap:.0%}, el máximo sugerido para tu perfil."))

    # 2. Número de activos
    if n < 5:
        rec.append(dict(nivel="sugerencia", titulo=f"Tienes solo {n} activo{'s' if n != 1 else ''}",
                        texto="Con 5 a 15 activos de sectores distintos se reduce bastante el riesgo de que una sola empresa te afecte. "
                              "En la pestaña Explora puedes encontrar candidatos con buen puntaje."))

    # 3. Concentración por sector
    if "Sector" in pos:
        sec = pos.groupby("Sector")["Peso"].sum().sort_values(ascending=False)
        sec = sec[~sec.index.isin(["Cripto", "Sin sector"])]
        if len(sec) and sec.iloc[0] > 0.40:
            rec.append(dict(nivel="alerta" if sec.iloc[0] > 0.6 else "sugerencia",
                            titulo=f"{sec.iloc[0]:.0%} de tu dinero está en un solo sector ({sec.index[0]})",
                            texto="Las empresas de un mismo sector suelen subir y bajar juntas. Agregar otros sectores te protege "
                                  "cuando a ese le va mal."))

    # 4. Criptomonedas según perfil
    if "Tipo" in pos:
        w_cri = pos.loc[pos["Tipo"] == "Cripto", "Peso"].sum()
        if w_cri > lim["cripto"] + 0.02:
            rec.append(dict(nivel="alerta" if tipo == "Conservador" else "sugerencia",
                            titulo=f"Las criptomonedas son {w_cri:.0%} de tu portafolio",
                            texto=(f"Para un perfil {tipo.lower()} sugerimos máximo {lim['cripto']:.0%}. " if lim["cripto"] > 0 else
                                   "Para un perfil conservador sugerimos no tener criptomonedas o tener muy poco. ")
                                  + "Pueden caer más de 50% en pocos meses."))

    # 5. Riesgo total vs perfil
    vol = riesgo.get("Volatilidad")
    if vol is not None and vol == vol:
        if vol > lim["vol"]:
            rec.append(dict(nivel="alerta", titulo=f"Tu portafolio se mueve más de lo que encaja con tu perfil ({vol:.0%} al año)",
                            texto=f"Para un perfil {tipo.lower()} lo esperable es cerca de {lim['vol']:.0%} o menos. Reducir los activos "
                                  "más volátiles o sumar activos estables (por ejemplo, empresas de servicios públicos) lo acercaría."))
        else:
            rec.append(dict(nivel="bien", titulo=f"El nivel de riesgo encaja con tu perfil ({vol:.0%} de volatilidad anual)",
                            texto=f"Está por debajo del {lim['vol']:.0%} que consideramos razonable para un perfil {tipo.lower()}."))

    # 6. Diversificación real (correlación)
    if corr_media is not None and corr_media == corr_media and n >= 3:
        if corr_media > 0.6:
            rec.append(dict(nivel="sugerencia", titulo="Tus activos tienden a moverse juntos",
                            texto=f"La correlación promedio es {corr_media:.2f} (1 = se mueven igual). Aunque tengas varios activos, "
                                  "en una caída bajarían casi todos a la vez. Busca activos de otros sectores o países."))
        elif corr_media < 0.35:
            rec.append(dict(nivel="bien", titulo="Buena diversificación",
                            texto=f"Tus activos se mueven de forma bastante independiente (correlación promedio {corr_media:.2f})."))

    # 7. Posiciones con pérdidas fuertes
    for _, f in pos[pos["Rentabilidad"] < -0.25].iterrows():
        rec.append(dict(nivel="sugerencia", titulo=f"{f['Activo']} acumula {f['Rentabilidad']:.0%}",
                        texto="No es razón para vender por miedo, pero sí para revisar si la empresa sigue siendo buena: "
                              "mírala en la pestaña Analiza y decide con calma si mantenerla."))

    # 8. Activos con puntaje bajo
    if "Puntaje" in pos:
        bajos = pos[(pos["Puntaje"] < 40) & pos["Puntaje"].notna()]
        if len(bajos):
            rec.append(dict(nivel="sugerencia", titulo=f"Puntaje bajo en el radar: {', '.join(bajos['Activo'])}",
                            texto="Frente a empresas similares se ven caras o con fundamentales débiles. Vale la pena revisarlas."))

    # 9. Contra el índice
    if rent_port is not None and rent_indice is not None and rent_port == rent_port and rent_indice == rent_indice:
        dif = rent_port - rent_indice
        if dif < -0.03:
            rec.append(dict(nivel="sugerencia", titulo=f"Vas {abs(dif) * 100:.1f} puntos al año por debajo del S&P 500",
                            texto=f"Con las mismas compras en el S&P 500 (medido en {moneda}) te habría ido mejor. Un fondo indexado "
                                  "puede ser una buena base del portafolio, con algunas acciones elegidas alrededor."))
        elif dif > 0.03:
            rec.append(dict(nivel="bien", titulo=f"Le vas ganando al S&P 500 por {dif * 100:.1f} puntos al año",
                            texto="Tus compras han rendido más que si hubieras invertido lo mismo en el índice."))

    orden = {"alerta": 0, "sugerencia": 1, "bien": 2}
    return sorted(rec, key=lambda x: orden[x["nivel"]])
