# ======================================
# APP STREAMLIT – ANALÍTICA FEB (CSV)
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import norm
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
pd.options.display.float_format = '{:.2f}'.format
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import streamlit as st
#from pdf_export import build_pdf_report
import os
import vl_convert as vlc

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.stats import norm

#from pdf_export import build_pdf_report

# Decimales

MAX_DECIMALS = 2

def format_max_2_decimals(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, (float, np.floating)):
        txt = f"{float(x):.{MAX_DECIMALS}f}".rstrip("0").rstrip(".")
        return "0" if txt in {"-0", "-0.0", ""} else txt
    return x


def fmt_num(x, decimals=MAX_DECIMALS, suffix=""):
    if pd.isna(x):
        return "—"
    val = float(x)
    txt = f"{val:.{decimals}f}".rstrip("0").rstrip(".")
    if txt in {"-0", "-0.0", ""}:
        txt = "0"
    return f"{txt}{suffix}"


def st_dataframe_2d(df, **kwargs):
    if isinstance(df, pd.Series):
        st.dataframe(df.to_frame().style.format(format_max_2_decimals), **kwargs)
    elif isinstance(df, pd.DataFrame):
        st.dataframe(df.style.format(format_max_2_decimals), **kwargs)
    else:
        st.dataframe(df, **kwargs)

# Exportación en PDF
#from fpdf import FPDF
#from datetime import datetime


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_x(self.l_margin)
        self.cell(self.epw, 10, "Informe de Analisis de Baloncesto", align="C")
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def clean_text(text):
    if text is None:
        return ""
    text = str(text)

    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "•": "-",
        "·": "-",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N",
        "⚪": "",
        "⚫": "",
        "🟣": "",
        "📊": "",
        "📄": "",
        "🏀": "",
        "👥": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def write_block(pdf, text, font_family="Helvetica", style="", size=11, h=7):
    text = clean_text(text)
    if not text:
        return
    pdf.set_font(font_family, style, size)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, h, text)
    pdf.ln(1)


def build_pdf_report(title, subtitle="", sections=None, image_paths=None):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    write_block(pdf, title, style="B", size=16, h=10)

    if subtitle:
        write_block(pdf, subtitle, size=11, h=8)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_x(pdf.l_margin)
    pdf.cell(pdf.epw, 8, clean_text(f"Fecha de generacion: {datetime.now().strftime('%d/%m/%Y %H:%M')}"))
    pdf.ln(10)

    if image_paths:
        for img in image_paths:
            if img:
                pdf.set_x(pdf.l_margin)
                pdf.image(img, x=pdf.l_margin, w=180)
                pdf.ln(6)

    if sections:
        for sec in sections:
            heading = sec.get("heading", "")
            body = sec.get("body", "")

            if heading:
                write_block(pdf, heading, style="B", size=12, h=8)

            if body:
                write_block(pdf, body, size=11, h=7)

            pdf.ln(1)

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)
    return pdf_bytes


def save_altair_chart(chart, path):
    png_data = vlc.vegalite_to_png(chart.to_dict())
    with open(path, "wb") as f:
        f.write(png_data)


# ==============================
# CONFIGURACIÓN STREAMLIT
# ==============================
st.set_page_config(
    page_title="Analítica FEB",
    page_icon="🏀",
    layout="wide"
)

# ==============================
# LOGIN BÁSICO
# ==============================
USERS = {
    "admin": "admin",
    "professor": "feb2026"
}

def login_page():
    st.title("🏀 Aplicación FEB – Inicio de sesión")

    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Entrar"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Credenciales incorrectas")

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if not st.session_state["logged_in"]:
        login_page()
        st.stop()

# ==============================
# CARGA DATOS(CSV)
# ==============================
@st.cache_data
def load_team_metrics():
    df_j = pd.read_csv("df_jornades_all.csv")
    df_r = pd.read_csv("liga_ranks.csv")

    num_cols = ["POS","PosesionesTEAM","OER","DER","eFG","ORB","AST","TOV","PPP","NET"]
    for c in num_cols:
        if c in df_j.columns:
            df_j[c] = pd.to_numeric(df_j[c], errors="coerce")
        if c in df_r.columns:
            df_r[c] = pd.to_numeric(df_r[c], errors="coerce")

    df_j["Jornada_num"] = (
        df_j["Jornada"].astype(str).str.replace("J","", regex=False).astype(int)
    )

    return df_j, df_r
@st.cache_data
def load_quintets():
    dfq = pd.read_csv("df_total_comb_jug.csv")

    # Asegura tipos numéricos
    num_cols = ["pos","plus_minus","puntos_anotats_total","puntos_rebuts_total","OER","DER","NET",
                "plus_minus_total","pos_total"]
    for c in num_cols:
        if c in dfq.columns:
            dfq[c] = pd.to_numeric(dfq[c], errors="coerce")

    # Quintetos: df5 (si existe) o jugadora_5 no nula
    if "tipus_df" in dfq.columns:
        dfq = dfq[dfq["tipus_df"].astype(str).str.lower().eq("df5")].copy()
    else:
        dfq = dfq[dfq["jugadora_5"].notna()].copy()

    # Identificador de quinteto (string)
    players_cols = ["jugadora_1","jugadora_2","jugadora_3","jugadora_4","jugadora_5"]
    for c in players_cols:
        if c in dfq.columns:
            dfq[c] = dfq[c].astype(str)

    dfq["Quintet"] = dfq[players_cols].agg(" | ".join, axis=1)

    # KPI derivado útil
    dfq["PM_100"] = np.where(dfq["pos"] > 0, (dfq["plus_minus"] / dfq["pos"]) * 100, np.nan)

    return dfq
    
@st.cache_data
def load_combos_all():
    dfc = pd.read_csv("df_total_comb_jug.csv")

    # Numéricas (añade/quita si quieres)
    num_cols = [
        "pos","plus_minus","OER","DER","NET","PPP",
        "puntos_anotats_total","puntos_rebuts_total",
        "pos_total","plus_minus_total"
    ]
    for c in num_cols:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    # Asegura columnas de jugadoras como string (importante para el join)
    for i in range(1, 6):
        col = f"jugadora_{i}"
        if col in dfc.columns:
            dfc[col] = dfc[col].astype(str)

    # Normaliza tipus_df si existe
    if "tipus_df" in dfc.columns:
        dfc["tipus_df"] = dfc["tipus_df"].astype(str).str.lower()

    return dfc




@st.cache_resource
def load_win_model():
    model_path = Path(__file__).resolve().parent / "winprob_model.joblib"
    if not model_path.exists():
        return None, None
    pack = joblib.load(model_path)
    return pack["model"], pack["features"]


def predict_winprob_ml(teamA_row, teamB_row, home=0):
    model, feats = load_win_model()
    if model is None:
        return None

    x = {
        "NET_diff": float(teamA_row["NET"]) - float(teamB_row["NET"]),
        "OER_diff": float(teamA_row["OER"]) - float(teamB_row["OER"]),
        "DER_diff": float(teamA_row["DER"]) - float(teamB_row["DER"]),
        "eFG_diff": float(teamA_row["eFG"]) - float(teamB_row["eFG"]),
        "ORB_diff": float(teamA_row["ORB"]) - float(teamB_row["ORB"]),
        "AST_diff": float(teamA_row["AST"]) - float(teamB_row["AST"]),
        "TOV_diff": float(teamA_row["TOV"]) - float(teamB_row["TOV"]),
        "POS_diff": float(teamA_row["POS"]) - float(teamB_row["POS"]),
        "home": float(home),
    }

    X = pd.DataFrame([x])[feats]
    p = model.predict_proba(X)[:, 1][0]
    return float(p)


@st.cache_data
def load_players_boxscore():
    dfp = pd.read_csv("boxscore_total_temporada_tots_equips.csv")

    # Normaliza strings
    for c in ["NOM", "TEAM"]:
        if c in dfp.columns:
            dfp[c] = dfp[c].astype(str).str.strip()

    # Convierte numéricos (robusto)
    num_cols = [
        "GP","MIN","PTS","TREB","ASIS","PER","REC","VAL","PLUSMINUS",
        "poss_used","PPP",
        "MIN_pg","PTS_pg","TREB_pg","ASIS_pg","PER_pg","REC_pg","VAL_pg"
    ]
    # porcentajes (pueden venir como strings)
    pct_cols = ["eFG%", "TS%"]

    for c in num_cols + pct_cols:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    # Derivadas útiles (por si faltan)
    if "VAL_pg" not in dfp.columns and ("VAL" in dfp.columns and "GP" in dfp.columns):
        dfp["VAL_pg"] = dfp["VAL"] / dfp["GP"]

    if "MIN_pg" not in dfp.columns and ("MIN" in dfp.columns and "GP" in dfp.columns):
        dfp["MIN_pg"] = dfp["MIN"] / dfp["GP"]

    # Limpieza mínima: sin nombre o equipo
    dfp = dfp.dropna(subset=["NOM", "TEAM"]).copy()

    return dfp

@st.cache_data
def load_eventos_tiros():
    df = pd.read_csv("eventos_todos_equipos.csv")

    # Texto limpio
    for c in ["Jugador", "Accion", "Jornada", "TEAM", "tipo", "Rol", "bloc", "costat"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # numéricas
    for c in ["quarter", "pos_x", "pos_y", "x_plot", "y_plot", "made"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # made como entero
    if "made" in df.columns:
        df["made"] = df["made"].fillna(0).astype(int)

    return df


# ==============================
# FUNCIONES EXTRA
# ==============================

def compute_win_loss_profile(df_team):
    """
    Analiza las diferencias entre partidos ganados y perdidos.
    Gestiona casos especiales:
    - Equipoo invicto
    - Equipoo sin victorias
    """

    if "Resultat" not in df_team.columns:
        return {"status": "no_column", "data": None}

    metrics = ["OER","DER","eFG","ORB","AST","TOV","PPP","POS"]
    metrics = [m for m in metrics if m in df_team.columns]

    df_w = df_team[df_team["Resultat"] == "W"]
    df_l = df_team[df_team["Resultat"] == "L"]

    # CASO 1: invicto
    if df_l.empty and not df_w.empty:
        return {"status": "all_wins", "data": df_w[metrics].mean(numeric_only=True)}

    # CASO 2: ninguna victoria
    if df_w.empty and not df_l.empty:
        return {"status": "all_losses", "data": df_l[metrics].mean(numeric_only=True)}

    # CASO 3: sin datos
    if df_w.empty and df_l.empty:
        return {"status": "no_data", "data": None}

    # CASO NORMAL
    rows = []
    for m in metrics:
        w_mean = pd.to_numeric(df_w[m], errors="coerce").mean()
        l_mean = pd.to_numeric(df_l[m], errors="coerce").mean()

        rows.append({
            "Metrica": m,
            "Win": w_mean,
            "Loss": l_mean,
            "Diff (W-L)": w_mean - l_mean
        })

    out = pd.DataFrame(rows)
    out["impact"] = out["Diff (W-L)"].abs()
    out = out.sort_values("impact", ascending=False).drop(columns="impact")

    return {"status": "ok", "data": out}

def _weighted_mean(series, weights):
    """
    Media ponderada robusta.
    Ignora NaN y pesos <= 0.
    """
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")

    mask = s.notna() & w.notna() & (w > 0)

    if mask.sum() == 0:
        return np.nan

    return np.average(s[mask], weights=w[mask])
from scipy.stats import norm

def win_prob_from_margin(margin_pts, sd=10, p_min=0.005, p_max=0.995):
    """
    Convierte un margen esperado (en puntos) a probabilidad con una Normal(0, sd).
    p_min/p_max evitan valores 0% o 100% exactos.
    """
    p = norm.cdf(margin_pts / sd)
    return float(np.clip(p, p_min, p_max))

def _fmt_val_rank(row, metric, decimals=2):
    """
    Devuelve el string 'valor (#rank)' si existe rank_metric en row.
    """
    val = row.get(metric, np.nan)
    rk = row.get(f"rank_{metric}", np.nan)

    # formato del valor
    if pd.isna(val):
        val_str = ""
    else:
        val_str = f"{float(val):.{decimals}f}"

    # formato del rank
    if pd.isna(rk):
        return val_str
    try:
        rk_int = int(rk)
        return f"{val_str} (#{rk_int})" if val_str != "" else f"(#{rk_int})"
    except Exception:
        return val_str

def previa_partido_ranks(df_r, equipo_a, equipo_b, sd=10):
    a = df_r[df_r["EQUIPO"] == equipo_a].iloc[0]
    b = df_r[df_r["EQUIPO"] == equipo_b].iloc[0]

    # Posesiones esperadas
    poss = np.nanmean([a.get("POS", np.nan), b.get("POS", np.nan)])

    # IMPORTANT:
    # NET es por 100 posesiones. Si queremos margen en puntos:
    net_a = float(a.get("NET", np.nan))
    net_b = float(b.get("NET", np.nan))
    net_diff = net_a - net_b

    # margen en puntos esperado (no lo mostraremos en el informe porque así lo has pedido)
    margin_pts = (net_diff / 100.0) * poss if pd.notna(poss) and pd.notna(net_diff) else np.nan

    # Probabilidad (nunca 100%)
    p_win = win_prob_from_margin(margin_pts, sd=sd) if pd.notna(margin_pts) else np.nan

    informe = f"""
PREVIA PARTIT

{a['Team']} vs {b['Team']}

Posesions esperades: {poss:.2f}
Probabilitat victòria {a['Team']}: {100*p_win:.2f} %
""".strip()

    # Tabla de métricas con rank al lado
    metrics = ["OER","DER","eFG","ORB","AST","TOV","PPP","POS","NET"]
    metrics = [m for m in metrics if m in df_r.columns]  # robust

    met_tbl = pd.DataFrame({
        "Metrica": metrics,
        a["Team"]: [_fmt_val_rank(a, m, decimals=2) for m in metrics],
        b["Team"]: [_fmt_val_rank(b, m, decimals=2) for m in metrics],
    })

    return {"informe": informe, "met_tbl": met_tbl}

def similar_players(df_players, player_name, k=5):

    features = [
        "PTS_pg",
        "TREB_pg",
        "ASIS_pg",
        "REC_pg",
        "PER_pg",
        "VAL_pg",
        "eFG%",
        "TS%"
    ]

    df = df_players.dropna(subset=features).copy()

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = NearestNeighbors(n_neighbors=k+1)
    model.fit(X_scaled)

    idx = df[df["NOM"] == player_name].index[0]

    distances, indices = model.kneighbors([X_scaled[idx]])

    similar = df.iloc[indices[0][1:]]

    return similar[["NOM","TEAM","PTS_pg","TREB_pg","ASIS_pg","VAL_pg"]]


def page_previa_partit():
    st.title("Previa del partido")

    df_j, df_r = load_team_metrics()
    teams = df_r[["EQUIPO","Team"]].dropna().sort_values("EQUIPO")

    colA, colB = st.columns(2)

    with colA:
        eq_a = st.selectbox(
            "Equipo A",
            teams["EQUIPO"].astype(int).tolist(),
            index=0,
            format_func=lambda x: f"{x} — {teams.loc[teams['EQUIPO']==x,'Team'].values[0]}"
        )

    with colB:
        eq_b = st.selectbox(
            "Equipo B",
            teams["EQUIPO"].astype(int).tolist(),
            index=1,
            format_func=lambda x: f"{x} — {teams.loc[teams['EQUIPO']==x,'Team'].values[0]}"
        )

    if eq_a == eq_b:
        st.warning("Elige dos equipos diferentes.")
        return

    # sd ajustable si quieres (más sd = probabilidades más moderadas)
    sd = st.slider("Incertidumbre del modelo (sd, puntos)", 5, 25, 12, 1)

    res = previa_partido_ranks(df_r, eq_a, eq_b, sd=sd)

    st.code(res["informe"])
    st_dataframe_2d(res["met_tbl"], use_container_width=True, hide_index=True)

def build_combo_table(dfc_team: pd.DataFrame, k: int, min_pos: int):
    """
    Devuelve un dataframe agregado por combinación de k jugadoras:
    - pos (sum)
    - plus_minus (sum)
    - OER/DER/NET/PPP (mean si existen)
    - PM_100 (derivado)
    """

    players_cols = [f"jugadora_{i}" for i in range(1, k + 1)]
    for c in players_cols:
        if c not in dfc_team.columns:
            return pd.DataFrame()

    # Filtra por dfk si está, si no por jugadora_k no nula (o no "nan")
    d = dfc_team.copy()
    if "tipus_df" in d.columns:
        d = d[d["tipus_df"].eq(f"df{k}")]
    else:
        d = d[d[f"jugadora_{k}"].notna()]

    if d.empty:
        return pd.DataFrame()

    # Crea etiqueta de combinación
    d["Combo"] = d[players_cols].agg(" | ".join, axis=1)

    # Agregaciones disponibles
    agg = {}
    if "pos" in d.columns:
        agg["pos"] = ("pos", "sum")
    if "plus_minus" in d.columns:
        agg["plus_minus"] = ("plus_minus", "sum")
    for m in ["OER", "DER", "NET", "PPP"]:
        if m in d.columns:
            agg[m] = (m, "mean")

    g = d.groupby("Combo", as_index=False).agg(**agg)

    # Filtros y derivados
    if "pos" in g.columns:
        g = g[g["pos"] >= min_pos].copy()

    if "pos" in g.columns and "plus_minus" in g.columns:
        g["PM_100"] = np.where(g["pos"] > 0, (g["plus_minus"] / g["pos"]) * 100, np.nan)
    else:
        g["PM_100"] = np.nan

    # Ordenación por NET (si está), si no por PM_100, y si no por OER
    if "NET" in g.columns:
        g = g.sort_values("NET", ascending=False)
    elif "PM_100" in g.columns:
        g = g.sort_values("PM_100", ascending=False)
    elif "OER" in g.columns:
        g = g.sort_values("OER", ascending=False)

    return g
    def _weighted_mean(series, weights):
        s = pd.to_numeric(series, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")

        m = (s.notna()) & (w.notna()) & (w > 0)
        if m.sum() == 0:
            return np.nan


def compute_on_off_player(dfc_team: pd.DataFrame, player_name: str, k: int = 5):
    """
    Calcula ON / OFF / DIFF para una jugadora usando lineups de k jugadoras (por defecto quintetos, df5).
    Devuelve un dataframe con métricas agregadas.
    """
    d = dfc_team.copy()

    # Selecciona df5 (quintetos) para evitar solapamientos
    if "tipus_df" in d.columns:
        d = d[d["tipus_df"].astype(str).str.lower().eq(f"df{k}")]
    else:
        d = d[d.get(f"jugadora_{k}", pd.Series([np.nan]*len(d))).notna()]

    if d.empty:
        return pd.DataFrame()

    # Columnas de jugadoras del quinteto
    players_cols = [f"jugadora_{i}" for i in range(1, k + 1)]
    for c in players_cols:
        if c in d.columns:
            d[c] = d[c].astype(str)

    # ON si la jugadora aparece en el quinteto
    on_mask = d[players_cols].apply(lambda row: player_name in row.values, axis=1)
    d_on = d[on_mask].copy()
    d_off = d[~on_mask].copy()

    def agg_block(dd: pd.DataFrame):
        out = {}

        # Posesiones y plus-minus totales (si existen)
        if "pos" in dd.columns:
            out["pos"] = pd.to_numeric(dd["pos"], errors="coerce").sum()
        else:
            out["pos"] = np.nan

        if "plus_minus" in dd.columns:
            out["plus_minus"] = pd.to_numeric(dd["plus_minus"], errors="coerce").sum()
        else:
            out["plus_minus"] = np.nan

        # Preferimos puntos totales si están: permite calcular un OER/DER "real"
        has_pts_for = "puntos_anotats_total" in dd.columns
        has_pts_against = "puntos_rebuts_total" in dd.columns
        has_pos = "pos" in dd.columns

        if has_pts_for and has_pos:
            pts_for = pd.to_numeric(dd["puntos_anotats_total"], errors="coerce").sum()
            out["PTS_FOR"] = pts_for
            out["OER"] = (pts_for / out["pos"] * 100) if out["pos"] and out["pos"] > 0 else np.nan
        else:
            out["PTS_FOR"] = np.nan
            out["OER"] = _weighted_mean(dd.get("OER", np.nan), dd.get("pos", np.nan)) if has_pos else pd.to_numeric(dd.get("OER", np.nan), errors="coerce").mean()

        if has_pts_against and has_pos:
            pts_against = pd.to_numeric(dd["puntos_rebuts_total"], errors="coerce").sum()
            out["PTS_AGAINST"] = pts_against
            out["DER"] = (pts_against / out["pos"] * 100) if out["pos"] and out["pos"] > 0 else np.nan
        else:
            out["PTS_AGAINST"] = np.nan
            out["DER"] = _weighted_mean(dd.get("DER", np.nan), dd.get("pos", np.nan)) if has_pos else pd.to_numeric(dd.get("DER", np.nan), errors="coerce").mean()

        # NET: mejor calcularlo como OER-DER (consistente)
        out["NET"] = out["OER"] - out["DER"] if pd.notna(out["OER"]) and pd.notna(out["DER"]) else (
            _weighted_mean(dd.get("NET", np.nan), dd.get("pos", np.nan)) if has_pos else pd.to_numeric(dd.get("NET", np.nan), errors="coerce").mean()
        )

        # PM por 100 posesiones (si tenemos PM y pos)
        if pd.notna(out["pos"]) and out["pos"] > 0 and pd.notna(out["plus_minus"]):
            out["PM_100"] = (out["plus_minus"] / out["pos"]) * 100
        else:
            out["PM_100"] = np.nan

        return out

    on = agg_block(d_on)
    off = agg_block(d_off)

    # DIFF = ON - OFF
    diff = {}
    for k in ["pos", "plus_minus", "OER", "DER", "NET", "PM_100"]:
        a, b = on.get(k, np.nan), off.get(k, np.nan)
        diff[k] = (a - b) if pd.notna(a) and pd.notna(b) else np.nan

    res = pd.DataFrame(
        [
            {"Grup": "ON", **on},
            {"Grup": "OFF", **off},
            {"Grup": "DIFF (ON-OFF)", **diff},
        ]
    )

    return res

def prepare_shot_data(eventos_df):
    df = eventos_df.copy()

    # coordenadas finales para dibujar
    df["x"] = pd.to_numeric(df["x_plot"], errors="coerce")
    df["y"] = pd.to_numeric(df["y_plot"], errors="coerce")

    # aseguramos tipo y made
    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip()

    if "made" in df.columns:
        df["made"] = pd.to_numeric(df["made"], errors="coerce").fillna(0).astype(int)

    # eliminar filas sin coordenadas útiles
    df = df.dropna(subset=["x", "y", "tipo"]).copy()

    # crear zona_tiro
    df["zona_tiro"] = df.apply(assign_shot_zone, axis=1)

    return df

def draw_vertical_half_court(ax=None, line_color="navy", lw=1.2):
    """
    Dibuja una media pista vertical en coordenadas:
    x: 0..47
    y: -25..25
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7))

    # Marco exterior
    ax.add_patch(Rectangle((0, -25), 47, 50, fill=False, ec=line_color, lw=lw))

    # Aro
    ax.add_patch(Circle((5.25, 0), 0.75, fill=False, ec=line_color, lw=lw))

    # Tablero
    ax.plot([4, 4], [-3, 3], color=line_color, lw=2)

    # Zona
    ax.add_patch(Rectangle((0, -8), 19, 16, fill=False, ec=line_color, lw=lw))

    # Semicírculo tiro libre
    ax.add_patch(Arc((19, 0), 12, 12, angle=0, theta1=-90, theta2=90,
                     ec=line_color, lw=lw))

    # Semicírculo restringida
    ax.add_patch(Arc((5.25, 0), 8, 8, angle=0, theta1=-90, theta2=90,
                     ec=line_color, lw=lw))

    # Triple
    ax.plot([0, 14], [-22, -22], color=line_color, lw=lw)
    ax.plot([0, 14], [22, 22], color=line_color, lw=lw)

    radio_triple = 23.75
    ang = np.degrees(np.arcsin(22 / radio_triple))
    ax.add_patch(Arc((5.25, 0), 2 * radio_triple, 2 * radio_triple,
                     angle=0, theta1=-ang, theta2=ang,
                     ec=line_color, lw=lw))

    ax.set_xlim(0, 47)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax



def plot_shot_map(data, title="Mapa de tir", add_legend=False):
    fig, ax = plt.subplots(figsize=(7, 8))
    draw_vertical_half_court(ax=ax)

    # Tiros de 2 anotados
    df = data[(data["tipo"] == "2PT") & (data["made"])]
    ax.scatter(df["x"], df["y"], marker='x', c='green', s=70, linewidths=1.5)

    # Tiros de 2 fallados
    df = data[(data["tipo"] == "2PT") & (~data["made"])]
    ax.scatter(df["x"], df["y"], marker='x', c='red', s=70, linewidths=1.5)

    # Tiros de 3 anotados
    df = data[(data["tipo"] == "3PT") & (data["made"])]
    ax.scatter(df["x"], df["y"], facecolors='none', edgecolors='green', s=70, linewidths=1.5)

    # Tiros de 3 fallados
    df = data[(data["tipo"] == "3PT") & (~data["made"])]
    ax.scatter(df["x"], df["y"], facecolors='none', edgecolors='red', s=70, linewidths=1.5)

    if add_legend:
        add_manual_legend(ax)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig

def add_manual_legend(ax):
    ax.scatter(36, 22, marker='x', c='green', s=60, linewidths=1.5)
    ax.text(37.5, 22, "2PT anotat", va="center", fontsize=10)

    ax.scatter(36, 19, marker='x', c='red', s=60, linewidths=1.5)
    ax.text(37.5, 19, "2PT fallat", va="center", fontsize=10)

    ax.scatter(36, 16, facecolors='none', edgecolors='green', s=60, linewidths=1.5)
    ax.text(37.5, 16, "3PT anotat", va="center", fontsize=10)

    ax.scatter(36, 13, facecolors='none', edgecolors='red', s=60, linewidths=1.5)
    ax.text(37.5, 13, "3PT fallat", va="center", fontsize=10)



def assign_shot_zone(row):
    """
    Clasifica un tiro en 10 zonas usando coordenadas de media pista:
    x: 0..47
    y: -25..25
    aro aprox en (5.25, 0)
    """
    x = row["x"]
    y = row["y"]
    tipo = row["tipo"]

    dist = np.sqrt((x - 5.25)**2 + (y - 0.0)**2)

    # 5 zonas de 3
    if tipo == "3PT":
        if abs(y) >= 19:
            return "Triple esquina izquierda" if y < 0 else "Triple esquina derecha"

        if y < -8:
            return "Triple ala izquierda"
        elif y > 8:
            return "Triple ala derecha"
        else:
            return "Triple frontal"

    # 5 zonas de 2
    else:
        if dist <= 4:
            return "Aro"

        if x <= 19 and abs(y) <= 8:
            return "Pintura"

        if y < -8:
            return "Media izquierda"
        elif y > 8:
            return "Media derecha"
        else:
            return "Media frontal"


ZONAS_ORDEN = [
    "Aro",
    "Pintura",
    "Media izquierda",
    "Media frontal",
    "Media derecha",
    "Triple esquina izquierda",
    "Triple ala izquierda",
    "Triple frontal",
    "Triple ala derecha",
    "Triple esquina derecha"
]




def render_pdf_download_button(
    file_name: str,
    title: str,
    subtitle: str,
    sections: list
):
    pdf_data = build_pdf_report(
        title=title,
        subtitle=subtitle,
        sections=sections
    )

    st.download_button(
        label="📄 Descargar informe PDF",
        data=pdf_data,
        file_name=file_name,
        mime="application/pdf"
    )
    

# ==============================
# PÁGINA 0– HOME
# ==============================

def show_home():
    import streamlit as st
    
    st.title("🏀 Análisis y Predicción en Baloncesto FEB")

    st.markdown("""
    Bienvenido a esta aplicación de análisis avanzado de baloncesto.

    Este proyecto combina **análisis de datos** y **machine learning** para explorar el rendimiento de equipos y jugadoras, 
    así como predecir probabilidades de victoria en partidos.
    """)

    st.divider()

    st.header("🔍 ¿Qué puedes hacer en esta app?")
    
    st.markdown("""
    - 📊 Analizar estadísticas de equipos
    - ⚖️ Comparar rendimiento entre equipos
    - 👤 Explorar estadísticas de jugadoras
    - 🎯 Visualizar mapas de tiro
    - 🧠 Obtener probabilidades de victoria antes de un partido
    """)

    st.divider()

    st.header("🧭 ¿Cómo usar la aplicación?")
    
    st.markdown("""
    Utiliza el menú lateral para navegar entre las distintas secciones:

    - **Analítica de equipos** → métricas avanzadas por equipo  
    - **Comparativa de Equipos** → comparación directa entre todos los equipos
    - **Comparativa de Equipos L3** → comparación directa entre todos los equipos en las ultimas tres jornadas
    - **Previa del partido** → comparación directa entre dos equipos  
    - **Jugadoras** → análisis individual de jugadoras
    - **Mapas de tiro por jornada** → Mapas de tiro y rendimiento por zonaspor equipo y jugadoras
    """)

    st.divider()

    st.header("🤖 Modelo predictivo")
    
    st.markdown("""
    La app incluye un modelo de **regresión logística** entrenado para estimar la probabilidad de victoria.

    Características del modelo:
    - Variables agregadas por equipo (hasta t-1)
    - Diferencias entre equipos
    - Validación temporal
    """)

    st.divider()

    st.header("📁 Datos")
    
    st.markdown("""
    Los datos utilizados incluyen:
    - Estadísticas de equipos por partido
    - Eventos de juego
    - Métricas agregadas por jornada
    """)

    st.divider()

    st.success("👉 Selecciona una sección en el menú lateral para comenzar")

    
# ==============================
# PÁGINA 1 – ANALÍTICA EQUIPOS
# ==============================
def page_analitica_equips():
    st.title("🏀 Analítica de equipos")

    pdf_images = []

    # ==========================
    # DATOS BASE (equipos / jornadas / ranking)
    # ==========================
    df_j, df_r = load_team_metrics()
    teams = df_r[["EQUIPO", "Team"]].dropna().sort_values("EQUIPO")

    equipo_sel = st.selectbox(
        "Selecciona equipo",
        teams["EQUIPO"].astype(int).tolist(),
        format_func=lambda x: f"{x} — {teams.loc[teams['EQUIPO'] == x, 'Team'].values[0]}",
    )

    team_name = teams.loc[teams["EQUIPO"] == equipo_sel, "Team"].values[0]
    df_team = df_j[df_j["EQUIPO"] == equipo_sel].sort_values("Jornada_num").copy()
    tot = df_r[df_r["EQUIPO"] == equipo_sel].iloc[0]

    w = int((df_team["Resultat"] == "W").sum()) if "Resultat" in df_team.columns else 0
    l = int((df_team["Resultat"] == "L").sum()) if "Resultat" in df_team.columns else 0
    st.caption(f"**{team_name}** | Partidos: {len(df_team)} | Record: {w}-{l}")

    # Variables resumen para PDF
    combo_summary = "No disponible."
    onoff_summary = "No disponible."
    wl_summary = "No disponible o no comparable."
    metric_sel = None
    total_value = None

    # ==========================
    # 1) PERFIL vs LIGA (barras agrupadas)
    # ==========================
    st.subheader("Perfil vs media de la liga")

    metrics = ["OER", "DER", "eFG", "ORB", "AST", "TOV", "PPP", "POS"]
    metrics = [m for m in metrics if m in df_r.columns]

    league_avg = df_r[metrics].mean(numeric_only=True)

    df_profile = (
        pd.DataFrame(
            {
                "Metrica": metrics,
                "Equipo": [float(tot[m]) for m in metrics],
                "Lliga": [float(league_avg[m]) for m in metrics],
            }
        )
        .melt(id_vars="Metrica", var_name="Grup", value_name="Valor")
        .copy()
    )

    bar_profile = (
        alt.Chart(df_profile)
        .mark_bar()
        .encode(
            x=alt.X("Metrica:N", sort=metrics),
            y=alt.Y("Valor:Q", axis=alt.Axis(format=".2f")),
            xOffset="Grup:N",
            color=alt.Color("Grup:N", legend=alt.Legend(title="Grup")),
            tooltip=["Metrica", "Grup", alt.Tooltip("Valor:Q", format=".2f")],
        )
        .properties(height=320)
    )

    st.altair_chart(bar_profile, use_container_width=True)

    chart_path_profile = f"temp_profile_{equipo_sel}.png"
    save_altair_chart(bar_profile.properties(width=700, height=320), chart_path_profile)
    pdf_images.append(chart_path_profile)

    st.divider()

    # ==========================
    # TABLA DE RANKS DEL EQUIPO
    # ==========================
    st.subheader("📌 Ranking del equipo")

    df_ranks_team = pd.DataFrame(
        {
            "Metrica": metrics,
            "Rank": [tot.get(f"rank_{m}", np.nan) for m in metrics],
            "Valor": [tot.get(m, np.nan) for m in metrics],
        }
    )

    df_ranks_team["Rank"] = pd.to_numeric(df_ranks_team["Rank"], errors="coerce")
    df_ranks_team["Valor"] = pd.to_numeric(df_ranks_team["Valor"], errors="coerce")
    df_ranks_team = df_ranks_team.sort_values("Rank", na_position="last")

    df_ranks_team["Rank"] = df_ranks_team["Rank"].apply(
        lambda x: f"#{int(x)}" if pd.notna(x) else ""
    )

    st_dataframe_2d(df_ranks_team, use_container_width=True, hide_index=True)

    # ==========================
    # 2) EVOLUCIÓN POR JORNADA (selección de métrica)
    # ==========================
    st.subheader("Evolución por jornada (selecciona métrica)")

    metric_sel = st.selectbox(
        "Elige métrica",
        metrics,
        index=metrics.index("PPP") if "PPP" in metrics else 0,
    )

    total_value = float(tot[metric_sel])

    base = alt.Chart(df_team).encode(
        x=alt.X("Jornada:N", sort=alt.SortField("Jornada_num")),
        y=alt.Y(f"{metric_sel}:Q", title=metric_sel, axis=alt.Axis(format=".2f")),
        tooltip=[
            alt.Tooltip("Jornada:N", title="Jornada"),
            alt.Tooltip(f"{metric_sel}:Q", format=".2f"),
        ] + (
            [alt.Tooltip("Resultat:N", title="Resultat")]
            if "Resultat" in df_team.columns else []
        ),
    )

    pts = (
        base.mark_circle(size=80).encode(
            color=alt.Color(
                "Resultat:N",
                scale=alt.Scale(domain=["W", "L"], range=["#1B9E77", "#D95F02"]),
                legend=alt.Legend(title="Resultat"),
            )
        )
        if "Resultat" in df_team.columns
        else base.mark_circle(size=80)
    )

    line = base.mark_line(opacity=0.35)

    hline = (
        alt.Chart(pd.DataFrame({"y": [total_value]}))
        .mark_rule(strokeDash=[6, 4], strokeWidth=2)
        .encode(y="y:Q")
    )

    chart_evol = (line + pts + hline).properties(height=320)
    st.altair_chart(chart_evol, use_container_width=True)

    chart_path_evol = f"temp_evol_{equipo_sel}_{metric_sel}.png"
    save_altair_chart(chart_evol.properties(width=700, height=320), chart_path_evol)
    pdf_images.append(chart_path_evol)

    # ==========================================================
    # 3) ANÁLISIS DE COMBINACIONES (1..5 jugadoras) + ON/OFF
    # ==========================================================
    st.divider()
    st.subheader("👥 Análisis de combinaciones de jugadoras (1–5)")

    dfc = load_combos_all()

    dfc_team = dfc[dfc["equip"].astype(str) == str(team_name)].copy()
    if dfc_team.empty:
        dfc_team = dfc[
            dfc["equip"].astype(str).str.contains(str(team_name), case=False, na=False)
        ].copy()

    g = pd.DataFrame()

    if dfc_team.empty:
        st.info("No he encontrado combinaciones para este equipo en el archivo df_total_comb_jug.csv.")
    else:
        k_label_to_k = {
            "1 jugadora": 1,
            "2 jugadoras": 2,
            "3 jugadoras": 3,
            "4 jugadoras": 4,
            "Quintetos (5 jugadoras)": 5,
        }

        k_label = st.selectbox("Elige el tipo de análisis", list(k_label_to_k.keys()), index=4)
        k = k_label_to_k[k_label]

        min_pos_default = 60 if k == 5 else 30
        min_pos = st.slider("Mínimo de posesiones por combinación", 0, 300, min_pos_default, 5)

        g = build_combo_table(dfc_team, k=k, min_pos=min_pos)

        if g.empty:
            st.info("No hay datos para este filtro (prueba a bajar el mínimo de posesiones).")
        else:
            show_cols = ["Combo"]
            for c in ["pos", "OER", "DER", "NET", "plus_minus", "PM_100", "PPP"]:
                if c in g.columns:
                    show_cols.append(c)

            colA, colB = st.columns([1, 1])

            with colA:
                st.markdown("**Mejores combinaciones**")
                st_dataframe_2d(g[show_cols].head(15), use_container_width=True, hide_index=True)

            with colB:
                st.markdown("**Peores combinaciones**")
                sort_col = (
                    "NET" if "NET" in g.columns else
                    ("PM_100" if "PM_100" in g.columns else
                     ("OER" if "OER" in g.columns else None))
                )

                if sort_col:
                    st_dataframe_2d(
                        g[show_cols].tail(15).sort_values(sort_col, ascending=True),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st_dataframe_2d(
                        g[show_cols].tail(15),
                        use_container_width=True,
                        hide_index=True,
                    )

            top_combo_txt = []
            for _, row in g.head(5).iterrows():
                combo_txt = f"{row['Combo']}"
                if "NET" in row and pd.notna(row["NET"]):
                    combo_txt += f" | NET: {row['NET']:.2f}"
                if "pos" in row and pd.notna(row["pos"]):
                    combo_txt += f" | pos: {row['pos']:.0f}"
                top_combo_txt.append(combo_txt)
            combo_summary = "\n".join(top_combo_txt) if top_combo_txt else "No disponible."

            if "OER" in g.columns and "DER" in g.columns:
                st.markdown("### Mapa (OER vs DER) — tamaño = posesiones, color = NET")

                enc = {
                    "x": alt.X("OER:Q", title="OER (mitjana)", axis=alt.Axis(format=".2f")),
                    "y": alt.Y("DER:Q", title="DER (mitjana)", axis=alt.Axis(format=".2f")),
                    "tooltip": ["Combo"] + [
                        c for c in ["pos", "OER", "DER", "NET", "PM_100", "PPP"] if c in g.columns
                    ],
                }

                if "pos" in g.columns:
                    enc["size"] = alt.Size("pos:Q", title="Possessions")
                if "NET" in g.columns:
                    enc["color"] = alt.Color("NET:Q", title="NET (mitjana)")
                elif "PM_100" in g.columns:
                    enc["color"] = alt.Color("PM_100:Q", title="PM por 100")

                scatter = (
                    alt.Chart(g)
                    .mark_circle(opacity=0.75)
                    .encode(**enc)
                    .properties(height=420)
                    .interactive()
                )

                st.altair_chart(scatter, use_container_width=True)

                chart_path_scatter = f"temp_scatter_{equipo_sel}_{k}.png"
                save_altair_chart(scatter.properties(width=700, height=420), chart_path_scatter)
                pdf_images.append(chart_path_scatter)
            else:
                st.info("No puedo dibujar el mapa porque faltan columnas OER/DER en estas combinaciones.")

    # ==========================
    # 4) ON / OFF de una jugadora (basado en quintetos df5)
    # ==========================
    st.divider()
    st.subheader("🟣 On/Off de una jugadora (basado en quintetos)")

    df5_team = dfc_team.copy() if not dfc_team.empty else pd.DataFrame()

    if not df5_team.empty and "tipus_df" in df5_team.columns:
        df5_team = df5_team[df5_team["tipus_df"].astype(str).str.lower().eq("df5")].copy()

    if df5_team.empty:
        st.info("No he encontrado datos de quintetos (df5) para hacer el ON/OFF.")
        onoff = pd.DataFrame()
    else:
        player_cols = [f"jugadora_{i}" for i in range(1, 6) if f"jugadora_{i}" in df5_team.columns]

        if len(player_cols) == 0:
            st.info("No he encontrado columnas de jugadoras en el df5 para poder hacer ON/OFF.")
            onoff = pd.DataFrame()
        else:
            players = pd.unique(df5_team[player_cols].values.ravel("K"))
            players = [p for p in players if pd.notna(p) and str(p).lower() != "nan"]
            players = sorted(set(map(str, players)))

            if len(players) == 0:
                st.info("No he encontrado jugadoras en el df5 para poder hacer ON/OFF.")
                onoff = pd.DataFrame()
            else:
                player_sel = st.selectbox("Selecciona jugadora", players, key="onoff_player")

                min_pos_onoff = st.slider(
                    "Mínimo de posesiones para considerar el ON/OFF como fiable",
                    0, 500, 80, 10,
                    key="onoff_min_pos"
                )

                onoff = compute_on_off_player(dfc_team, player_sel, k=5)

                if onoff.empty:
                    st.info("No hay datos suficientes para esta jugadora.")
                else:
                    pos_on = (
                        float(onoff.loc[onoff["Grup"] == "ON", "pos"].values[0])
                        if "pos" in onoff.columns and len(onoff.loc[onoff["Grup"] == "ON", "pos"].values) > 0
                        else np.nan
                    )
                    pos_off = (
                        float(onoff.loc[onoff["Grup"] == "OFF", "pos"].values[0])
                        if "pos" in onoff.columns and len(onoff.loc[onoff["Grup"] == "OFF", "pos"].values) > 0
                        else np.nan
                    )

                    if pd.notna(pos_on) and pos_on < min_pos_onoff:
                        st.warning(f"ON con muestra pequeña: {pos_on:.2f} posesiones.")
                    if pd.notna(pos_off) and pos_off < min_pos_onoff:
                        st.warning(f"OFF con muestra pequeña: {pos_off:.2f} posesiones.")

                    onoff_show_cols = ["Grup", "pos", "OER", "DER", "NET", "PM_100", "plus_minus"]
                    onoff_show_cols = [c for c in onoff_show_cols if c in onoff.columns]

                    st_dataframe_2d(onoff[onoff_show_cols], use_container_width=True, hide_index=True)

                    rows_txt = []
                    cols_for_pdf = [c for c in ["Grup", "pos", "OER", "DER", "NET", "PM_100"] if c in onoff.columns]
                    for _, row in onoff[cols_for_pdf].iterrows():
                        vals = []
                        for col in cols_for_pdf:
                            val = row[col]
                            if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
                                vals.append(f"{col}: {val:.2f}")
                            else:
                                vals.append(f"{col}: {val}")
                        rows_txt.append(" | ".join(vals))
                    onoff_summary = "\n".join(rows_txt) if rows_txt else "No disponible."

                    options_bar = [m for m in ["NET", "PM_100", "OER", "DER"] if m in onoff.columns]
                    if len(options_bar) > 0:
                        metric_bar = st.selectbox("Visualiza ON vs OFF por", options_bar, key="onoff_metric_bar")

                        bar_onoff_df = onoff[onoff["Grup"].isin(["ON", "OFF"])][["Grup", metric_bar]].copy()

                        bar_onoff = (
                            alt.Chart(bar_onoff_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Grup:N", sort=["ON", "OFF"]),
                                y=alt.Y(f"{metric_bar}:Q", title=metric_bar, axis=alt.Axis(format=".2f")),
                                tooltip=["Grup", metric_bar],
                            )
                            .properties(height=240)
                        )

                        st.altair_chart(bar_onoff, use_container_width=True)

                        chart_path_onoff = f"temp_onoff_{equipo_sel}_{player_sel.replace(' ', '_')}.png"
                        save_altair_chart(bar_onoff.properties(width=700, height=240), chart_path_onoff)
                        pdf_images.append(chart_path_onoff)

                        diff_val = onoff.loc[onoff["Grup"] == "DIFF (ON-OFF)", metric_bar].values
                        if len(diff_val) > 0 and pd.notna(diff_val[0]):
                            st.caption(f"**DIFF {metric_bar} (ON−OFF): {float(diff_val[0]):.2f}**")

    # ==========================
    # 5) CAMBIOS CUANDO GANA VS PIERDE
    # ==========================
    st.divider()
    st.subheader("📊 Cambios en métricas cuando el equipo gana vs pierde")

    res_wl = compute_win_loss_profile(df_team)

    if res_wl["status"] == "no_column":
        st.info("No hay columna 'Resultado' disponible.")

    elif res_wl["status"] == "all_wins":
        st.success("Equipo invicto — no hay comparación posible W vs L.")
        st.caption("Medias en victorias:")
        st_dataframe_2d(res_wl["data"], use_container_width=True)
        wl_summary = "Equipo invicto. No hay comparacion posible entre victorias y derrotas."

    elif res_wl["status"] == "all_losses":
        st.warning("El equipo no ha ganado ningún partido.")
        st.caption("Medias en derrotas:")
        st_dataframe_2d(res_wl["data"], use_container_width=True)
        wl_summary = "El equipo no ha ganado ningun partido."

    elif res_wl["status"] == "no_data":
        st.info("No hay datos suficientes.")
        wl_summary = "No hay datos suficientes."

    else:
        wl = res_wl["data"]
        st_dataframe_2d(wl, use_container_width=True, hide_index=True)

        chart_wl = (
            alt.Chart(wl)
            .mark_bar()
            .encode(
                x=alt.X("Metrica:N", sort=alt.SortField("Diff (W-L)", order="descending")),
                y=alt.Y("Diff (W-L):Q", axis=alt.Axis(format=".2f")),
                color=alt.condition(
                    alt.datum["Diff (W-L)"] > 0,
                    alt.value("#1B9E77"),
                    alt.value("#D95F02"),
                ),
                tooltip=["Metrica", "Win", "Loss", "Diff (W-L)"],
            )
            .properties(height=350)
        )

        st.altair_chart(chart_wl, use_container_width=True)

        chart_path_wl = f"temp_wl_{equipo_sel}.png"
        save_altair_chart(chart_wl.properties(width=700, height=350), chart_path_wl)
        pdf_images.append(chart_path_wl)

        wl_lines = []
        for _, row in wl.head(10).iterrows():
            wl_lines.append(
                f"{row['Metrica']}: Win={row['Win']:.2f} | Loss={row['Loss']:.2f} | Diff={row['Diff (W-L)']:.2f}"
            )
        wl_summary = "\n".join(wl_lines) if wl_lines else "No disponible."

    # ==========================
    # EXPORTACION PDF
    # ==========================
    st.divider()
    st.subheader("📄 Exportar informe completo")

    metrics_summary = "\n".join(
        [
            f"{m}: {round(float(tot[m]), 2)} (Rank {tot.get(f'rank_{m}', '-')})"
            for m in metrics
            if pd.notna(tot.get(m))
        ]
    )

    evol_body = "No disponible."
    if metric_sel is not None and total_value is not None:
        evol_body = (
            f"Metrica seleccionada: {metric_sel}\n"
            f"Valor de referencia total: {total_value:.2f}"
        )

    pdf_data = build_pdf_report(
        title=f"Informe de Analitica - {team_name}",
        subtitle="Resumen completo de analitica de equipos",
        sections=[
            {
                "heading": "Equipo",
                "body": f"{team_name} (ID: {equipo_sel})",
            },
            {
                "heading": "Resumen general",
                "body": f"Partidos: {len(df_team)}\nRecord: {w}-{l}",
            },
            {
                "heading": "Metricas principales",
                "body": metrics_summary,
            },
            {
                "heading": "Evolucion por jornada",
                "body": evol_body,
            },
            {
                "heading": "Mejores combinaciones",
                "body": combo_summary,
            },
            {
                "heading": "Analisis ON/OFF",
                "body": onoff_summary,
            },
            {
                "heading": "Cambios en metricas cuando el equipo gana vs pierde",
                "body": wl_summary,
            },
        ],
        image_paths=pdf_images,
    )

    st.download_button(
        label="📥 Descargar informe PDF completo",
        data=pdf_data,
        file_name=f"analitica_completa_{team_name.replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

    # limpiar archivos temporales
    for img_path in pdf_images:
        if os.path.exists(img_path):
            os.remove(img_path)
            
# ==============================
# PÁGINA — COMPARATIVA DE EQUIPOS (SIMPLE)
# ==============================
def page_comparativa_equips():
    st.title("📊 Comparativa de equipos")

    pdf_images = []

    df_j, df_r = load_team_metrics()
    df = df_r.copy()

    # --- Solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    ban_exact = {"EQUIPO", "n_jornades", "W", "L", "WinPct", "W_pct"}
    options = [
        c for c in numeric_cols
        if (c not in ban_exact) and (not str(c).startswith("rank_"))
    ]

    if len(options) < 2:
        st.error("No hay suficientes métricas numéricas para hacer la comparativa.")
        return

    default_x = "OER" if "OER" in options else options[0]
    default_y = "DER" if "DER" in options else options[1]
    default_size = "POS" if "POS" in options else "(ninguno)"

    st.markdown("### Total temporada")

    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox("Eje X", options, index=options.index(default_x))
    with c2:
        y_var = st.selectbox("Eje Y", options, index=options.index(default_y))
    with c3:
        size_var = st.selectbox(
            "Tamaño (opc.)",
            ["(ninguno)"] + options,
            index=(["(ninguno)"] + options).index(default_size)
        )

    # datos para el gráfico
    cols_plot = ["EQUIPO", "Team", x_var, y_var]
    if size_var != "(ninguno)":
        cols_plot.append(size_var)

    df_plot = df[cols_plot].copy()

    df_plot[x_var] = pd.to_numeric(df_plot[x_var], errors="coerce")
    df_plot[y_var] = pd.to_numeric(df_plot[y_var], errors="coerce")
    if size_var != "(ninguno)":
        df_plot[size_var] = pd.to_numeric(df_plot[size_var], errors="coerce")

    df_plot = df_plot.dropna(subset=[x_var, y_var]).copy()

    if df_plot.empty:
        st.warning("No hay datos válidos para estas métricas.")
        return

    # --- dominio con margen
    x_min, x_max = float(df_plot[x_var].min()), float(df_plot[x_var].max())
    y_min, y_max = float(df_plot[y_var].min()), float(df_plot[y_var].max())

    def pad(minv, maxv, frac=0.06):
        span = maxv - minv
        if span == 0:
            span = abs(maxv) if maxv != 0 else 1.0
        return span * frac

    x_pad = pad(x_min, x_max, frac=0.06)
    y_pad = pad(y_min, y_max, frac=0.06)

    x_domain = [x_min - x_pad, x_max + x_pad]
    y_domain = [y_min - y_pad, y_max + y_pad]

    # --- base chart
    base = alt.Chart(df_plot).encode(
        x=alt.X(
            f"{x_var}:Q",
            title=x_var,
            scale=alt.Scale(domain=x_domain),
            axis=alt.Axis(format=".2f")
        ),
        y=alt.Y(
            f"{y_var}:Q",
            title=y_var,
            scale=alt.Scale(domain=y_domain),
            axis=alt.Axis(format=".2f")
        ),
        tooltip=[
            alt.Tooltip("Team:N", title="Equipo"),
            alt.Tooltip(f"{x_var}:Q", title=x_var, format=".2f"),
            alt.Tooltip(f"{y_var}:Q", title=y_var, format=".2f"),
        ] + (
            [alt.Tooltip(f"{size_var}:Q", title=size_var, format=".2f")]
            if size_var != "(ninguno)" else []
        ),
    )

    if size_var != "(ninguno)":
        points = base.mark_circle(opacity=0.9).encode(
            size=alt.Size(f"{size_var}:Q", title=size_var, legend=alt.Legend(title=size_var))
        )
    else:
        points = base.mark_circle(size=140, opacity=0.9)

    labels = base.mark_text(
        align="left",
        dx=7,
        dy=-7,
        fontSize=11
    ).encode(text="Team:N")

    chart = (points + labels).properties(height=620)

    st.altair_chart(chart, use_container_width=True)

    # guardar gráfico para PDF
    chart_path_comp = f"temp_comparativa_{x_var}_{y_var}.png"
    save_altair_chart(chart.properties(width=900, height=620), chart_path_comp)
    pdf_images.append(chart_path_comp)

    with st.expander("📋 Ver tabla"):
        show_cols = ["EQUIPO", "Team", x_var, y_var]
        if size_var != "(ninguno)":
            show_cols.append(size_var)
        st_dataframe_2d(df_plot[show_cols], use_container_width=True, hide_index=True)

    # ==========================
    # RESUMEN PARA PDF
    # ==========================
    df_rank_x = df_plot[["Team", x_var]].sort_values(x_var, ascending=False).head(5).copy()
    df_rank_y = df_plot[["Team", y_var]].sort_values(y_var, ascending=False).head(5).copy()

    top_x_txt = "\n".join(
        [f"{i+1}. {row['Team']} - {row[x_var]:.2f}" for i, (_, row) in enumerate(df_rank_x.iterrows())]
    )

    top_y_txt = "\n".join(
        [f"{i+1}. {row['Team']} - {row[y_var]:.2f}" for i, (_, row) in enumerate(df_rank_y.iterrows())]
    )

    size_txt = "Sin variable de tamaño."
    if size_var != "(ninguno)":
        df_rank_size = df_plot[["Team", size_var]].sort_values(size_var, ascending=False).head(5).copy()
        size_txt = "\n".join(
            [f"{i+1}. {row['Team']} - {row[size_var]:.2f}" for i, (_, row) in enumerate(df_rank_size.iterrows())]
        )

    # ==========================
    # EXPORTACION PDF
    # ==========================
    st.divider()
    st.subheader("📄 Exportar informe completo")

    pdf_data = build_pdf_report(
        title="Informe de Comparativa de Equipos",
        subtitle="Comparacion global de metricas de equipos",
        sections=[
            {
                "heading": "Configuracion del analisis",
                "body": (
                    f"Eje X: {x_var}\n"
                    f"Eje Y: {y_var}\n"
                    f"Tamano: {size_var}"
                ),
            },
            {
                "heading": f"Top 5 equipos en {x_var}",
                "body": top_x_txt,
            },
            {
                "heading": f"Top 5 equipos en {y_var}",
                "body": top_y_txt,
            },
            {
                "heading": "Resumen de la variable de tamano",
                "body": size_txt,
            },
            {
                "heading": "Descripcion",
                "body": (
                    "El grafico muestra la posicion relativa de todos los equipos en dos metricas "
                    "seleccionadas por el usuario, con la opcion de anadir una tercera variable en el tamano del punto."
                ),
            },
        ],
        image_paths=pdf_images,
    )

    st.download_button(
        label="📥 Descargar informe PDF completo",
        data=pdf_data,
        file_name=f"comparativa_equipos_{x_var}_{y_var}.pdf",
        mime="application/pdf",
    )

    # limpiar temporales
    for img_path in pdf_images:
        if os.path.exists(img_path):
            os.remove(img_path)

            
# ==============================
# PÁGINA — COMPARATIVA (ÚLTIMOS 3 PARTIDOS)
# ==============================
def page_comparativa_equips_last3():

    st.title("📊 Comparativa de equipos — Últimos 3 partidos")

    pdf_images = []

    df_j, df_r = load_team_metrics()

    # ---------------------------------
    # 1) PREPARAR ÚLTIMOS 3 PARTIDOS
    # ---------------------------------
    df_games = df_j.copy()

    df_games["Jornada_num"] = pd.to_numeric(
        df_games["Jornada_num"], errors="coerce"
    )

    df_last3 = (
        df_games
        .sort_values(["EQUIPO", "Jornada_num"])
        .groupby("EQUIPO")
        .tail(3)
    )

    numeric_cols = df_last3.select_dtypes(include=[np.number]).columns.tolist()

    exclude = {"EQUIPO", "Jornada_num"}
    metrics_cols = [c for c in numeric_cols if c not in exclude]

    df_last3_mean = (
        df_last3
        .groupby("EQUIPO")[metrics_cols]
        .mean()
        .reset_index()
    )

    teams = df_r[["EQUIPO", "Team"]]
    df_last3_mean = df_last3_mean.merge(teams, on="EQUIPO", how="left")

    df = df_last3_mean.copy()

    # ---------------------------------
    # 2) VARIABLES DISPONIBLES
    # ---------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    options = [c for c in numeric_cols if c != "EQUIPO"]

    if len(options) < 2:
        st.error("No hay suficientes métricas.")
        return

    default_x = "OER" if "OER" in options else options[0]
    default_y = "DER" if "DER" in options else options[1]
    default_size = "POS" if "POS" in options else "(ninguno)"

    st.markdown("### Rendimiento reciente (últimos 3 partidos)")

    c1, c2, c3 = st.columns(3)

    with c1:
        x_var = st.selectbox("Eje X", options, index=options.index(default_x))

    with c2:
        y_var = st.selectbox("Eje Y", options, index=options.index(default_y))

    with c3:
        size_var = st.selectbox(
            "Tamaño (opc.)",
            ["(ninguno)"] + options,
            index=(["(ninguno)"] + options).index(default_size)
        )

    # ---------------------------------
    # 3) DATASET PLOT
    # ---------------------------------
    plot_cols = ["EQUIPO", "Team", x_var, y_var] + (
        [size_var] if size_var != "(ninguno)" else []
    )
    df_plot = df[plot_cols].copy()

    df_plot[x_var] = pd.to_numeric(df_plot[x_var], errors="coerce")
    df_plot[y_var] = pd.to_numeric(df_plot[y_var], errors="coerce")
    if size_var != "(ninguno)":
        df_plot[size_var] = pd.to_numeric(df_plot[size_var], errors="coerce")

    df_plot = df_plot.dropna(subset=[x_var, y_var])

    if df_plot.empty:
        st.warning("No hay datos.")
        return

    # ---------------------------------
    # 4) DOMINIO DINÁMICO
    # ---------------------------------
    x_min, x_max = df_plot[x_var].min(), df_plot[x_var].max()
    y_min, y_max = df_plot[y_var].min(), df_plot[y_var].max()

    def pad(minv, maxv, frac=0.06):
        span = maxv - minv
        if span == 0:
            span = abs(maxv) if maxv != 0 else 1
        return span * frac

    x_pad = pad(x_min, x_max)
    y_pad = pad(y_min, y_max)

    x_domain = [x_min - x_pad, x_max + x_pad]
    y_domain = [y_min - y_pad, y_max + y_pad]

    # ---------------------------------
    # 5) GRÁFICO
    # ---------------------------------
    base = alt.Chart(df_plot).encode(
        x=alt.X(
            f"{x_var}:Q",
            title=x_var,
            scale=alt.Scale(domain=x_domain),
            axis=alt.Axis(format=".2f")
        ),
        y=alt.Y(
            f"{y_var}:Q",
            title=y_var,
            scale=alt.Scale(domain=y_domain),
            axis=alt.Axis(format=".2f")
        ),
        tooltip=[
            alt.Tooltip("Team:N", title="Equipo"),
            alt.Tooltip(f"{x_var}:Q", title=x_var, format=".2f"),
            alt.Tooltip(f"{y_var}:Q", title=y_var, format=".2f"),
        ] + (
            [alt.Tooltip(f"{size_var}:Q", title=size_var, format=".2f")]
            if size_var != "(ninguno)" else []
        ),
    )

    if size_var != "(ninguno)":
        points = base.mark_circle(opacity=0.9).encode(
            size=alt.Size(f"{size_var}:Q", legend=alt.Legend(title=size_var))
        )
    else:
        points = base.mark_circle(size=140, opacity=0.9)

    labels = base.mark_text(
        align="left",
        dx=7,
        dy=-7,
        fontSize=11
    ).encode(text="Team:N")

    chart = (points + labels).properties(height=620)

    st.altair_chart(chart, use_container_width=True)

    chart_path_last3 = f"temp_last3_{x_var}_{y_var}.png"
    save_altair_chart(chart.properties(width=900, height=620), chart_path_last3)
    pdf_images.append(chart_path_last3)

    # ---------------------------------
    # 6) TABLA
    # ---------------------------------
    with st.expander("📋 Ver tabla (últimos 3 partidos)"):
        show_cols = ["EQUIPO", "Team", x_var, y_var]
        if size_var != "(ninguno)":
            show_cols.append(size_var)

        st_dataframe_2d(
            df_plot[show_cols],
            use_container_width=True,
            hide_index=True
        )

    # ---------------------------------
    # 7) RESUMEN PARA PDF
    # ---------------------------------
    top_x = df_plot[["Team", x_var]].sort_values(x_var, ascending=False).head(5)
    top_y = df_plot[["Team", y_var]].sort_values(y_var, ascending=False).head(5)

    top_x_txt = "\n".join(
        [f"{i+1}. {row['Team']} - {row[x_var]:.2f}" for i, (_, row) in enumerate(top_x.iterrows())]
    )

    top_y_txt = "\n".join(
        [f"{i+1}. {row['Team']} - {row[y_var]:.2f}" for i, (_, row) in enumerate(top_y.iterrows())]
    )

    size_txt = "Sin variable de tamaño."
    if size_var != "(ninguno)":
        top_size = df_plot[["Team", size_var]].sort_values(size_var, ascending=False).head(5)
        size_txt = "\n".join(
            [f"{i+1}. {row['Team']} - {row[size_var]:.2f}" for i, (_, row) in enumerate(top_size.iterrows())]
        )

    # ---------------------------------
    # 8) EXPORTACIÓN PDF
    # ---------------------------------
    st.divider()
    st.subheader("📄 Exportar informe completo")

    pdf_data = build_pdf_report(
        title="Informe de Comparativa de Equipos - Ultimos 3 Partidos",
        subtitle="Comparacion del rendimiento reciente por equipo",
        sections=[
            {
                "heading": "Configuracion del analisis",
                "body": (
                    f"Eje X: {x_var}\n"
                    f"Eje Y: {y_var}\n"
                    f"Tamano: {size_var}\n"
                    f"Periodo analizado: ultimos 3 partidos de cada equipo"
                ),
            },
            {
                "heading": f"Top 5 equipos en {x_var}",
                "body": top_x_txt,
            },
            {
                "heading": f"Top 5 equipos en {y_var}",
                "body": top_y_txt,
            },
            {
                "heading": "Resumen de la variable de tamano",
                "body": size_txt,
            },
            {
                "heading": "Descripcion",
                "body": (
                    "El grafico compara a todos los equipos a partir de la media de sus "
                    "ultimos 3 partidos, permitiendo detectar tendencias recientes de rendimiento."
                ),
            },
        ],
        image_paths=pdf_images,
    )

    safe_x = str(x_var).replace("%", "pct").replace("/", "_")
    safe_y = str(y_var).replace("%", "pct").replace("/", "_")

    st.download_button(
        label="📥 Descargar informe PDF completo",
        data=pdf_data,
        file_name=f"comparativa_last3_{safe_x}_{safe_y}.pdf",
        mime="application/pdf",
    )

    for img_path in pdf_images:
        if os.path.exists(img_path):
            os.remove(img_path)
        
# ==========================================
# PREVIA PARTIDO — versión “staff-ready”
# - Valor + rank en la tabla
# - Probabilidad nunca 0%/100% + intervalo
# - Favoritooo textual
# - Ventajas automáticas (ranking)
# - Matchup atac vs defensa
# - Ritmo esperado (pace) vs media de la liga
# - Claves del partido (reglas simples)
# - (Opcional) Lineups clave + Impact player (ON/OFF) si tienes df_total_comb_jug.csv y helpers
# ==========================================

# --------------------------
# Helpers de formato / stats
# --------------------------
def _fmt_val_rank(row, metric, decimals=2, show_rank=True):
    """String 'valor (#rank)' si existe rank_metric; si no, solo valor."""
    val = row.get(metric, np.nan)
    rk = row.get(f"rank_{metric}", np.nan)

    if pd.isna(val):
        val_str = ""
    else:
        val_str = f"{float(val):.{decimals}f}"

    if (not show_rank) or pd.isna(rk):
        return val_str

    try:
        rk_int = int(rk)
        return f"{val_str} (#{rk_int})" if val_str else f"(#{rk_int})"
    except Exception:
        return val_str


def win_prob_from_margin(margin_pts, sd=12, p_min=0.005, p_max=0.995):
    """
    Converteix marge en puntos a probabilitat amb N(0, sd).
    Clamp para evitar 0% o 100% exactos.
    """
    if pd.isna(margin_pts):
        return np.nan
    p = norm.cdf(margin_pts / sd)
    return float(np.clip(p, p_min, p_max))


def prob_interval_from_margin(margin_pts, sd=12, halfwidth_pts=5, p_min=0.005, p_max=0.995):
    """
    Intervalo simple de probabilidad moviendo el margen ± halfwidth_pts.
    (Es una heurística útil para “staff”, no un IC estadístico formal.)
    """
    if pd.isna(margin_pts):
        return (np.nan, np.nan)
    p_lo = norm.cdf((margin_pts - halfwidth_pts) / sd)
    p_hi = norm.cdf((margin_pts + halfwidth_pts) / sd)
    p_lo = float(np.clip(p_lo, p_min, p_max))
    p_hi = float(np.clip(p_hi, p_min, p_max))
    return (p_lo, p_hi)


def favorite_label(p):
    if pd.isna(p):
        return "Sense dades"
    if p >= 0.75:
        return "Favorito clar"
    if p >= 0.60:
        return "Favorito moderat"
    if p >= 0.52:
        return "Lleu favorit"
    if p > 0.48:
        return "Partit igualat"
    return "Desavantatge"


def pace_label(poss, league_pace):
    if pd.isna(poss) or pd.isna(league_pace):
        return "Ritmo desconegut"
    if poss >= league_pace + 2:
        return "Ritmo alt"
    if poss <= league_pace - 2:
        return "Ritmo lent"
    return "Ritmo mitjà"


# --------------------------
# Core: previa + tablas
# --------------------------
def build_previa_partit(df_r, equipo_a, equipo_b):
    """
    Devuelve un dict con:
      - headline (texto corto)
      - tables: metrics table con valor+rank
      - advantages: listas de ventajas por ranking
      - matchup: edges de ataque/defensa
      - insights: claves del partido (heurísticas)
      - p_win + interval
    """
    a = df_r[df_r["EQUIPO"] == equipo_a].iloc[0]
    b = df_r[df_r["EQUIPO"] == equipo_b].iloc[0]

    # posesiones esperadas y pace de liga
    poss = np.nanmean([a.get("POS", np.nan), b.get("POS", np.nan)])
    league_pace = pd.to_numeric(df_r.get("POS", np.nan), errors="coerce").mean()

    # NET es por 100 posesiones -> margen en puntos ≈ (NET_diff/100)*poss
    net_a = pd.to_numeric(a.get("NET", np.nan), errors="coerce")
    net_b = pd.to_numeric(b.get("NET", np.nan), errors="coerce")
    net_diff = net_a - net_b
    margin_pts = (net_diff / 100.0) * poss if pd.notna(net_diff) and pd.notna(poss) else np.nan

    SD_MODEL = 12  # incertesa fixa del model (no visible a UI)

    p_win = win_prob_from_margin(margin_pts, sd=SD_MODEL)
    p_lo, p_hi = (np.nan, np.nan)  # ja no s’utilitza

    fav = favorite_label(p_win)
    pace_txt = pace_label(poss, league_pace)

    # Tabla de métricas (valor + rank)
    metrics = ["OER", "DER", "eFG", "ORB", "AST", "TOV", "PPP", "POS", "NET"]
    metrics = [m for m in metrics if m in df_r.columns]

    def _decimals(m):
        return 1 if m == "POS" else 2

    met_tbl = pd.DataFrame({
    "Metrica": metrics,
    a["Team"]: [_fmt_val_rank(a, m, decimals=_decimals(m), show_rank=True) for m in metrics],
    b["Team"]: [_fmt_val_rank(b, m, decimals=_decimals(m), show_rank=True) for m in metrics],
    })

    # Ventajas automáticas (ranking)
    adv_a, adv_b = [], []
    for m in ["OER", "DER", "eFG", "ORB", "AST", "TOV", "PPP"]:
        ra = pd.to_numeric(a.get(f"rank_{m}", np.nan), errors="coerce")
        rb = pd.to_numeric(b.get(f"rank_{m}", np.nan), errors="coerce")
        if pd.isna(ra) or pd.isna(rb):
            continue
        # Ranking bajo = mejor
        if ra < rb:
            adv_a.append(m)
        elif rb < ra:
            adv_b.append(m)

    # Matchup ataque vs defensa (heurística)
    oer_a = pd.to_numeric(a.get("OER", np.nan), errors="coerce")
    der_a = pd.to_numeric(a.get("DER", np.nan), errors="coerce")
    oer_b = pd.to_numeric(b.get("OER", np.nan), errors="coerce")
    der_b = pd.to_numeric(b.get("DER", np.nan), errors="coerce")

    attack_edge_A = oer_a - der_b if pd.notna(oer_a) and pd.notna(der_b) else np.nan
    attack_edge_B = oer_b - der_a if pd.notna(oer_b) and pd.notna(der_a) else np.nan

    matchup_tbl = pd.DataFrame({
        "Matchup": [f"Atac {a['Team']} vs Defensa {b['Team']}", f"Atac {b['Team']} vs Defensa {a['Team']}"],
        "Edge (pts/100)": [attack_edge_A, attack_edge_B],
    })

    # Claves del partido (reglas simples)
    # (Puedes afinarlas después; ahora son “buenas para empezar”)
    insights = []

    def _rank(x, m):
        return pd.to_numeric(x.get(f"rank_{m}", np.nan), errors="coerce")

    # eFG + OER
    if pd.notna(_rank(a, "eFG")) and pd.notna(_rank(b, "DER")):
        if _rank(a, "eFG") <= 3 and _rank(b, "DER") >= 8:
            insights.append(f"{a['Team']}: avantatge clar en eficàcia de tir (eFG alt) contra una defensa permissiva (DER).")
    if pd.notna(_rank(b, "eFG")) and pd.notna(_rank(a, "DER")):
        if _rank(b, "eFG") <= 3 and _rank(a, "DER") >= 8:
            insights.append(f"{b['Team']}: avantatge clar en eficàcia de tir (eFG alt) contra una defensa permissiva (DER).")

    # ORB (segundas oportunidatos)
    if pd.notna(_rank(a, "ORB")) and _rank(a, "ORB") <= 3:
        insights.append(f"{a['Team']}: pot generar segundas oportunidades (ORB molt alt).")
    if pd.notna(_rank(b, "ORB")) and _rank(b, "ORB") <= 3:
        insights.append(f"{b['Team']}: pot generar segundas oportunidades (ORB molt alt).")

    # TOV (control de pilota)
    # Asumimos que un 'TOV' menor es mejor, así que un rank bajo = bueno.
    if pd.notna(_rank(a, "TOV")) and pd.notna(_rank(b, "TOV")):
        if _rank(a, "TOV") <= 3 and _rank(b, "TOV") >= 10:
            insights.append(f"{a['Team']}: avantatge en control de pilota (TOV).")
        if _rank(b, "TOV") <= 3 and _rank(a, "TOV") >= 10:
            insights.append(f"{b['Team']}: avantatge en control de pilota (TOV).")

    # AST (creación)
    if pd.notna(_rank(a, "AST")) and _rank(a, "AST") <= 3:
        insights.append(f"{a['Team']}: molta creación de joc (AST alt) — atenció a col·lapses i extra-pass.")
    if pd.notna(_rank(b, "AST")) and _rank(b, "AST") <= 3:
        insights.append(f"{b['Team']}: molta creación de joc (AST alt) — atenció a col·lapses i extra-pass.")

    headline = {
        "team_a": a["Team"],
        "team_b": b["Team"],
        "poss": poss,
        "league_pace": league_pace,
        "pace_label": pace_txt,
        "favorite_label": fav,
        "p_win": p_win
    }

    return {
        "headline": headline,
        "met_tbl": met_tbl,
        "advantages": {"A": adv_a, "B": adv_b},
        "matchup_tbl": matchup_tbl,
        "insights": insights,
        # por si después quieres usarlo en otros bloques
        "margin_pts": margin_pts,
    }


# --------------------------
# (Opcional) Lineups clave e Impact player
# Requiere que ya tengas en el proyecto:
# - load_combos_all()
# - build_combo_table(dfc_team, k, min_pos)
# - compute_on_off_player(dfc_team, player_name, k=5)
# Si no existen o falla, la previa sigue funcionando.
# --------------------------
def try_lineups_and_impact(team_name, min_pos_lineups=50, topn=3):
    out = {"lineups": None, "impact": None, "error": None}
    try:
        dfc = load_combos_all()
        dfc_team = dfc[dfc["equip"].astype(str) == str(team_name)].copy()
        if dfc_team.empty:
            dfc_team = dfc[dfc["equip"].astype(str).str.contains(str(team_name), case=False, na=False)].copy()
        if dfc_team.empty:
            return out

        # lineups (df5)
        g5 = build_combo_table(dfc_team, k=5, min_pos=min_pos_lineups)
        if not g5.empty and "NET" in g5.columns:
            out["lineups"] = {
                "top": g5.head(topn),
                "bottom": g5.tail(topn).sort_values("NET", ascending=True)
            }

        # impact player: mejor DIFF NET (ON-OFF) entre jugadoras con muestra suficiente
        # (heurística rápida; puedes refinarla)
        df5_team = dfc_team.copy()
        if "tipus_df" in df5_team.columns:
            df5_team = df5_team[df5_team["tipus_df"].astype(str).str.lower().eq("df5")].copy()

        player_cols = [f"jugadora_{i}" for i in range(1, 6) if f"jugadora_{i}" in df5_team.columns]
        if df5_team.empty or len(player_cols) == 0:
            return out

        players = pd.unique(df5_team[player_cols].values.ravel("K"))
        players = [p for p in players if pd.notna(p) and str(p).lower() != "nan"]
        players = sorted(set(map(str, players)))

        best = None
        for p in players:
            onoff = compute_on_off_player(dfc_team, p, k=5)
            if onoff is None or onoff.empty:
                continue
            # muestra
            pos_on = float(onoff.loc[onoff["Grup"] == "ON", "pos"].values[0]) if "pos" in onoff.columns else 0
            pos_off = float(onoff.loc[onoff["Grup"] == "OFF", "pos"].values[0]) if "pos" in onoff.columns else 0
            if min(pos_on, pos_off) < 60:
                continue

            diff_net = onoff.loc[onoff["Grup"] == "DIFF (ON-OFF)", "NET"].values
            if len(diff_net) == 0 or pd.isna(diff_net[0]):
                continue
            diff_net = float(diff_net[0])

            if (best is None) or (diff_net > best["diff_net"]):
                best = {"player": p, "diff_net": diff_net, "pos_on": pos_on, "pos_off": pos_off}

        out["impact"] = best
        return out

    except Exception as e:
        out["error"] = str(e)
        return out


# ==============================
# PÁGINA 4 – PREVIA PARTIDO
# ==============================
def page_previa_partit():
    st.title("Previa del partido")

    pdf_images = []

    df_j, df_r = load_team_metrics()
    teams = df_r[["EQUIPO", "Team"]].dropna().sort_values("EQUIPO")

    colA, colB = st.columns(2)
    with colA:
        eq_a = st.selectbox(
            "Equipo A",
            teams["EQUIPO"].astype(int).tolist(),
            index=0,
            format_func=lambda x: f"{x} — {teams.loc[teams['EQUIPO']==x,'Team'].values[0]}",
        )
    with colB:
        eq_b = st.selectbox(
            "Equipo B",
            teams["EQUIPO"].astype(int).tolist(),
            index=1,
            format_func=lambda x: f"{x} — {teams.loc[teams['EQUIPO']==x,'Team'].values[0]}",
        )

    if eq_a == eq_b:
        st.warning("Elige dos equipos diferentes.")
        return

    res = build_previa_partit(df_r, eq_a, eq_b)

    h = res["headline"]
    team_a = h["team_a"]
    team_b = h["team_b"]
    a = df_r[df_r["EQUIPO"] == eq_a].iloc[0]
    b = df_r[df_r["EQUIPO"] == eq_b].iloc[0]

    p_ml = predict_winprob_ml(a, b, home=1)

    # --------------------------
    # CABECERA “staff”
    # --------------------------
    st.subheader("Resumen")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Ritmo esperado", fmt_num(h["poss"]) if pd.notna(h["poss"]) else "—", h["pace_label"])
    with c2:
        st.metric("Favorito", team_a, h["favorite_label"])
    with c3:
        st.metric("Prob. victoria A", fmt_num(100*h["p_win"], suffix=" %") if pd.notna(h["p_win"]) else "—")
    with c4:
        st.metric("Prob. victoria A (ML)", fmt_num(100*p_ml, suffix=" %") if pd.notna(p_ml) else "—")

    st.divider()

    # --------------------------
    # VENTAJAS
    # --------------------------
    st.subheader("Ventajas (según rankings)")
    advA = res["advantages"]["A"]
    advB = res["advantages"]["B"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{team_a}**")
        if len(advA) == 0:
            st.caption("Sin ventajas claras según los rankings seleccionados.")
        else:
            st.write(" • " + "\n • ".join(advA))
    with col2:
        st.markdown(f"**{team_b}**")
        if len(advB) == 0:
            st.caption("Sin ventajas claras según los rankings seleccionados.")
        else:
            st.write(" • " + "\n • ".join(advB))

    st.divider()

    # --------------------------
    # MATCHUP ataque vs defensa
    # --------------------------
    st.subheader("Emparejamiento ataque vs defensa")
    st_dataframe_2d(res["matchup_tbl"], use_container_width=True, hide_index=True)

    matchup_summary = "No disponible."
    mm = res["matchup_tbl"].copy()

    try:
        mm["Edge (pts/100)"] = pd.to_numeric(mm["Edge (pts/100)"], errors="coerce")

        chart_matchup = (
            alt.Chart(mm)
            .mark_bar()
            .encode(
                x=alt.X("Matchup:N", sort=None),
                y=alt.Y("Edge (pts/100):Q", axis=alt.Axis(format=".2f")),
                tooltip=["Matchup", alt.Tooltip("Edge (pts/100):Q", format=".2f")]
            )
            .properties(height=220)
        )

        st.altair_chart(chart_matchup, use_container_width=True)

        chart_path_matchup = f"temp_previa_matchup_{eq_a}_{eq_b}.png"
        save_altair_chart(chart_matchup.properties(width=800, height=220), chart_path_matchup)
        pdf_images.append(chart_path_matchup)

        matchup_summary = "\n".join(
            [
                f"{row['Matchup']}: {row['Edge (pts/100)']:.2f}"
                for _, row in mm.iterrows()
                if pd.notna(row["Edge (pts/100)"])
            ]
        ) or "No disponible."
    except Exception:
        pass

    st.divider()

    # --------------------------
    # CLAVES DEL PARTIDO
    # --------------------------
    st.subheader("Claves del partido")

    insights_summary = "No se han detectado claves claras con las reglas actuales."
    if len(res["insights"]) == 0:
        st.caption("No se han detectado claves claras con las reglas actuales.")
    else:
        insights_summary = "\n".join([f"- {s}" for s in res["insights"]])
        for s in res["insights"]:
            st.write(f"• {s}")

    st.divider()

    # --------------------------
    # TABLA DE MÉTRICAS COMPLETA
    # --------------------------
    st.subheader("Comparativa completa (valor + ranking)")
    st_dataframe_2d(res["met_tbl"], use_container_width=True, hide_index=True)

    # resumen tabla métricas para PDF
    metrics_table_summary = []
    for _, row in res["met_tbl"].iterrows():
        metrics_table_summary.append(
            f"{row['Metrica']}: {team_a} = {row[team_a]} | {team_b} = {row[team_b]}"
        )
    metrics_table_summary = "\n".join(metrics_table_summary)

    # --------------------------
    # (Opcional) Lineups clave + Impact player
    # --------------------------
    lineups_summary = "No disponible."
    impact_summary = "No disponible."

    with st.expander("👥 Lineups clau i Impact player (opcional)", expanded=False):
        extras = try_lineups_and_impact(team_a, min_pos_lineups=50, topn=3)

        if extras.get("error"):
            st.info("Este bloque es opcional. Parece que falta alguna función/archivo para activarlo.")
            st.caption(extras["error"])
            lineups_summary = "Bloque opcional no disponible por falta de funciones o datos."
            impact_summary = extras["error"]
        else:
            if extras.get("lineups") is None:
                st.caption("No hay lineups df5 disponibles (o no cumplen el mínimo de posesiones).")
                lineups_summary = "No hay lineups df5 disponibles o no cumplen el mínimo de posesiones."
            else:
                st.markdown("**Top quintetos (NET) — mínimo 50 posesiones**")
                st_dataframe_2d(extras["lineups"]["top"], use_container_width=True, hide_index=True)
                st.markdown("**Bottom quintetos (NET)**")
                st_dataframe_2d(extras["lineups"]["bottom"], use_container_width=True, hide_index=True)

                top_lines = []
                for _, row in extras["lineups"]["top"].iterrows():
                    combo = row["Combo"] if "Combo" in row else "Sin nombre"
                    net = row["NET"] if "NET" in row else np.nan
                    pos = row["pos"] if "pos" in row else np.nan
                    top_lines.append(
                        f"Top lineup: {combo} | NET={fmt_num(net)} | pos={fmt_num(pos)}"
                    )

                bottom_lines = []
                for _, row in extras["lineups"]["bottom"].iterrows():
                    combo = row["Combo"] if "Combo" in row else "Sin nombre"
                    net = row["NET"] if "NET" in row else np.nan
                    pos = row["pos"] if "pos" in row else np.nan
                    bottom_lines.append(
                        f"Bottom lineup: {combo} | NET={fmt_num(net)} | pos={fmt_num(pos)}"
                    )

                lineups_summary = "\n".join(top_lines + bottom_lines)

            imp = extras.get("impact")
            if imp is None:
                st.caption("No se ha podido estimar el Impact player (ON/OFF) con muestra suficiente.")
                impact_summary = "No se ha podido estimar el Impact player con muestra suficiente."
            else:
                st.success(
                    f"Impact player: **{imp['player']}** | DIFF NET (ON−OFF): **{imp['diff_net']:.2f}** "
                    f"| pos ON: {fmt_num(imp['pos_on'])} | pos OFF: {fmt_num(imp['pos_off'])}"
                )
                impact_summary = (
                    f"Impact player: {imp['player']} | "
                    f"DIFF NET (ON-OFF): {imp['diff_net']:.2f} | "
                    f"pos ON: {fmt_num(imp['pos_on'])} | "
                    f"pos OFF: {fmt_num(imp['pos_off'])}"
                )

    # --------------------------
    # EXPORTACION PDF
    # --------------------------
    st.divider()
    st.subheader("📄 Exportar informe completo")

    ventajas_a_txt = "\n".join([f"- {x}" for x in advA]) if advA else "Sin ventajas claras."
    ventajas_b_txt = "\n".join([f"- {x}" for x in advB]) if advB else "Sin ventajas claras."

    pdf_data = build_pdf_report(
        title=f"Previa del partido - {team_a} vs {team_b}",
        subtitle="Informe staff-ready de previa competitiva",
        sections=[
            {
                "heading": "Resumen",
                "body": (
                    f"Equipo A: {team_a}\n"
                    f"Equipo B: {team_b}\n"
                    f"Ritmo esperado: {fmt_num(h['poss'])} ({h['pace_label']})\n"
                    f"Favorito: {team_a} ({h['favorite_label']})\n"
                    f"Prob. victoria A: {fmt_num(100*h['p_win'], suffix=' %') if pd.notna(h['p_win']) else '—'}\n"
                    f"Prob. victoria A (ML): {fmt_num(100*p_ml, suffix=' %') if pd.notna(p_ml) else '—'}"
                ),
            },
            {
                "heading": f"Ventajas de {team_a}",
                "body": ventajas_a_txt,
            },
            {
                "heading": f"Ventajas de {team_b}",
                "body": ventajas_b_txt,
            },
            {
                "heading": "Emparejamiento ataque vs defensa",
                "body": matchup_summary,
            },
            {
                "heading": "Claves del partido",
                "body": insights_summary,
            },
            {
                "heading": "Comparativa completa de metricas",
                "body": metrics_table_summary,
            },
            {
                "heading": "Lineups clave",
                "body": lineups_summary,
            },
            {
                "heading": "Impact player",
                "body": impact_summary,
            },
        ],
        image_paths=pdf_images,
    )

    st.download_button(
        label="📥 Descargar informe PDF completo",
        data=pdf_data,
        file_name=f"previa_{team_a.replace(' ', '_')}_vs_{team_b.replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

    for img_path in pdf_images:
        if os.path.exists(img_path):
            os.remove(img_path)
            
# ==============================
# PÁGINA 4 — JUGADORAS
# ==============================
def page_jugadoras():
    st.title("🧍‍♂️ Jugadoras — Boxscore temporada")

    pdf_images = []

    dfp = load_players_boxscore().copy()

    # --------------------------
    # Filtros globales
    # --------------------------
    st.subheader("Filtros")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        teams = sorted(dfp["TEAM"].dropna().unique().tolist())
        team_sel = st.selectbox("Equipo", ["(Todos)"] + teams, index=0)

    with c2:
        min_gp = int(
            st.number_input("Mínimo de partidos (GP)", min_value=0, max_value=200, value=5, step=1)
        )

    with c3:
        if "MIN_pg" in dfp.columns:
            min_minpg = float(
                st.number_input(
                    "Mínimo de minutos por partido",
                    min_value=0.0,
                    max_value=40.0,
                    value=8.0,
                    step=0.5
                )
            )
        else:
            min_minpg = None
            st.caption("No hay MIN_pg")

    # Aplica filtros
    dff = dfp.copy()
    if team_sel != "(Todos)":
        dff = dff[dff["TEAM"] == team_sel].copy()
    if "GP" in dff.columns:
        dff = dff[dff["GP"] >= min_gp].copy()
    if min_minpg is not None and "MIN_pg" in dff.columns:
        dff = dff[dff["MIN_pg"] >= min_minpg].copy()

    if dff.empty:
        st.warning("No hay jugadoras con estos filtros.")
        return

    st.caption(f"Jugadoras mostradas: {len(dff)}")

    st.divider()

    # --------------------------
    # 1) Perfil de la jugadora
    # --------------------------
    st.subheader("Perfil de jugadora")

    dff = dff.sort_values(["TEAM", "NOM"]).copy()
    player_labels = (dff["NOM"] + " — " + dff["TEAM"]).tolist()

    idx_default = 0
    player_sel = st.selectbox("Selecciona jugadora", player_labels, index=idx_default)
    nom_sel, team_of_sel = player_sel.split(" — ", 1)

    p = dff[(dff["NOM"] == nom_sel) & (dff["TEAM"] == team_of_sel)].iloc[0]

    def _fmt(x, dec=2):
        return "—" if pd.isna(x) else f"{float(x):.{dec}f}"

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("MIN/partido", _fmt(p.get("MIN_pg", np.nan), 2))
    with k2:
        st.metric("PTS/partido", _fmt(p.get("PTS_pg", np.nan), 2))
    with k3:
        st.metric("REB/partido", _fmt(p.get("TREB_pg", np.nan), 2))
    with k4:
        st.metric("AST/partido", _fmt(p.get("ASIS_pg", np.nan), 2))
    with k5:
        st.metric("eFG%", _fmt(p.get("eFG%", np.nan), 2))
    with k6:
        st.metric("TS%", _fmt(p.get("TS%", np.nan), 2))

    # Tabla pequeña con stats extra
    extras_cols = [
        ("GP", 2), ("MIN", 2), ("PTS", 2), ("TREB", 2), ("ASIS", 2),
        ("REC", 2), ("PER", 2), ("VAL", 2), ("VAL_pg", 2),
        ("PPP", 2), ("poss_used", 2), ("PLUSMINUS", 2)
    ]
    rows = []
    for col, dec in extras_cols:
        if col in dff.columns:
            rows.append({"Metrica": col, "Valor": _fmt(p.get(col, np.nan), dec)})

    if rows:
        extras_df = pd.DataFrame(rows)
        st_dataframe_2d(extras_df, use_container_width=True, hide_index=True)
    else:
        extras_df = pd.DataFrame()

    st.divider()

    # --------------------------
    # 2) JUGADORAS SIMILARES (KNN)
    # --------------------------
    st.subheader("Jugadoras similares")

    similar = similar_players(dfp, p["NOM"], k=5)
    st_dataframe_2d(similar, use_container_width=True, hide_index=True)

    features = [
        "PTS_pg",
        "TREB_pg",
        "ASIS_pg",
        "REC_pg",
        "PER_pg",
        "VAL_pg",
        "eFG%",
        "TS%"
    ]

    df_sim = dfp.dropna(subset=features).copy()
    df_sim["selected"] = (
        (df_sim["NOM"] == p["NOM"]) &
        (df_sim["TEAM"] == p["TEAM"])
    )

    X = df_sim[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    df_sim["PC1"] = coords[:, 0]
    df_sim["PC2"] = coords[:, 1]

    fig = px.scatter(
        df_sim,
        x="PC1",
        y="PC2",
        color="selected",
        color_discrete_map={
            True: "red",
            False: "lightgray"
        },
        hover_data=["NOM", "TEAM", "PTS_pg", "VAL_pg"],
        title="Mapa de jugadoras (PCA)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Intentar guardar PCA para PDF (requiere kaleido)
    pca_pdf_status = "No exportado."
    try:
        chart_path_pca = f"temp_pca_{nom_sel.replace(' ', '_')}_{team_of_sel.replace(' ', '_')}.png"
        fig.write_image(chart_path_pca)
        pdf_images.append(chart_path_pca)
        pca_pdf_status = "Incluido en el PDF."
    except Exception:
        pca_pdf_status = "No se pudo incluir el PCA en el PDF (instala kaleido para exportar gráficos Plotly)."

    st.divider()

    # --------------------------
    # 3) Ranking de jugadoras
    # --------------------------
    st.subheader("Ranking de jugadoras")

    rank_options = []
    preferred = ["VAL_pg", "PTS_pg", "TREB_pg", "ASIS_pg", "eFG%", "TS%", "PPP", "MIN_pg"]
    for m in preferred:
        if m in dff.columns:
            rank_options.append(m)

    if not rank_options:
        pg_like = [c for c in dff.columns if c.endswith("_pg")]
        rank_options = pg_like if pg_like else [c for c in dff.columns if pd.api.types.is_numeric_dtype(dff[c])]

    metric_rank = st.selectbox("Ordenar por", rank_options, index=0)
    topn = st.slider("Top N", 5, 50, 15, 5)

    top = dff.sort_values(metric_rank, ascending=False).head(topn).copy()

    show_cols = ["NOM", "TEAM", "GP"]
    if "MIN_pg" in top.columns:
        show_cols.append("MIN_pg")
    show_cols += [metric_rank]

    st_dataframe_2d(top[show_cols], use_container_width=True, hide_index=True)

    # gráfico ranking
    ranking_bar_summary = "No disponible."
    try:
        bar_df = top[["NOM", metric_rank]].dropna().copy()
        bar_rank = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric_rank}:Q", title=metric_rank, axis=alt.Axis(format=".2f")),
                y=alt.Y("NOM:N", sort="-x", title="Jugadora"),
                tooltip=["NOM", alt.Tooltip(f"{metric_rank}:Q", format=".2f")],
            )
            .properties(height=420)
        )
        st.altair_chart(bar_rank, use_container_width=True)

        chart_path_rank = f"temp_rank_players_{metric_rank.replace('%', 'pct')}.png"
        save_altair_chart(bar_rank.properties(width=900, height=420), chart_path_rank)
        pdf_images.append(chart_path_rank)

        ranking_bar_summary = "\n".join(
            [f"{i+1}. {row['NOM']} - {row[metric_rank]:.2f}" for i, (_, row) in enumerate(bar_df.iterrows())]
        )
    except Exception:
        pass

    st.divider()

    # --------------------------
    # 4) Comparador Jugadora A vs B
    # --------------------------
    st.subheader("Comparativa A vs B")

    colA, colB = st.columns(2)

    all_players = dfp.copy()
    if team_sel != "(Todos)":
        all_players = all_players[all_players["TEAM"] == team_sel].copy()

    all_players = all_players.dropna(subset=["NOM", "TEAM"]).copy()
    all_players["label"] = all_players["NOM"] + " — " + all_players["TEAM"]
    labels_all = all_players["label"].sort_values().tolist()

    with colA:
        pA_lab = st.selectbox("Jugadora A", labels_all, index=0, key="jugA")
    with colB:
        pB_lab = st.selectbox("Jugadora B", labels_all, index=min(1, len(labels_all)-1), key="jugB")

    nomA, teamA = pA_lab.split(" — ", 1)
    nomB, teamB = pB_lab.split(" — ", 1)

    A = all_players[(all_players["NOM"] == nomA) & (all_players["TEAM"] == teamA)].iloc[0]
    B = all_players[(all_players["NOM"] == nomB) & (all_players["TEAM"] == teamB)].iloc[0]

    comp_metrics = []
    for m in ["MIN_pg", "PTS_pg", "TREB_pg", "ASIS_pg", "eFG%", "TS%", "VAL_pg", "PPP"]:
        if m in all_players.columns:
            comp_metrics.append(m)

    comp_rows = []
    for m in comp_metrics:
        dec = 3 if m == "PPP" else (1 if (m.endswith("_pg") or m in ["eFG%", "TS%", "VAL_pg", "MIN_pg"]) else 0)
        comp_rows.append({
            "Metrica": m,
            f"{nomA} ({teamA})": _fmt(A.get(m, np.nan), dec),
            f"{nomB} ({teamB})": _fmt(B.get(m, np.nan), dec),
        })

    comp_df = pd.DataFrame(comp_rows)
    st_dataframe_2d(comp_df, use_container_width=True, hide_index=True)

    # --------------------------
    # RESÚMENES PARA PDF
    # --------------------------
    player_profile_summary = (
        f"Jugadora seleccionada: {p['NOM']} ({p['TEAM']})\n"
        f"MIN/partido: {_fmt(p.get('MIN_pg', np.nan), 2)}\n"
        f"PTS/partido: {_fmt(p.get('PTS_pg', np.nan), 2)}\n"
        f"REB/partido: {_fmt(p.get('TREB_pg', np.nan), 2)}\n"
        f"AST/partido: {_fmt(p.get('ASIS_pg', np.nan), 2)}\n"
        f"eFG%: {_fmt(p.get('eFG%', np.nan), 2)}\n"
        f"TS%: {_fmt(p.get('TS%', np.nan), 2)}"
    )

    extras_summary = "No disponible."
    if not extras_df.empty:
        extras_summary = "\n".join(
            [f"{row['Metrica']}: {row['Valor']}" for _, row in extras_df.iterrows()]
        )

    similar_summary = "No disponible."
    if isinstance(similar, pd.DataFrame) and not similar.empty:
        similar_summary = "\n".join(
            [
                f"{i+1}. {row['NOM']} ({row['TEAM']}) | "
                f"PTS_pg={_fmt(row.get('PTS_pg', np.nan), 2)} | "
                f"TREB_pg={_fmt(row.get('TREB_pg', np.nan), 2)} | "
                f"ASIS_pg={_fmt(row.get('ASIS_pg', np.nan), 2)} | "
                f"VAL_pg={_fmt(row.get('VAL_pg', np.nan), 2)}"
                for i, (_, row) in enumerate(similar.iterrows())
            ]
        )

    comp_summary = "No disponible."
    if not comp_df.empty:
        comp_summary = "\n".join(
            [
                f"{row['Metrica']}: {nomA} ({teamA}) = {row[f'{nomA} ({teamA})']} | "
                f"{nomB} ({teamB}) = {row[f'{nomB} ({teamB})']}"
                for _, row in comp_df.iterrows()
            ]
        )

    filter_summary = (
        f"Equipo filtrado: {team_sel}\n"
        f"Minimo de partidos: {min_gp}\n"
        f"Minimo de minutos por partido: {min_minpg if min_minpg is not None else 'No disponible'}"
    )

    # --------------------------
    # EXPORTACIÓN PDF
    # --------------------------
    st.divider()
    st.subheader("📄 Exportar informe completo")

    safe_player = str(p["NOM"]).replace(" ", "_").replace("/", "_")

    pdf_data = build_pdf_report(
        title=f"Informe de Jugadoras - {p['NOM']}",
        subtitle="Resumen individual, similares, ranking y comparativa",
        sections=[
            {
                "heading": "Filtros aplicados",
                "body": filter_summary,
            },
            {
                "heading": "Perfil de la jugadora",
                "body": player_profile_summary,
            },
            {
                "heading": "Estadisticas adicionales",
                "body": extras_summary,
            },
            {
                "heading": "Jugadoras similares",
                "body": similar_summary,
            },
            {
                "heading": "Estado de exportacion del grafico PCA",
                "body": pca_pdf_status,
            },
            {
                "heading": f"Ranking de jugadoras por {metric_rank}",
                "body": ranking_bar_summary,
            },
            {
                "heading": f"Comparativa {nomA} vs {nomB}",
                "body": comp_summary,
            },
        ],
        image_paths=pdf_images,
    )

    st.download_button(
        label="📥 Descargar informe PDF completo",
        data=pdf_data,
        file_name=f"jugadoras_{safe_player}.pdf",
        mime="application/pdf",
    )

    # limpiar temporales
    for img_path in pdf_images:
        if os.path.exists(img_path):
            os.remove(img_path)
            
# ==============================
# PÁGINA 5 — MAPA DE TIRO
# ==============================

def save_matplotlib_figure(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")

def safe_text_filename(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("%", "pct")


def page_mapes_tir():
    st.title("🏀 Mapas de tir")

    pdf_images = []

    eventos = load_eventos_tiros()
    tiros = prepare_shot_data(eventos)

    # --------------------------
    # Filtros
    # --------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        jornades = sorted(
            tiros["Jornada"].dropna().unique(),
            key=lambda x: int(x.replace("J", ""))
        )
        jornada_sel = st.selectbox("Jornada", jornades)

    tiros_jornada = tiros[tiros["Jornada"] == jornada_sel].copy()

    with c2:
        if "TEAM" in tiros_jornada.columns:
            teams = sorted(tiros_jornada["TEAM"].dropna().unique().tolist())
            team_sel = st.selectbox("Equipo", teams)
            tiros_jornada = tiros_jornada[tiros_jornada["TEAM"] == team_sel].copy()
        else:
            team_sel = None
            st.caption("No hay columna TEAM en el dataset.")

    with c3:
        jugadores = sorted(tiros_jornada["Jugador"].dropna().unique().tolist())
        jugadora_sel = st.selectbox("Jugadora", jugadores)

    tiros_jugadora = tiros_jornada[tiros_jornada["Jugador"] == jugadora_sel].copy()

    st.divider()

    # --------------------------
    # KPIs simples
    # --------------------------
    k1, k2, k3, k4, k5 = st.columns(5)

    n_tiros = len(tiros_jugadora)
    encertats = int(tiros_jugadora["made"].sum()) if n_tiros > 0 else 0
    pct = 100 * encertats / n_tiros if n_tiros > 0 else 0

    int2 = (tiros_jugadora["tipo"] == "2PT").sum()
    made2 = ((tiros_jugadora["tipo"] == "2PT") & (tiros_jugadora["made"] == 1)).sum()

    int3 = (tiros_jugadora["tipo"] == "3PT").sum()
    made3 = ((tiros_jugadora["tipo"] == "3PT") & (tiros_jugadora["made"] == 1)).sum()

    with k1:
        st.metric("Tirs totals", fmt_num(n_tiros))
    with k2:
        st.metric("Encertats", fmt_num(encertats))
    with k3:
        st.metric("FG%", fmt_num(pct, suffix="%"))
    with k4:
        st.metric("2PT (Made/Int)", f"{fmt_num(made2)}/{fmt_num(int2)}")
    with k5:
        st.metric("3PT (Made/Int)", f"{fmt_num(made3)}/{fmt_num(int3)}")

    st.divider()

    # --------------------------
    # Mapas
    # --------------------------
    fig_j = None
    fig_t = None

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Mapa de la jugadora")
        if tiros_jugadora.empty:
            st.info("No hay tiros para esta jugadora.")
        else:
            fig_j = plot_shot_map(
                tiros_jugadora,
                title=f"Mapa de tiro — {jugadora_sel}",
                add_legend=True
            )
            st.pyplot(fig_j)

            path_j = f"temp_mapa_jugadora_{safe_text_filename(jugadora_sel)}_{safe_text_filename(jornada_sel)}.png"
            save_matplotlib_figure(fig_j, path_j)
            pdf_images.append(path_j)

    with colB:
        st.subheader("Mapa del equipo")
        if tiros_jornada.empty:
            st.info("No hay tiros para este equipo y jornada.")
        else:
            title_team = f"Mapa de tir — {team_sel}" if team_sel is not None else "Mapa de tir de l’equip"
            fig_t = plot_shot_map(
                tiros_jornada,
                title=title_team,
                add_legend=False
            )
            st.pyplot(fig_t)

            path_t = f"temp_mapa_equipo_{safe_text_filename(team_sel)}_{safe_text_filename(jornada_sel)}.png"
            save_matplotlib_figure(fig_t, path_t)
            pdf_images.append(path_t)

    # --------------------------
    # Rendimiento por zonas — jugadora
    # --------------------------
    st.subheader("Rendimiento por zonas — jugadora")

    zona_stats_jug = (
        tiros_jugadora.groupby("zona_tiro", as_index=False)
        .agg(
            encestados=("made", "sum"),
            tiros=("zona_tiro", "size")
        )
    ) if not tiros_jugadora.empty else pd.DataFrame(columns=["zona_tiro", "encestados", "tiros"])

    if not zona_stats_jug.empty:
        zona_stats_jug["FG%"] = (
            100 * zona_stats_jug["encestados"] / zona_stats_jug["tiros"]
        ).round(2)

        zona_stats_jug["zona_tiro"] = pd.Categorical(
            zona_stats_jug["zona_tiro"],
            categories=ZONAS_ORDEN,
            ordered=True
        )

        zona_stats_jug = zona_stats_jug.sort_values("zona_tiro")

    st_dataframe_2d(zona_stats_jug, use_container_width=True, hide_index=True)

    st.subheader("Volumen por zonas - jugadora")

    zona_jug_summary = "No disponible."
    try:
        fig_z = px.bar(
            zona_stats_jug,
            x="zona_tiro",
            y="tiros",
            hover_data=["encestados", "FG%"],
            title="Distribución de tiros por zona"
        )

        st.plotly_chart(fig_z, use_container_width=True)

        path_z_j = f"temp_zonas_jugadora_{safe_text_filename(jugadora_sel)}_{safe_text_filename(jornada_sel)}.png"
        fig_z.write_image(path_z_j)
        pdf_images.append(path_z_j)

        zona_jug_summary = "\n".join(
            [
                f"{row['zona_tiro']}: tiros={row['tiros']} | encestados={row['encestados']} | FG%={row['FG%']:.2f}"
                for _, row in zona_stats_jug.iterrows()
            ]
        ) if not zona_stats_jug.empty else "No disponible."
    except Exception:
        zona_jug_summary = "No se pudo exportar el grafico de volumen por zonas de la jugadora."

    # --------------------------
    # Rendimiento por zonas — equipo
    # --------------------------
    st.subheader("Rendimiento por zonas — equipo")

    zona_stats_team = (
        tiros_jornada.groupby("zona_tiro", as_index=False)
        .agg(
            encestados=("made", "sum"),
            tiros=("zona_tiro", "size")
        )
    ) if not tiros_jornada.empty else pd.DataFrame(columns=["zona_tiro", "encestados", "tiros"])

    if not zona_stats_team.empty:
        zona_stats_team["FG%"] = (
            100 * zona_stats_team["encestados"] / zona_stats_team["tiros"]
        ).round(2)

        zona_stats_team["zona_tiro"] = pd.Categorical(
            zona_stats_team["zona_tiro"],
            categories=ZONAS_ORDEN,
            ordered=True
        )

        zona_stats_team = zona_stats_team.sort_values("zona_tiro")

    st_dataframe_2d(zona_stats_team, use_container_width=True, hide_index=True)

    st.subheader("Volumen por zonas — equipo")

    zona_team_summary = "No disponible."
    try:
        fig_z_team = px.bar(
            zona_stats_team,
            x="zona_tiro",
            y="tiros",
            hover_data=["encestados", "FG%"],
            title="Distribución de tiros por zona"
        )

        st.plotly_chart(fig_z_team, use_container_width=True)

        path_z_t = f"temp_zonas_equipo_{safe_text_filename(team_sel)}_{safe_text_filename(jornada_sel)}.png"
        fig_z_team.write_image(path_z_t)
        pdf_images.append(path_z_t)

        zona_team_summary = "\n".join(
            [
                f"{row['zona_tiro']}: tiros={row['tiros']} | encestados={row['encestados']} | FG%={row['FG%']:.2f}"
                for _, row in zona_stats_team.iterrows()
            ]
        ) if not zona_stats_team.empty else "No disponible."
    except Exception:
        zona_team_summary = "No se pudo exportar el grafico de volumen por zonas del equipo."

    # --------------------------
    # EXPORTACIÓN PDF
    # --------------------------
    st.divider()
    st.subheader("📄 Exportar informe completo")

    summary_kpis = (
        f"Jornada: {jornada_sel}\n"
        f"Equipo: {team_sel}\n"
        f"Jugadora: {jugadora_sel}\n"
        f"Tiros totales: {fmt_num(n_tiros)}\n"
        f"Encertats: {fmt_num(encertats)}\n"
        f"FG%: {fmt_num(pct, suffix='%')}\n"
        f"2PT: {fmt_num(made2)}/{fmt_num(int2)}\n"
        f"3PT: {fmt_num(made3)}/{fmt_num(int3)}"
    )

    pdf_data = build_pdf_report(
        title=f"Informe de Mapas de Tiro - {jugadora_sel}",
        subtitle="Analisis por jornada de jugadora y equipo",
        sections=[
            {
                "heading": "Resumen",
                "body": summary_kpis,
            },
            {
                "heading": "Rendimiento por zonas - jugadora",
                "body": zona_jug_summary,
            },
            {
                "heading": "Rendimiento por zonas - equipo",
                "body": zona_team_summary,
            },
        ],
        image_paths=pdf_images,
    )

    st.download_button(
        label="📥 Descargar informe PDF completo",
        data=pdf_data,
        file_name=f"mapa_tiro_{safe_text_filename(jugadora_sel)}_{safe_text_filename(jornada_sel)}.pdf",
        mime="application/pdf",
    )

    for img_path in pdf_images:
        if os.path.exists(img_path):
            os.remove(img_path)

    if fig_j is not None:
        plt.close(fig_j)
    if fig_t is not None:
        plt.close(fig_t)

# ==============================
# MAIN
# ==============================

def main():
    check_login()

    st.sidebar.title(f"Bienvenido/a, {st.session_state.get('username','')}")

    page = st.sidebar.radio(
        "Navegación",
        ["Home", "Analítica de equipos", "Comparativa de equipos","Comparativa de equipos L3", "Previa del partido","Jugadoras","Mapas de tiro por jornada"]
    )

    if st.sidebar.button("Cerrar sesión"):
        st.session_state["logged_in"] = False
        st.rerun()

    if page == "Home":
        show_home()
    elif page == "Analítica de equipos":
        page_analitica_equips()
    elif page == "Comparativa de equipos":
        page_comparativa_equips()
    elif page == "Comparativa de equipos L3":
        page_comparativa_equips_last3()
    elif page == "Previa del partido":
        page_previa_partit()
    elif page == "Jugadoras":
        page_jugadoras()
    elif page == "Mapas de tiro por jornada":
        page_mapes_tir()


if __name__ == "__main__":
    main()
