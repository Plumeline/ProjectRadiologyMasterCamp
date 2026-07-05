from __future__ import annotations

import tempfile
import time
import sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails
from src.database import insert_run, init_db

st.set_page_config(
    page_title="HEALTH & IA — EFREI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path(__file__).resolve().parent / "medical_ai_evidence.sqlite"
init_db(DB_PATH)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp {
    background: linear-gradient(160deg, #070d1a 0%, #0c1525 60%, #091220 100%);
    color: #cbd5e1;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1525 0%, #0d1f38 100%) !important;
    border-right: 1px solid rgba(0,212,170,0.15) !important;
}
[data-testid="stSidebarContent"] { padding-top: 1.5rem; }

/* Sidebar radio buttons */
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    transition: color 0.2s;
}
[data-testid="stSidebar"] label:hover { color: #00d4aa !important; }

/* Headings */
h1 { color: #f1f5f9 !important; font-weight: 800 !important; letter-spacing: -1px !important; }
h2 { color: #e2e8f0 !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
h3 { color: #cbd5e1 !important; font-weight: 600 !important; }

/* Warning & alert banners */
[data-testid="stAlert"] {
    background: rgba(234,179,8,0.08) !important;
    border: 1px solid rgba(234,179,8,0.25) !important;
    border-radius: 12px !important;
    color: #fbbf24 !important;
}

/* Info boxes */
[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(0,212,170,0.07) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    border-radius: 12px !important;
}

/* Metrics cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 22px !important;
    transition: border-color 0.3s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,212,170,0.3);
}
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,212,170,0.04) !important;
    border: 2px dashed rgba(0,212,170,0.25) !important;
    border-radius: 16px !important;
    transition: all 0.3s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0,212,170,0.55) !important;
    background: rgba(0,212,170,0.07) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Expander */
details {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 4px 8px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] iframe { border-radius: 12px !important; }

/* Download button */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #00d4aa 0%, #0891b2 100%) !important;
    color: #070d1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stDownloadButton"] button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

/* Caption */
p.caption, small { color: #475569 !important; }

/* Spinner */
[data-testid="stSpinner"] { color: #00d4aa !important; }

/* Image */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Markdown bullet points */
ul { padding-left: 1.2rem; }
ul li { color: #94a3b8; margin-bottom: 0.3rem; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0c1525; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = Path(__file__).resolve().parent / "data" / "logo.jpeg"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown("<div style='text-align:center; font-size:2.5rem;'>🩻</div>", unsafe_allow_html=True)
        
    st.markdown("""
    <div style="text-align:center; padding-bottom: 1rem;">
        <div style="font-size:1.1rem; font-weight:800; color:#f1f5f9; letter-spacing:-0.5px; margin-top:8px;">HEALTH <span style='color:#00d4aa'>&</span> IA</div>
        <div style="font-size:0.65rem; color:#00d4aa; letter-spacing:2px; text-transform:uppercase; margin-top:2px;">EFREI · 2025–2026</div>
    </div>
    <hr style="border-color:rgba(0,212,170,0.2); margin: 0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🔬 Analyser une radio", "📊 Tableau de bord", "📖 Guide d'utilisation"], label_visibility="collapsed")

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.06); margin: 1.5rem 0 1rem 0;">
    <div style="font-size:0.7rem; color:#334155; text-align:center; line-height:1.6;">
        Prototype pédagogique uniquement.<br>
        Non destiné au diagnostic médical.<br><br>
        <span style="color:#1e3a5f;">© EFREI Paris · Filière Data</span>
    </div>
    """, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
CLASS_CONFIG = {
    "normal": {
        "color": "#22c55e", "bg": "rgba(34,197,94,0.10)",
        "border": "rgba(34,197,94,0.30)", "icon": "✅", "label": "NORMAL"
    },
    "suspected_opacity": {
        "color": "#f59e0b", "bg": "rgba(245,158,11,0.10)",
        "border": "rgba(245,158,11,0.30)", "icon": "⚠️", "label": "SUSPECTED OPACITY"
    },
    "uncertain": {
        "color": "#94a3b8", "bg": "rgba(148,163,184,0.10)",
        "border": "rgba(148,163,184,0.25)", "icon": "❓", "label": "UNCERTAIN"
    },
}

def load_db() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM runs", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# ── Page : Analyse ─────────────────────────────────────────────────────────
if page == "🔬 Analyser une radio":

    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <h1 style="margin:0; font-size:2rem;">🔬 Analyser une radiographie</h1>
        <p style="color:#475569; margin-top:4px; font-size:0.9rem;">Déposez une image pour obtenir une analyse structurée et tracée.</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️  **Prototype pédagogique.** Non destiné au diagnostic. Validation par un professionnel qualifié requise.")

    col_upload, col_mode = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader("Radiographie thoracique frontale", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    with col_mode:
        mode = st.radio("Mode d'analyse", ["baseline", "improved"], horizontal=True)

    if uploaded:
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getvalue())

        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(Image.open(tmp_path), caption=f"📁 {uploaded.name}", width=320)

        with col_res:
            with st.spinner("Analyse en cours..."):
                pred = apply_safety_guardrails(toy_predict(tmp_path, mode=mode))
                case_id = f"web_{Path(uploaded.name).stem}_{int(time.time())}"
                insert_run(DB_PATH, case_id, str(tmp_path), pred, dataset_source="web_upload")

            cfg = CLASS_CONFIG.get(pred["predicted_class"], CLASS_CONFIG["uncertain"])

            # Résultat principal (carte colorée)
            st.markdown(f"""
            <div style="background:{cfg['bg']}; border:1.5px solid {cfg['border']};
                        border-radius:16px; padding:20px 24px; margin-bottom:16px;">
                <div style="font-size:0.65rem; color:#475569; text-transform:uppercase;
                            letter-spacing:2px; font-weight:600; margin-bottom:6px;">Classe prédite</div>
                <div style="font-size:2.2rem; font-weight:800; color:{cfg['color']};
                            letter-spacing:-0.5px; line-height:1.1;">{cfg['icon']} {cfg['label']}</div>
            </div>
            """, unsafe_allow_html=True)

            # KPI metrics
            m1, m2 = st.columns(2)
            m1.metric("Confiance", f"{pred['confidence'] * 100:.1f} %")
            m2.metric("Qualité image", pred.get("image_quality", "N/A").upper())

            st.caption(f"🔧 Mode : **{mode}**  ·  Modèle : **{pred.get('model_name', 'N/A')}**  ·  Prompt : **{pred.get('prompt_version', 'N/A')}**")

        # Détails en pleine largeur
        st.divider()
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.markdown("**Observations visuelles**")
            for obs in pred.get("visual_evidence", []):
                st.markdown(f"- {obs}")
            st.markdown("**Justification**")
            st.info(pred.get("justification", "N/A"))
        with col_d2:
            st.markdown("**Limites du modèle**")
            for lim in pred.get("limitations", []):
                st.markdown(f"- {lim}")
            with st.expander("Voir la réponse JSON brute"):
                st.json(pred)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02); border:2px dashed rgba(255,255,255,0.07);
                    border-radius:20px; padding:3rem; text-align:center; margin-top:2rem;">
            <div style="font-size:3rem; margin-bottom:12px;">🩻</div>
            <div style="color:#475569; font-size:1rem; font-weight:500;">Déposez une image ci-dessus pour commencer l'analyse</div>
            <div style="color:#334155; font-size:0.8rem; margin-top:8px;">
                Images disponibles dans <code style="color:#00d4aa;">data/sample_images/</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Page : Guide d'utilisation ────────────────────────────────────────────
elif page == "📖 Guide d'utilisation":

    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <h1 style="margin:0; font-size:2rem;">📖 Guide d'utilisation</h1>
        <p style="color:#475569; margin-top:4px; font-size:0.9rem;">Tout ce qu'il faut savoir pour utiliser HEALTH & IA.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Présentation ──
    st.markdown("""
    <div style="background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.2);
                border-radius:16px; padding:24px 28px; margin-bottom:1.5rem;">
        <div style="font-size:1.1rem; font-weight:700; color:#00d4aa; margin-bottom:8px;">🩺 Qu'est-ce que HEALTH & IA ?</div>
        <div style="color:#94a3b8; line-height:1.8; font-size:0.95rem;">
            <b style="color:#e2e8f0;">HEALTH & IA</b> est un prototype pédagogique d'intelligence artificielle médicale développé dans le cadre
            du MasterCamp EFREI 2025–2026. Il analyse des radiographies thoraciques frontales et retourne
            un résultat structuré en trois classes possibles, accompagné d'une justification et de métadonnées traçables.
            <br><br>
            ⚠️ <b style="color:#fbbf24;">Ce prototype n'est pas un outil de diagnostic médical.</b>
            Toute interprétation doit être validée par un professionnel de santé qualifié.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Étapes ──
    st.markdown("### 🚀 Comment utiliser l'application ?")

    steps = [
        ("1", "#00d4aa", "Choisir votre image",
         "Rendez-vous sur l'onglet <b>🔬 Analyser une radio</b>. Déposez une radiographie thoracique frontale au format PNG ou JPG dans la zone de dépôt. Des images de test sont disponibles dans le dossier <code style='color:#00d4aa'>data/sample_images/</code>."),
        ("2", "#0891b2", "Sélectionner le mode d'analyse",
         "Choisissez entre deux modes :<br>• <b style='color:#e2e8f0;'>Baseline</b> — prompt standard, résultat de référence.<br>• <b style='color:#e2e8f0;'>Improved</b> — prompt renforcé avec règle d'incertitude, résultat plus prudent."),
        ("3", "#6366f1", "Lire les résultats",
         "L'IA retourne une <b>classe prédite</b> parmi trois possibilités, un <b>score de confiance</b> et la <b>qualité détectée de l'image</b>. Les observations visuelles, la justification et les limites du modèle sont également affichées."),
        ("4", "#f59e0b", "Consulter le tableau de bord",
         "L'onglet <b>📊 Tableau de bord</b> centralise toutes les analyses effectuées : KPIs, graphiques de répartition, évolution de la confiance et historique détaillé. Vous pouvez exporter l'historique complet en CSV."),
    ]

    for num, color, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:16px; align-items:flex-start; margin-bottom:16px;
                    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                    border-radius:14px; padding:20px 22px;">
            <div style="min-width:36px; height:36px; border-radius:50%; background:{color}22;
                        border:2px solid {color}; display:flex; align-items:center;
                        justify-content:center; font-weight:800; color:{color}; font-size:1rem;">{num}</div>
            <div>
                <div style="font-weight:700; color:#e2e8f0; font-size:1rem; margin-bottom:6px;">{title}</div>
                <div style="color:#94a3b8; font-size:0.88rem; line-height:1.7;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Classes ──
    st.markdown("### 🏷️ Les 3 classes possibles")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("""
        <div style="background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.3);
                    border-radius:14px; padding:20px; text-align:center;">
            <div style="font-size:2rem;">✅</div>
            <div style="font-weight:800; color:#22c55e; font-size:1.1rem; margin:8px 0 6px 0;">NORMAL</div>
            <div style="color:#64748b; font-size:0.82rem; line-height:1.6;">Aucune opacité suspecte détectée. Radiographie dans les limites attendues.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.3);
                    border-radius:14px; padding:20px; text-align:center;">
            <div style="font-size:2rem;">⚠️</div>
            <div style="font-weight:800; color:#f59e0b; font-size:1.1rem; margin:8px 0 6px 0;">SUSPECTED OPACITY</div>
            <div style="color:#64748b; font-size:0.82rem; line-height:1.6;">Région suspecte détectée. Une consultation médicale est fortement recommandée.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c3:
        st.markdown("""
        <div style="background:rgba(148,163,184,0.08); border:1px solid rgba(148,163,184,0.25);
                    border-radius:14px; padding:20px; text-align:center;">
            <div style="font-size:2rem;">❓</div>
            <div style="font-weight:800; color:#94a3b8; font-size:1.1rem; margin:8px 0 6px 0;">UNCERTAIN</div>
            <div style="color:#64748b; font-size:0.82rem; line-height:1.6;">Le modèle ne peut pas trancher. Image de qualité insuffisante ou cas ambigu.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Limites ──
    st.divider()
    st.markdown("""
    <div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.2);
                border-radius:14px; padding:20px 24px;">
        <div style="font-weight:700; color:#f87171; margin-bottom:10px;">⛔ Limites importantes à connaître</div>
        <ul style="color:#94a3b8; font-size:0.88rem; line-height:2; margin:0; padding-left:1.2rem;">
            <li>Ce prototype utilise un <b style='color:#e2e8f0;'>modèle jouet</b> — les résultats sont simulés à des fins pédagogiques.</li>
            <li>Il n'a <b style='color:#e2e8f0;'>aucune valeur diagnostique</b> réelle et ne remplace pas un radiologue.</li>
            <li>Les images doivent être des <b style='color:#e2e8f0;'>radiographies thoraciques frontales</b> uniquement.</li>
            <li>Les résultats sont <b style='color:#e2e8f0;'>tracés en base SQLite</b> localement à des fins d'audit et de recherche.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Page : Dashboard ───────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <h1 style="margin:0; font-size:2rem;">📊 Tableau de bord</h1>
        <p style="color:#475569; margin-top:4px; font-size:0.9rem;">Suivi en temps réel des analyses effectuées.</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️  Données issues d'un prototype pédagogique à but éducatif uniquement.")

    df = load_db()

    if df.empty:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02); border:2px dashed rgba(255,255,255,0.07);
                    border-radius:20px; padding:3rem; text-align:center; margin-top:2rem;">
            <div style="font-size:3rem; margin-bottom:12px;">📭</div>
            <div style="color:#475569;">Aucune analyse enregistrée. Commencez par analyser une image.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(df)
        avg_lat = df["latency_ms"].mean()
        avg_conf = df["confidence"].mean()
        uncertain_rate = (df["predicted_class"] == "uncertain").mean() * 100

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Analyses totales", f"{total}")
        k2.metric("Latence moyenne", f"{avg_lat:.1f} ms")
        k3.metric("Confiance moyenne", f"{avg_conf * 100:.1f} %")
        k4.metric("Taux d'incertitude", f"{uncertain_rate:.1f} %")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Répartition des diagnostics**")
            class_counts = df["predicted_class"].value_counts().reset_index()
            class_counts.columns = ["Classe", "Nombre"]
            st.bar_chart(class_counts.set_index("Classe"), y="Nombre", color="#00d4aa")

        with c2:
            st.markdown("**Évolution de la confiance**")
            st.line_chart(df["confidence"], color="#00d4aa")

        st.divider()
        st.markdown("**Historique détaillé des 10 dernières analyses**")
        df_display = df.sort_values(by="id", ascending=False).head(10)[
            ["case_id", "predicted_class", "confidence", "latency_ms",
             "model_name", "prompt_version", "dataset_source", "created_at"]
        ]
        st.dataframe(df_display, use_container_width=True)

        st.download_button(
            label="⬇️ Exporter tout l'historique en CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="historique_analyses.csv",
            mime="text/csv",
        )
