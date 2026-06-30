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

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")

DB_PATH = Path(__file__).resolve().parent / "medical_ai_evidence.sqlite"

# Initialize database on startup
init_db(DB_PATH)

# Sidebar Navigation
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Aller à", ["Analyser une radio", "Tableau de bord & Historique"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Projet EFREI 2025-2026**\n\n"
    "Prototype pédagogique d'IA médicale multimodale."
)

if page == "Analyser une radio":
    st.title("Assistant radiologue virtuel — Analyse")
    st.warning("⚠️ Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.")
    
    uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])
    mode = st.selectbox("Mode", ["baseline", "improved"])
    
    if uploaded:
        # Save file preserving the original name so that the toy simulator can detect it
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getvalue())
    
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(Image.open(tmp_path), caption=f"Image uploadée : {uploaded.name}", use_container_width=True)
        with col2:
            st.subheader("Résultats de l'analyse")
            with st.spinner("Analyse de l'image en cours..."):
                pred = apply_safety_guardrails(toy_predict(tmp_path, mode=mode))
                
                # Log the prediction run to SQLite database
                case_id = f"web_{Path(uploaded.name).stem}_{int(time.time())}"
                insert_run(DB_PATH, case_id, str(tmp_path), pred)
                
            st.metric("Classe prédite", pred["predicted_class"].upper())
            st.metric("Confiance", f"{pred['confidence'] * 100:.1f} %")
            
            st.write("**Observations visuelles**", pred["visual_evidence"])
            st.write("**Justification**", pred["justification"])
            st.write("**Limites du modèle**", pred["limitations"])
            
            with st.expander("Voir la réponse JSON brute"):
                st.json(pred)
    else:
        st.info("💡 Utilisez une des images synthétiques dans `data/sample_images` pour tester l'application.")

else:
    st.title("Tableau de bord & Historique des analyses")
    st.warning("⚠️ Données issues d'un prototype pédagogique à but éducatif uniquement.")
    
    def load_metrics_data():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM runs", conn)
        conn.close()
        return df

    try:
        df = load_metrics_data()
    except Exception:
        df = pd.DataFrame()
        
    if df.empty:
        st.info("Aucune analyse n'a été enregistrée pour le moment. Allez sur l'onglet 'Analyser une radio' pour débuter.")
    else:
        # Top-level indicators
        total_runs = len(df)
        avg_latency = df["latency_ms"].mean()
        avg_confidence = df["confidence"].mean()
        
        st.subheader("Indicateurs clés de performance (KPI)")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Analyses totales", f"{total_runs}")
        col_m2.metric("Latence moyenne", f"{avg_latency:.1f} ms")
        col_m3.metric("Confiance moyenne", f"{avg_confidence * 100:.1f} %")
        
        st.write("---")
        
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            st.subheader("Répartition des diagnostics")
            class_counts = df["predicted_class"].value_counts().reset_index()
            class_counts.columns = ["Classe", "Nombre"]
            st.bar_chart(class_counts.set_index("Classe"), y="Nombre")
            
        with col_chart2:
            st.subheader("Historique des latences d'inférence")
            st.line_chart(df["latency_ms"])
            
        st.write("---")
        st.subheader("Historique détaillé des 10 dernières analyses")
        df_display = df.sort_values(by="id", ascending=False).head(10)[["case_id", "predicted_class", "confidence", "latency_ms", "created_at"]]
        st.dataframe(df_display, use_container_width=True)

