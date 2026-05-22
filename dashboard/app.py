"""Streamlit electoral misinformation detection dashboard for INEC and fact-checkers."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.models.multilingual_classifier import NigerianElectionMisinfo, SAMPLE_CLAIMS_BY_LANGUAGE

st.set_page_config(page_title="Electoral Misinfo Detection", page_icon="🗳️", layout="wide")
st.title("🗳️ Electoral Misinformation Detection — Nigeria")
st.caption("Multi-modal AI | Audio deepfakes · Document forgery · Multilingual fake news (EN/Pidgin/Hausa/Yoruba/Igbo)")

VERDICT_COLOURS = {"REAL": "#00CC00", "MISLEADING": "#FFA500", "FABRICATED": "#CC0000"}
classifier = NigerianElectionMisinfo()


@st.cache_data
def generate_demo_submissions(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    labels = np.random.choice(["REAL", "MISLEADING", "FABRICATED"], n, p=[0.45, 0.30, 0.25])
    modalities = np.random.choice(["Text", "Audio", "Document"], n, p=[0.60, 0.20, 0.20])
    languages = np.random.choice(["English", "Pidgin", "Hausa", "Yoruba", "Igbo"], n,
                                   p=[0.40, 0.25, 0.18, 0.10, 0.07])
    platforms = np.random.choice(["WhatsApp", "Twitter/X", "Facebook", "Telegram", "Email"], n,
                                   p=[0.55, 0.20, 0.15, 0.07, 0.03])
    return pd.DataFrame({
        "submission_id": [f"SUB-{i:05d}" for i in range(n)],
        "date": pd.date_range("2023-02-01", periods=n, freq="3h"),
        "modality": modalities,
        "language": languages,
        "platform": platforms,
        "verdict": labels,
        "confidence": np.clip(np.random.beta(7, 2, n), 0.60, 0.99).round(3),
        "reported_to_inec": labels == "FABRICATED",
    })


submissions = generate_demo_submissions()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Submissions", len(submissions))
with col2:
    fabricated = len(submissions[submissions["verdict"] == "FABRICATED"])
    st.metric("Fabricated Content", fabricated, delta_color="inverse")
with col3:
    misleading = len(submissions[submissions["verdict"] == "MISLEADING"])
    st.metric("Misleading Content", misleading, delta_color="inverse")
with col4:
    real = len(submissions[submissions["verdict"] == "REAL"])
    st.metric("Verified Real", real)

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Monitoring Dashboard", "🔍 Live Classifier", "📋 Flagged Content"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        verdict_counts = submissions["verdict"].value_counts()
        fig = px.pie(
            values=verdict_counts.values, names=verdict_counts.index,
            color=verdict_counts.index, color_discrete_map=VERDICT_COLOURS,
            title="Submission Verdicts",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        daily = submissions.set_index("date").resample("D")["submission_id"].count().reset_index()
        daily.columns = ["date", "submissions"]
        fig2 = px.bar(daily, x="date", y="submissions", title="Daily Submission Volume",
                       color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        lang_verdict = submissions.groupby(["language", "verdict"]).size().reset_index(name="count")
        fig3 = px.bar(
            lang_verdict, x="language", y="count", color="verdict",
            color_discrete_map=VERDICT_COLOURS,
            title="Verdicts by Language",
            barmode="stack",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        platform_fab = (
            submissions[submissions["verdict"] == "FABRICATED"]
            .groupby("platform")["submission_id"].count()
            .reset_index().rename(columns={"submission_id": "fabricated_count"})
            .sort_values("fabricated_count", ascending=False)
        )
        fig4 = px.bar(
            platform_fab, x="platform", y="fabricated_count",
            title="Fabricated Content by Platform",
            color_discrete_sequence=["#CC0000"],
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.subheader("Classify Content in Real Time")
    col_inp, col_samples = st.columns([2, 1])

    with col_samples:
        st.write("**Sample Claims (click to load):**")
        lang_sample = st.selectbox("Language", ["en", "pcm", "ha", "yo", "ig"])
        for sample in SAMPLE_CLAIMS_BY_LANGUAGE.get(lang_sample, []):
            if st.button(sample[:60] + "...", key=sample):
                st.session_state.text_input = sample

    with col_inp:
        text_input = st.text_area(
            "Enter text to classify",
            value=st.session_state.get("text_input", ""),
            height=120,
            placeholder="Paste a WhatsApp message, tweet, or social media post here...",
        )
        modality = st.radio("Content Type", ["Text", "Audio (demo)", "Document (demo)"], horizontal=True)

        if st.button("🔍 Detect Misinformation"):
            with st.spinner("Analysing..."):
                result = classifier.classify(text_input)

            colour = VERDICT_COLOURS[result.label]
            icon = "✅" if result.label == "REAL" else "⚠️" if result.label == "MISLEADING" else "❌"

            st.markdown(f"""
            <div style='background:{colour};color:white;padding:16px;border-radius:8px;margin-top:12px'>
            <h3>{icon} VERDICT: {result.label}</h3>
            <b>Confidence:</b> {result.confidence:.1%} &nbsp;|&nbsp; <b>Language:</b> {result.detected_language.upper()}<br>
            <br><b>Action:</b> {result.action}
            </div>
            """, unsafe_allow_html=True)

            if result.claim_flags:
                st.warning("**Claim Flags:**\n" + "\n".join(f"• {f}" for f in result.claim_flags))

            prob_df = pd.DataFrame(
                {"Verdict": list(result.probabilities.keys()),
                 "Probability": list(result.probabilities.values())}
            )
            fig_prob = px.bar(
                prob_df, x="Verdict", y="Probability",
                color="Verdict", color_discrete_map=VERDICT_COLOURS,
                range_y=[0, 1], title="Class Probabilities",
            )
            st.plotly_chart(fig_prob, use_container_width=True)

with tab3:
    st.subheader("Flagged Content — Reported to INEC")
    flagged = submissions[submissions["verdict"] == "FABRICATED"].sort_values("date", ascending=False)
    st.error(f"{len(flagged)} fabricated submissions flagged for INEC and platform moderators")
    st.dataframe(
        flagged[["submission_id", "date", "modality", "language", "platform", "confidence"]].head(50),
        use_container_width=True,
    )

st.markdown("---")
st.caption("MOMAH MOSES .C. · Geospatial AI Engineer & Data Scientist · github.com/Momahmoses")
