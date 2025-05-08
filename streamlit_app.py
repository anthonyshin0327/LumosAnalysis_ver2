# streamlit_app.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="LFA Analysis Pipeline", layout="wide")

st.title("🔬 LFA Development Data Analysis App")

st.markdown("""
Welcome to the all-in-one platform for:
- 📁 Uploading and structuring raw LFA data
- 📊 Performing 4PL fitting and IC50 extraction
- 🧪 Running RSM and ANOVA
- 📈 Viewing experiment dashboards
- 💾 Exporting reports in Excel, PDF, and Google Sheets
""")

st.info("Use the sidebar to navigate through the pages.")