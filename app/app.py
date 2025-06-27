import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from ydata_profiling import ProfileReport
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import os
import base64
import re
import streamlit.components.v1 as components

# === Background Image ===
def set_bg_image(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{img_base64}");
                background-size: cover;
                background-position: center;
                color: white;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_bg_image("images/background_image.jpg")

# === Sanitize filenames ===
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|\s]', "_", str(name))

# === PDF Report Generator ===
def generate_pdf(df, stats, missing, duplicates, plot_files):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "\U0001F4CA AI-Powered Data Report")
    y -= 40

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    y -= 20
    c.drawString(50, y, f"Duplicate rows: {duplicates}")
    y -= 30

    c.drawString(50, y, "Summary Stats:")
    y -= 20
    for col in stats.columns[:4]:
        c.drawString(60, y, f"{col}: mean={stats[col]['mean']:.2f}, std={stats[col]['std']:.2f}")
        y -= 20

    if not missing.empty:
        c.drawString(50, y, "Missing Values:")
        y -= 20
        for idx, val in missing.items():
            c.drawString(60, y, f"{idx}: {val} missing")
            y -= 20

    for plot_file in plot_files:
        c.showPage()
        img = ImageReader(plot_file)
        c.drawImage(img, 50, 200, width=500, height=300)

    c.save()
    buffer.seek(0)
    return buffer

# === Plot Generator ===
def generate_eda_plots(df):
    plot_files = []
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        safe_col = sanitize_filename(col)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Histogram: {col}')
        fname = f"hist_{safe_col}.png"
        fig.savefig(fname)
        plot_files.append(fname)
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot: {col}')
        fname = f"box_{safe_col}.png"
        fig.savefig(fname)
        plot_files.append(fname)
        plt.close(fig)

    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        fname = "correlation_heatmap.png"
        fig.savefig(fname)
        plot_files.append(fname)
        plt.close(fig)

    return plot_files

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA Tool", "About"])

# === EDA Tool ===
if page == "EDA Tool":
    st.title("\U0001F4CA Advanced EDA + PDF + Profiling")
    st.write("Upload your CSV file and explore your data with auto EDA, profiling, and downloadable reports.")

    file = st.file_uploader("Upload CSV", type=['csv'])
    if file:
        df = pd.read_csv(file)

        st.subheader("\U0001F50D Basic Information")
        st.write(df.head())
        st.write(f"**Shape:** {df.shape}")
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Memory Usage:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("\u2705 Data Quality Checks")
        st.write("**Missing Values:**")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
        duplicates = df.duplicated().sum()
        st.write(f"**Duplicate Rows:** {duplicates}")

        st.subheader("\U0001F4C8 Descriptive Statistics")
        stats = df.describe()
        st.write(stats)

        st.subheader("\U0001F4C9 Distributions & Outliers")
        plot_files = generate_eda_plots(df)
        for plot in plot_files:
            st.image(plot)

        st.subheader("\U0001F4CB Download PDF Report")
        pdf_bytes = generate_pdf(df, stats, missing, duplicates, plot_files)
        st.download_button("\U0001F4C4 Download EDA Report (PDF)", pdf_bytes, "eda_report.pdf", "application/pdf")

        for file in plot_files:
            os.remove(file)

        st.subheader("\U0001F4D1 Automated Profiling Report")
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        profile.to_file("profiling_report.html")

        with open("profiling_report.html", "r", encoding="utf-8") as f:
            html_content = f.read()
            components.html(html_content, height=1000, scrolling=True)

        with open("profiling_report.html", "rb") as f:
            st.download_button("\u2B07\uFE0F Download HTML Profiling Report", f, "profiling_report.html", "text/html")

# === About Section ===
elif page == "About":
    st.title("\U0001F468\u200D\U0001F4BB About This App")

    st.image("images/rayhan.jpg", width=150)

    st.markdown("""
    **Creator:** Rayhan Mahmud Ansari  
    \U0001F393 Dept. of CSE, Sylhet Engineering College  
    \U0001F4E7 rayhan_mahmud@sec.ac.bd  
    [\U0001F310 GitHub](https://github.com/rayhanansari11) | [\U0001F517 LinkedIn](https://www.linkedin.com/in/rayhan-mahmud-ansari-566d/)  

    **What this app does:**
    - \U0001F4CA Upload and explore CSV files  
    - \U0001F4CB Automatically generate statistics, plots, and correlation  
    - \U0001F4D1 Full profiling via YData Profiling  
    - \U0001F4C4 Generate custom PDF report with visuals  

    **Tech used:** Streamlit, Pandas, Seaborn, Matplotlib, YData Profiling, ReportLab
    """)
