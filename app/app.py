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
import missingno as msno
from scipy.stats import zscore
import tempfile

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

# === Plot Generator (no auto-saving) ===
def generate_eda_plots(df):
    plot_files = []
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        safe_col = sanitize_filename(col)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Histogram: {col}')
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmpfile.name)
        plot_files.append(tmpfile.name)
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot: {col}')
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmpfile.name)
        plot_files.append(tmpfile.name)
        plt.close(fig)

    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmpfile.name)
        plot_files.append(tmpfile.name)
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

        # === Dynamic Filtering ===
        st.subheader("🔍 Dynamic Data Filtering")
        with st.expander("Apply Filters"):
            filters = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    options = df[col].dropna().unique().tolist()
                    selected = st.multiselect(f"Filter {col}", options)
                    if selected:
                        filters[col] = selected
                else:
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    selected_range = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
                    filters[col] = selected_range

            for key, val in filters.items():
                if isinstance(val, list):
                    df = df[df[key].isin(val)]
                else:
                    df = df[(df[key] >= val[0]) & (df[key] <= val[1])]

        # === Basic Info ===
        st.subheader("\U0001F50D Basic Information")
        st.write(df.head())
        st.write(f"**Shape:** {df.shape}")
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Memory Usage:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        # === Data Quality ===
        st.subheader("\u2705 Data Quality Checks")
        st.write("**Missing Values:**")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
        duplicates = df.duplicated().sum()
        st.write(f"**Duplicate Rows:** {duplicates}")

        # === Stats ===
        st.subheader("\U0001F4C8 Descriptive Statistics")
        stats = df.describe()
        st.write(stats)

        # === Aggregation Summary ===
        st.subheader("🧮 Aggregation Summary")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        group_cols = st.multiselect("Group by", df.columns)
        if group_cols:
            agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "median", "min", "max"])
            agg_df = df.groupby(group_cols)[num_cols].agg(agg_func)
            st.dataframe(agg_df)

        # === Visual Explorer ===
        st.subheader("📊 Visual Explorer")
        x_axis = st.selectbox("X-axis", df.columns)
        y_axis = st.selectbox("Y-axis", df.columns)
        chart_type = st.radio("Chart Type", ["Scatter", "Bar", "Line"])

        if chart_type == "Scatter":
            chart = alt.Chart(df).mark_circle(size=60).encode(x=x_axis, y=y_axis, tooltip=df.columns.tolist()).interactive()
        elif chart_type == "Bar":
            chart = alt.Chart(df).mark_bar().encode(x=x_axis, y=y_axis)
        else:
            chart = alt.Chart(df).mark_line().encode(x=x_axis, y=y_axis)
        st.altair_chart(chart, use_container_width=True)

        # === Plots ===
        st.subheader("\U0001F4C9 Distributions & Outliers")
        plot_files = generate_eda_plots(df)
        for plot in plot_files:
            st.image(plot)

        # === Missing Matrix ===
        st.subheader("🧩 Missing Value Matrix")
        fig = msno.matrix(df)
        st.pyplot(fig.figure)

        # === Trend & Forecast ===
        st.subheader("📈 Trend & Forecast (Beta)")
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0 and len(num_cols) > 0:
            date_col = st.selectbox("Select Date Column", text_cols)
            target_col = st.selectbox("Target Column", num_cols)
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                trend_df = df[[date_col, target_col]].dropna().sort_values(date_col)
                trend_df = trend_df.groupby(date_col).mean().reset_index()

                chart = alt.Chart(trend_df).mark_line().encode(
                    x=date_col,
                    y=target_col,
                    tooltip=[date_col, target_col]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Could not generate trend chart: {e}")

        # === Outlier Detection ===
        st.subheader("🚨 Outlier Detection (Z-Score)")
        z_threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0)
        z_scores = df[num_cols].apply(zscore)
        outliers = (np.abs(z_scores) > z_threshold).any(axis=1)
        st.write(f"Found {outliers.sum()} outliers")
        st.dataframe(df[outliers])

        # === PDF Download ===
        st.subheader("\U0001F4CB Download PDF Report")
        pdf_bytes = generate_pdf(df, stats, missing, duplicates, plot_files)
        st.download_button("\U0001F4C4 Download EDA Report (PDF)", pdf_bytes, "eda_report.pdf", "application/pdf")

        for file in plot_files:
            os.remove(file)

        # === YData Profiling ===
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