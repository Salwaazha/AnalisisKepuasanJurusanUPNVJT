import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
from textwrap import dedent

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="🎓 Dashboard Analisis Kepuasan Mahasiswa Gen Z terhadap Jurusan yang Dipilih",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# THEME COLORS & FONTS (consistent purple)
# ---------------------------
PURPLE_MAIN = "#6a0dad"   # utama
PURPLE_ACCENT = "#9b59b6"
BG_LIGHT = "#f9f6ff"
CARD_BG = "#ffffff"
CARD_BG_DARK = "#1f1530"
TEXT_LIGHT = "#f7f3ff"
TEXT_DARK = "#260844"

# ---------------------------
# STYLES - import Poppins font & CSS
# ---------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(180deg, {BG_LIGHT}, #f3eefb);
    }}
    h1, h2, h3, h4 {{
        color: {TEXT_DARK};
        margin: 0;
        padding: 0;
    }}
    /* Header */
    .header-title {{
        text-align: center;
        margin-bottom: 8px;
    }}
    .sub-title {{
        text-align: center;
        color: {TEXT_DARK};
        margin-top: 4px;
        margin-bottom: 12px;
        font-weight:600;
    }}

    /* Card */
    .card {{
        background: {CARD_BG};
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 8px 24px rgba(106,13,173,0.12);
        border: 1px solid rgba(155,89,182,0.15);
    }}
    .card-dark {{
        background: {CARD_BG_DARK};
        color: {TEXT_LIGHT};
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 8px 34px rgba(0,0,0,0.35);
        border: 1px solid rgba(155,89,182,0.12);
    }}
    .metric {{
        text-align:center;
        padding: 10px;
    }}
    .metric h3 {{ margin-bottom:6px; color:{PURPLE_ACCENT}; font-weight:700; }}
    .metric h2 {{ margin-top:0; font-size:28px; color:{TEXT_DARK}; }}

    .insight {{
        background: linear-gradient(90deg, rgba(155,89,182,0.12), rgba(106,13,173,0.06));
        border-left: 6px solid {PURPLE_MAIN};
        padding: 12px;
        border-radius: 8px;
        margin-top:10px;
        color:{TEXT_DARK};
    }}

    /* Make chart titles consistent */
    .chart-title {{
        font-weight:700;
        color:{TEXT_DARK};
        font-size:18px;
        margin-bottom:6px;
        text-align : center;
        font-family : "Poppins";
    }}

    /* subtle hover scale for cards (visual feel) */
    .card:hover {{
        transform: translateY(-6px);
        transition: all 0.18s ease-out;
    }}

    /* small responsive */
    @media (max-width: 768px) {{
        .metric h2 {{ font-size:20px; }}
    }}

    .section-title {{
        font-family: 'Poppins';
        font-weight: 600;
        font-size: 18px;
        color: {TEXT_DARK};
        padding-bottom: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# HEADER
# ---------------------------
col1, col2, col3 = st.columns([1, 5, 2]) 

with col1: 
    st.image("LogoUPN.png", width=150) 
with col2: 
    st.markdown( """ <div style='text-align: center;'> <h1>🎓 Dashboard Analisis Kepuasan Mahasiswa Gen Z terhadap Jurusan Pilihan 🎓</h1> 
                <p style='font-size: 27px; font-weight: bold; margin: 0;'>Kelompok Escape - Sains Data UPN "Veteran Jawa Timur</p> 
                <p style='font-size: 22px; margin-top: 5px;'>Violin Chantika Ardianisya 24083010043 | Salwa Zahra Rahmawati 24083010083 | Siva Ifin Azzahra 24083010121</p> </div> """, 
                unsafe_allow_html=True ) 
with col3: 
    st.image("LogoSada.png", width=150)

st.markdown(
    """
    <hr style='border: none; height: 3px; background: linear-gradient(to right, #000, #555, #000); box-shadow: 0 2px 5px rgba(0,0,0,0.2); margin-top: 15px; margin-bottom: 30px;'>
    """,
    unsafe_allow_html=True
)

st.write("")  

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(path="AnalisisKepuasan_terakhir.csv"):
    df = pd.read_csv(path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Error: Tidak dapat menemukan file 'AnalisisKepuasan_terakhir.csv' di folder. Pastikan file berada di direktori yang sama dengan script ini.")
    st.stop()

data = df.copy()

# ---------------------------
# SIDEBAR 
# ---------------------------
st.sidebar.header("📚 Menu Utama")
page = st.sidebar.radio(
    "Pilih Halaman",
    ("📊 Overview Data", "📉 Statistika Deskriptif", "📈 Visualisasi & Hasil Analisis", "🔗 Hubungan Antar Variabel", "📈 Regresi Berganda", "🧩 Kesimpulan"),
)

# allow filtering by Program Studi (optional)
if "Program Studi" in data.columns:
    st.sidebar.markdown("---")
    selected = st.sidebar.multiselect("Filter Program Studi (opsional)", options=sorted(data["Program Studi"].unique()), default=list(sorted(data["Program Studi"].unique())))
    if selected:
        data = data[data["Program Studi"].isin(selected)]

PURPLE_SCALE = px.colors.sequential.PuRd  # built-in, purples
PRIMARY_HEX = PURPLE_MAIN

# ---------------------------
# Helper
# ---------------------------
def animated_bar_reveal(df_bar, x_col, y_col, title, color_scale=None, interval=300):
    df_bar = df_bar.reset_index(drop=True)
    frames = []
    for i in range(len(df_bar)):
        visible = [True if j <= i else False for j in range(len(df_bar))]
        frame = go.Frame(
            data=[
                go.Bar(
                    x=df_bar[x_col],
                    y=df_bar[y_col],
                    marker_color=[PRIMARY_HEX]*len(df_bar),
                    marker_line_color="rgba(0,0,0,0)",
                    text=df_bar[y_col].round(2),
                    textposition="outside",
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                    ),
            ],
            name=str(i)
        )
        frames.append(frame)

    fig = go.Figure(
        data=[go.Bar(x=df_bar[x_col], y=[0]*len(df_bar), marker_color=[PRIMARY_HEX]*len(df_bar),
                     text=[0]*len(df_bar), textposition="outside")],
        layout=go.Layout(
            title=title,
            xaxis={"tickangle":-45},
            yaxis={"title": y_col},
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "y":1.05,
                "x":1.15,
                "xanchor":"right",
                "yanchor":"top",
                "pad":{"t":0,"r":10},
                "buttons":[{
                    "label":"Play",
                    "method":"animate",
                    "args":[None, {"frame": {"duration": interval, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing":"cubic-in-out"}}]
                }]
            }]
        ),
        frames=frames
    )

    fig.update_layout(transition={"duration":350, "easing":"cubic-in-out"})
    return fig

# ---------------------------
# Page: Overview Data
# ---------------------------
if page == "📊 Overview Data":
    st.subheader("📊 Gambaran Umum Data Survei Kepuasan Mahasiswa Gen Z")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        <div class="card metric" style="
            background-color: #f5edff;
            padding: 10px 15px;
            border-radius: 10px;
            margin-top: 10px;
            margin-bottom: 30px;
            color: #3a0069;
            font-family: 'Poppins', sans-serif;
            font-size: 18px;
        ">
            <h4>🧩 Deskripsi Awal</h4>
            <p>Survei ini mengumpulkan data mengenai <b>tingkat kepuasan mahasiswa Gen Z terhadap jurusan yang dipilih</b>, 
            serta faktor-faktor yang mempengaruhi persepsi mereka terhadap pengalaman akademik dan lingkungan kampus.</p>
            <p>Berdasarkan hasil pengumpulan data, terdapat sebanyak <b>103 responden</b> yang berpartisipasi dalam survei ini. 
            Responden tersebut berasal dari berbagai fakultas, antara lain Fakultas Ilmu Komputer, Fakultas Teknik dan Sains, Fakultas Ekonomi dan Bisnis, 
                    Fakultas Hukum, Fakultas Ilmu Sosial dan Politik, Fakultas Pertanian, Fakultas Arsitektur dan Desain, Fakultas Kedokteran</p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="card metric" style="
            background-color: #f5edff;
            padding: 10px 15px;
            border-radius: 10px;
            margin-top: 10px;
            margin-bottom: 30px;
            color: #3a0069;
            font-family: 'Poppins', sans-serif;
            font-size: 18px;
        ">
            <h4>🎯 Tujuan Analisis</h4>
            <ul>
                1. Mengukur kepuasan mahasiswa UPNVJT terhadap jurusan yang dipilih.<br>
                2. Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap <b>kepuasan mahasiswa</b>.<br>
                3. Menganalisis <b>hubungan antar variabel</b>, seperti motivasi, tantangan, dan keinginan pindah jurusan.<br>
                4. Memberikan <b>insight rekomendatif</b> bagi pihak jurusan untuk meningkatkan pengalaman belajar mahasiswa.<br>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # === Preview data ===
    st.markdown("<h4 class='section-title'>🧾 Preview Data</h4>", unsafe_allow_html=True)
    st.dataframe(df)
    st.markdown("---")

    # === Key Metrics ===
    total_responden = len(data)
    total_prodi = data["Program Studi"].nunique() if "Program Studi" in data.columns else 0
    rata_kepuasan = round(data["Tingkat Kepuasan"].mean(), 2) if "Tingkat Kepuasan" in data.columns else np.nan
    rata_kesulitan = round(data["Tingkat Kesulitan Mata Kuliah"].mean(), 2) if "Tingkat Kesulitan Mata Kuliah" in data.columns else np.nan

    st.markdown("<h4 class='section-title'>📌 Ringkasan Umum</h4>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='card metric'><h3>Jumlah Responden</h3><h2>{total_responden}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card metric'><h3>Jumlah Program Studi</h3><h2>{total_prodi}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card metric'><h3>Rata-rata Kepuasan</h3><h2>{rata_kepuasan}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card metric'><h3>Rata-rata Kesulitan</h3><h2>{rata_kesulitan}</h2></div>", unsafe_allow_html=True)

    st.markdown("""
            <div style='
                background-color: #f5edff;
                border-left: 5px solid #6A0DAD;
                padding: 10px 15px;
                border-radius: 10px;
                margin-top: 10px;
                margin-bottom: 30px;
                color: #3a0069;
                font-family: "Poppins", sans-serif;
                font-size: 17px
            '>
        💡 Terdapat <b>103 responden</b> yang berasal dari <b>19 program studi</b>. Nilai <b>rata-rata kepuasan sebesar 7.83</b> menunjukkan bahwa sebagian besar mahasiswa merasa <b>cukup puas</b> terhadap pengalaman akademiknya. Sementara itu, <b>rata-rata tingkat kesulitan sebesar 5.4</b> mengindikasikan bahwa mahasiswa menghadapi <b>tantangan pada tingkat sedang</b> dalam proses pembelajaran.</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
# ---------------------------
# Page: Statistika Deskriptif
# ---------------------------
elif page == "📉 Statistika Deskriptif":
    st.subheader("📉 Statistika Deskriptif")

    hidden_cols = ["ID Responden", "NPM", "Nama Lengkap", "Relevensi Jurusan (Skor)"]
    data_filtered = data.drop(columns=[col for col in hidden_cols if col in data.columns])

    # === Statistik Variabel Numerik ===
    num_cols = data_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if num_cols:
        st.markdown("<div class='chart-title'>Statistik Variabel Numerik</div>", unsafe_allow_html=True)
        desc = data_filtered[num_cols].describe().T
        desc["range"] = desc["max"] - desc["min"]
        st.markdown(
                desc.style
                    .format(precision=2)
                    .set_properties(**{
                        'color': 'var(--text-color)',
                        'font-size': '16px',
                        'font-family': 'Poppins, sans-serif',
                        'background-color': 'var(--background-color)'
                    })
                    .set_table_styles([
                        {'selector': 'th', 'props': [('color', 'var(--text-color)')]},
                        {'selector': 'td', 'props': [('color', 'var(--text-color)')]},
                        {'selector': 'tbody th', 'props': [('color', 'var(--text-color)')]}
                    ])
                    .to_html(),
                unsafe_allow_html=True
            )
        
        # === Insight otomatis untuk variabel numerik ===
        mean_kepuasan = round(data_filtered["Tingkat Kepuasan"].mean(), 2) if "Tingkat Kepuasan" in data_filtered.columns else None
        max_motivasi = round(data_filtered["Tinggi Motivasi"].max(), 2) if "Tinggi Motivasi" in data_filtered.columns else None
        avg_kesulitan = round(data_filtered["Tingkat Kesulitan Mata Kuliah"].mean(), 2) if "Tingkat Kesulitan Mata Kuliah" in data_filtered.columns else None
        
        st.markdown(
            f"""
            <div style='
                background-color: #f5edff;
                border-left: 5px solid #6A0DAD;
                padding: 10px 15px;
                border-radius: 10px;
                margin-top: 10px;
                margin-bottom: 30px;
                color: #3a0069;
                font-family: "Poppins", sans-serif;
                font-size: 17px
            '>
        <div >
        💡 <b>Statistika Deskriptif Numerik:</b><br>
        • Rata-rata tingkat kepuasan mahasiswa adalah <b>{mean_kepuasan}</b>, menunjukkan bahwa mayoritas mahasiswa merasa cukup puas terhadap jurusannya.<br>
        • Rata-rata tingkat kesulitan mata kuliah sebesar <b>{avg_kesulitan}</b> menandakan tingkat tantangan akademik yang sedang.<br>
        • Nilai motivasi tertinggi mencapai <b>{max_motivasi}</b>, menandakan ada responden dengan semangat belajar tinggi.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Tidak ada kolom numerik yang tersedia.")
    st.markdown("</div>", unsafe_allow_html=True)

    # === Statistik Variabel Kategorik ===
    cat_cols = data_filtered.select_dtypes(include=["object", "category"]).columns.tolist()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if cat_cols:
        st.markdown("<div class='chart-title'>Ringkasan Variabel Kategorik</div>", unsafe_allow_html=True)
        cat_summary = pd.DataFrame({
            "Jumlah Kategori Unik": [data_filtered[c].nunique() for c in cat_cols],
            "Kategori Terbanyak": [data_filtered[c].mode()[0] if not data_filtered[c].mode().empty else "-" for c in cat_cols],
            "Frekuensi Tertinggi": [data_filtered[c].value_counts().max() for c in cat_cols],
            "Persentase Tertinggi (%)": [round(data_filtered[c].value_counts(normalize=True).max()*100, 2) for c in cat_cols]
        }, index=cat_cols)
        st.markdown(
            cat_summary.style
                .format(precision=2)
                .set_properties(**{
                        'color': 'var(--text-color)',
                        'font-size': '16px',
                        'font-family': 'Poppins, sans-serif',
                        'background-color': 'var(--background-color)'
                })
                .set_table_styles([
                    {'selector': 'th', 'props': [('color', 'var(--text-color)')]},
                    {'selector': 'td', 'props': [('color', 'var(--text-color)')]},
                    {'selector': 'tbody th', 'props': [('color', 'var(--text-color)')]}
                ])
                .to_html(),
            unsafe_allow_html=True
        )

        # === Insight otomatis kategorik ===
        st.markdown(
            f"""
            <div style='
                background-color: #f5edff;
                border-left: 5px solid #6A0DAD;
                padding: 10px 15px;
                border-radius: 10px;
                margin-top: 10px;
                margin-bottom: 30px;
                color: #3a0069;
                font-family: "Poppins", sans-serif;
                font-size: 17px
            '>
        <div>
        💡 <b>Statistika Deskriptif Kategorik:</b><br>
        • Mayoritas responden berasal dari <b>{cat_summary.loc['Fakultas','Kategori Terbanyak']}</b> dengan persentase <b>{cat_summary.loc['Fakultas','Persentase Tertinggi (%)']}%</b>.<br>
        • Program studi yang paling dominan adalah <b>{cat_summary.loc['Program Studi','Kategori Terbanyak']}</b>.<br>
        • Sebagian besar mahasiswa memilih jurusan karena <b>{cat_summary.loc['Alasan Memilih Jurusan','Kategori Terbanyak']}</b>.<br>
        • Ada sebagian kecil mahasiswa yang menyatakan <b>{cat_summary.loc['Keinginan Pindah Jurusan','Kategori Terbanyak']}</b>, menunjukkan adanya potensi ketidakpuasan kecil.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Tidak ada kolom kategorik yang tersedia.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Page: Visualisasi & Hasil Analisis
# ---------------------------
elif page == "📈 Visualisasi & Hasil Analisis":
    st.subheader("📈 Visualisasi & Hasil Analisis")
    # A. Rata-rata Kepuasan per Program Studi (animated reveal)
    if "Program Studi" in data.columns and "Tingkat Kepuasan" in data.columns:
        if "Program Studi" in data.columns and "Tingkat Kepuasan" in data.columns:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='chart-title'>Rata-Rata Kepuasan Berdasarkan Jurusan</div>", unsafe_allow_html=True)

            # Hitung rata-rata kepuasan per program studi
            avg = (
                data.groupby("Program Studi")["Tingkat Kepuasan"]
                .mean()
                .reset_index()
                .sort_values("Tingkat Kepuasan", ascending=False)
            )

            # Barchart warna ungu elegan
            fig = px.bar(
                avg,
                x="Program Studi",
                y="Tingkat Kepuasan",
                text="Tingkat Kepuasan",
                color="Tingkat Kepuasan",
                color_continuous_scale=["#D1C4E9", "#512DA8"],
                title=None
            )

            fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Tingkat Kepuasan: %{y:.2f}",
                marker_line_color="white",
                marker_line_width=1.5,
            )

            fig.update_layout(
                plot_bgcolor="#F3E5F5",
                paper_bgcolor="#F3E5F5",
                font=dict(family="Poppins", color="black", size=14),
                xaxis=dict(title="Program Studi", tickangle=-45, tickfont=dict(size=14), showgrid=False),
                yaxis=dict(title="Tingkat Kepuasan", showgrid=True, gridcolor="#28072E"),
                margin=dict(t=50, b=50, l=60, r=40),
                coloraxis_showscale=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Insight otomatis — dalam kotak ungu lembut dengan ikon lampu 💡
            top = avg.iloc[0]
            bottom = avg.iloc[-1]

            st.markdown(
                f"""
                <div class = 'insight' style='background-color:#f8f3ff; border-left:6px solid #7B1FA2;
                            padding:15px; border-radius:12px; margin-top:10px;
                            font-family:"Poppins" sans-serif; color:#4A148C; font-size:17px;'>
                    💡 <b>Program Studi {top['Program Studi']}</b> memiliki tingkat kepuasan tertinggi sebesar 
                    <b>{top['Tingkat Kepuasan']:.1f}</b>.<br>
                    Sementara itu, <b>{bottom['Program Studi']}</b> berada di posisi terendah dengan rata-rata kepuasan 
                    <b>{bottom['Tingkat Kepuasan']:.1f}</b>.<br>
                    Perbedaan ini bisa mencerminkan variasi dalam kualitas pembelajaran dan pengalaman mahasiswa di tiap program studi.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)


    # B. Distribusi Keinginan Pindah Jurusan (pie)
    if "Keinginan Pindah Jurusan" in data.columns:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Distribusi Keinginan Pindah Jurusan</div>", unsafe_allow_html=True)
        purple_palette = ["#d8b4fe", "#6749c2", "#441d88", "#261344"]

        pie = px.pie(
            data,
            names="Keinginan Pindah Jurusan",
            title="",
            hole=0.35,
            color_discrete_sequence=purple_palette
        )

        #Pengaturan tampilan pie chart
        pie.update_traces(
            textinfo="percent+label",
            textfont_size=16,
            textfont_color="black",
            pull=[0.03] * len(data["Keinginan Pindah Jurusan"].unique())
        )

        #Layout 
        pie.update_layout(
            showlegend=True,
            legend_title_text="Keinginan Pindah Jurusan",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=20, color="black")
            ),
            margin=dict(t=30, b=50, l=30, r=30),
            paper_bgcolor="#f3e8ff",
            plot_bgcolor="#F3E5F5"
        )

        st.plotly_chart(pie, use_container_width=True)

        #Insight
        pct = data["Keinginan Pindah Jurusan"].value_counts(normalize=True).mul(100).round(1)
        ya = pct.get("Ya", 0)
        st.markdown(
                f"""
                <div class='insight' 
                    style='background-color:#f8f3ff; 
                            border-left:5px solid #7B1FA2;
                            padding:15px; 
                            border-radius:12px; 
                            margin-top:10px;
                            font-family:Poppins, sans-serif; 
                            color:#4A148C; 
                            font-size:17px;'>
                    💡 Sekitar <b>{ya}%</b> responden menyatakan pernah ingin pindah jurusan. 
                    Ini indikator penting untuk menilai kecocokan jurusan vs harapan mahasiswa.
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)


    # C. Persepsi terhadap Jurusan (bar charts for several categorical columns)
    PRIMARY_HEX = "#512DA8"

    cols_persepsi = [
        "Relevansi Kurikulum Jurusan dengan Dunia Kerja",
        "Kesesuaian Jurusan dengan Minat",
        "Penilaian Prospek Kerja Jurusan"
    ]

    for col in cols_persepsi:
        if col in data.columns:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-title'>{col}</div>", unsafe_allow_html=True)

            vc = data[col].value_counts().reset_index()
            vc.columns = [col, "Jumlah"]
            vc = vc.sort_values(by="Jumlah", ascending=False).reset_index(drop=True)

            color_scale = ["#A78FE0", "#876ACA", "#7F5DCF", "#6941C7", "#4D29A0"]

            # Plot bar chart "Rata-rata Kepuasan"
            fig = px.bar(
                vc,
                x=col,
                y="Jumlah",
                text="Jumlah",
                color="Jumlah",
                color_continuous_scale=color_scale,
            )

            # Layout 
            fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=0.5)
            fig.update_layout(
                margin=dict(t=40, b=80),
                xaxis=dict(title="", tickangle=-35, tickfont=dict(size=16)),
                yaxis_title=dict(text="Jumlah Responden", font=dict(size=19)),
                coloraxis_showscale=False,
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Poppins"),
                paper_bgcolor="#f3e8ff",
                plot_bgcolor="#F3E5F5"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Insight 
            top_val = vc.iloc[0][col]
            st.markdown(f"""
            <div class = 'insight' 
                style='
                background-color: #f5edff;
                border-left: 5px solid #6A0DAD;
                padding: 10px 15px;
                border-radius: 10px;
                margin-top: 10px;
                margin-bottom: 30px;
                color: #3a0069;
                font-family: "Poppins", sans-serif;
                font-size: 17px
            '>
            💡 Mayoritas responden menilai <b>{col}</b> sebagai <b>{top_val}</b>. 
            Hal ini menunjukkan persepsi positif yang dominan. 
            Perhatikan distribusi kategori lain untuk perbaikan kurikulum dan strategi komunikasi antar jurusan.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Page: Hubungan Antar Variabel (Korelasi heatmap)
# ---------------------------
elif page == "🔗 Hubungan Antar Variabel":
    st.subheader("🔗 Hubungan Antar Variabel")
    num_cols = ["Tingkat Kepuasan", "Tingkat Kesulitan Mata Kuliah", "Tinggi Motivasi", "Jumlah Mata Kuliah Sesuai Minat", "Jumlah Stress dalam Seminggu"]
    num_cols = [c for c in num_cols if c in data.columns]
    if not num_cols:
        st.info("Tidak ada cukup variabel numerik untuk analisis korelasi.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Distribusi Keinginan Pindah Jurusan</div>", unsafe_allow_html=True)
        corr = data[num_cols].corr()
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#f3e8ff")  
        ax.set_facecolor("#F3E5F5")         
        sns.heatmap(
            corr, 
            annot=True, 
            cmap="Purples", 
            fmt=".2f", 
            linewidths=0.6, 
            vmin=-1, 
            vmax=1, 
            cbar_kws={"shrink": 0.3, "aspect": 5, "pad": 0.01},
            annot_kws={"size": 5, "color": "black"}
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=4)
        ax.tick_params(axis='x', labelsize=6, rotation=45)
        ax.tick_params(axis='y', labelsize=6)

        plt.subplots_adjust(bottom=0.25, top=0.95, left=0.25, right=0.90)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown("""
            <div class = 'insight' 
                style='
                background-color: #f5edff;
                border-left: 5px solid #6A0DAD;
                padding: 10px 15px;
                border-radius: 10px;
                margin-top: 10px;
                margin-bottom: 30px;
                color: #3a0069;
                font-family: "Poppins", sans-serif;
                font-size: 17px
            '>
        💡 Terdapat <b>korelasi positif</b> yang cukup kuat antara <b>Tingkat Kepuasan dan Tinggi Motivasi</b>, artinya <b>semakin tinggi motivasi mahasiswa, semakin tinggi pula tingkat kepuasan terhadap perkuliahan.</b><br>
            Selain itu, <b>Jumlah Mata Kuliah Sesuai Minat</b> juga menunjukkan <b>hubungan positif</b> dengan <b>Tingkat Kepuasan</b>, meskipun tidak terlalu kuat tetapi menunjukkan bahwa kesesuaian minat tetap berperan dalam kepuasan belajar.<br>
            Sementara itu, <b>Jumlah Stress dalam Seminggu</b> memiliki <b>korelasi negatif</b> dengan sebagian besar variabel lainnya, menandakan bahwa <b>semakin tinggi tingkat stres, cenderung menurunkan motivasi dan kepuasan mahasiswa.</b></div>""", unsafe_allow_html=True)

        # D. Kombinasi: Pairplot (Seaborn) + Cluster 3D
    num_cols = ["Tingkat Kepuasan", "Tingkat Kesulitan Mata Kuliah", "Tinggi Motivasi", "Jumlah Mata Kuliah Sesuai Minat", "Jumlah Stress dalam Seminggu"]
    num_cols = [c for c in num_cols if c in data.columns]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if len(num_cols) >= 2:
        sns.set(style="whitegrid", font="Poppins")

        # Pairplot 
        pairplot_fig = sns.pairplot(
            data[num_cols].dropna(),
            diag_kind="kde",
            corner=True,
            kind="reg",  
            plot_kws=dict(       
                line_kws=dict(color="#4D29A0", lw=1.5),  
                scatter_kws=dict(s=40, alpha=0.7, color="#4D29A0")  
            ),
            diag_kws=dict(color="#4D29A0", fill=True)
        )

        
        pairplot_fig.fig.patch.set_facecolor("#f3e8ff")  
        for ax in pairplot_fig.axes.flatten():
            if ax is not None:
                ax.set_facecolor("#F3E5F5")  

        # Jarak 
        pairplot_fig.fig.subplots_adjust(wspace=0.3, hspace=0.3)

        pairplot_fig.fig.suptitle(
            "Hubungan Antar Variabel Numerik",
            fontsize=15,
            color="#4D29A0",
            fontfamily="sans-serif",
            fontweight = 'bold',
            y=1.02
        )

        st.pyplot(pairplot_fig, use_container_width=True)

        # Insight 
        st.markdown("""
        <div class = 'insight'
            style='
            background-color: #f5edff;
            border-left: 5px solid #4D29A0;
            padding: 10px 15px;
            border-radius: 10px;
            margin-top: 10px;
            margin-bottom: 30px;
            color: #3a0069;
            font-family: "Poppins", sans-serif;
            font-size: 17px;
        '>
        💡 Sebagian besar titik pada scatter plot antara <b>Tingkat Motivasi dan Tingkat Kepuasan</b> membentuk <b>pola naik</b> yang cukup jelas, menunjukkan kecenderungan bahwa mahasiswa dengan <b>motivasi tinggi juga memiliki kepuasan tinggi</b>.
        Sebaliknya, sebaran titik antara <b>Jumlah Stress dalam Seminggu</b> dengan variabel lain tampak <b>menyebar</b> tanpa pola tertentu, menandakan bahwa <b>stres tidak memiliki hubungan linear yang kuat dengan motivasi maupun kepuasan</b>.
        Selain itu, persebaran data yang padat di area nilai menengah menunjukkan bahwa sebagian besar mahasiswa berada pada tingkat motivasi dan kepuasan yang moderat, dengan hanya sedikit yang ekstrem di kedua sisi.
        </div>
        """, unsafe_allow_html=True)

        # Cluster 3D 
        if len(num_cols) >= 3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center; color:#5E35B1;'>Cluster 3D Mahasiswa Berdasarkan Aspek Akademik</h4>", unsafe_allow_html=True)

            X = data[num_cols].dropna()
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(Xs)

            X_plot = data[num_cols].dropna().copy()
            X_plot["Cluster"] = labels.astype(str)

            fig_cluster = px.scatter_3d(
                X_plot,
                x=num_cols[0],
                y=num_cols[1],
                z=num_cols[2],
                color="Cluster",
                color_discrete_sequence=['#4D29A0', '#8E44AD', '#BB8FCE'],
                width=900,
                height=650
            )

            fig_cluster.update_traces(marker=dict(size=8, opacity=0.9))
            fig_cluster.update_layout(
                legend=dict(
                    title="Cluster",
                    font=dict(size=20, color="#4D29A0"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="#4D29A0",
                    borderwidth=1
                ),
                scene=dict(
                    xaxis=dict(title=dict(text=num_cols[0], font=dict(size=16))),
                    yaxis=dict(title=dict(text=num_cols[1], font=dict(size=16))),
                    zaxis=dict(title=dict(text=num_cols[2], font=dict(size=16))),
                    bgcolor="#F3E5F5"
                ),
                paper_bgcolor="#f5edff"
            )

            st.plotly_chart(fig_cluster, use_container_width=True)

            # Insight 
            st.markdown("""
            <div class = 'insight'
                style='
                background-color: #f5edff;
                border-left: 5px solid #4D29A0;
                padding: 10px 15px;
                border-radius: 10px;
                margin-top: 10px;
                margin-bottom: 30px;
                color: #3a0069;
                font-family: "Poppins", sans-serif;
                font-size: 17px;
            '>
            💡 Terlihat adanya beberapa segmen mahasiswa berdasarkan pola nilai atau persepsi akademik.
            Misalnya kelompok dengan <b>motivasi tinggi</b> dan <b>tingkat kesulitan rendah</b>. 
            Informasi ini dapat membantu menentukan strategi pembinaan yang lebih tepat sasaran.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Page: Korelasi & Regresi Berganda (diperluas)
# ---------------------------
elif page == "📈 Regresi Berganda":
    st.subheader("📈 Korelasi & Regresi Linear (Lengkap)")

    # Kolom numerik untuk memilih
    num_cols_all = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(num_cols_all) < 2:
        st.warning("Dibutuhkan minimal dua variabel numerik untuk analisis regresi.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Pengaturan Model Regresi Linear</div>", unsafe_allow_html=True)

        st.markdown("""
            <div style="
                font-size: 15px; 
                color: 'Black'; 
                font-weight: 400; 
                font-family: 'Poppins', sans-serif; 
                margin-bottom: -15px;
            ">
                Pilih variabel dependen (Y):
            </div>
        """, unsafe_allow_html=True)

        dep_var = st.selectbox("", num_cols_all, index=0)

        st.markdown("""
            <div style="
                font-size: 15px; 
                color: 'Black'; 
                font-weight: 400; 
                font-family: 'Poppins', sans-serif; 
                margin-top: 10px; 
                margin-bottom: -15px;
            ">
                Pilih variabel independen (X) — minimal 1:
            </div>
        """, unsafe_allow_html=True)

        # Multiselect
        indep_vars = st.multiselect(
            "",
            [c for c in num_cols_all if c != dep_var],
            default=[c for c in num_cols_all if c != dep_var][:2]
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if indep_vars:
            model_df = data[[dep_var] + indep_vars].dropna()
            X = model_df[indep_vars]
            X = sm.add_constant(X)
            y = model_df[dep_var]

            model = sm.OLS(y, X).fit()
            with st.expander("📄 Ringkasan Output Regresi (klik untuk buka)"):
                summary_html = f"""
                    <div style="
                        font-family: 'Poppins', Sans-serif;
                        font-size: 15px;
                        color: black;
                        background-color: #f8f8f8;
                        padding: 15px;
                        border-radius: 10px;
                        border: 1px solid #ddd;
                        white-space: pre-wrap;
                    ">
                        {model.summary()}
                    </div>
                """
                st.markdown(summary_html, unsafe_allow_html=True)

            # Render equation, coefficients, p-values, R-squared
            coefs = model.params
            pvals = model.pvalues
            rsq = model.rsquared
            adj_rsq = model.rsquared_adj

            # Rumus: Y = a + b1*X1 + b2*X2 + ...
            intercept = coefs.get("const", 0.0)
            eq_parts = [f"{intercept:.4f}"]
            for var in indep_vars:
                b = coefs.get(var, 0.0)
                eq_parts.append(f"{b:.4f}·{var}")
            equation = " + ".join(eq_parts)
            equation = f"{dep_var} = {equation}"

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-title'>Rumus Model & Interpretasi</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="font-size:16px; color:black; font-family:Poppins, sans-serif; margin-bottom:8px;">
                    <b>Rumus model (estimasi):</b> <code>{equation}</code>
                </div>
                <div style="font-size:16px; color:black; font-family:Poppins, sans-serif; margin-bottom:8px;">
                    <b>R-squared:</b> {rsq:.4f} — proporsi variabilitas <b>{dep_var}</b> yang dijelaskan oleh model.
                </div>
                <div style="font-size:16px; color:black; font-family:Poppins, sans-serif;">
                    <b>Adjusted R-squared:</b> {adj_rsq:.4f}
                </div>
            """, unsafe_allow_html=True)

            # coefficients table
            coef_table = pd.DataFrame({
                "Variable": coefs.index,  
                "Coefficient": coefs.round(20).values,
                "p-value": pvals.round(20).values
            })

            html_table = coef_table.to_html(index=False)

            st.markdown(f"""
                <div style="font-size:16px; color:black; font-family:Poppins, sans-serif;">
                    {html_table}
                </div>
            """, unsafe_allow_html=True)

            # Auto interpretasi
            interpretations = []
            for var in indep_vars:
                b = coefs.get(var, 0.0)
                p = pvals.get(var, 1.0)
                sign = "naik" if b > 0 else "turun" if b < 0 else "tidak berubah"
                signif = "signifikan" if p < 0.05 else "tidak signifikan"
                interpretations.append(f"- Jika {var} <b>bertambah 1 unit</b>, maka {dep_var} diperkirakan <b>{sign} sebesar {abs(b):.4f} unit</b> (p={p:.4f}, {signif}).")
            st.markdown(
                """
                <div class='insight' style='
                    background-color: #f5edff;
                    border-left: 5px solid #4D29A0;
                    padding: 10px 15px;
                    border-radius: 10px;
                    margin-top: 10px;
                    margin-bottom: 30px;
                    color: #3a0069;
                    font-family: "Poppins", sans-serif;
                    font-size: 16px;
                '>
                """ + "<br>".join(interpretations) + "</div>",
                unsafe_allow_html=True
            )

            # Scatter + regression line plot (Jika pilih 1 variabel x)
            if len(indep_vars) == 1:
                xvar = indep_vars[0]
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='chart-title'>Plot {dep_var} vs {xvar} + Garis Regresi</div>", unsafe_allow_html=True)
                # scatter dan garis prediksi
                scatter_fig = px.scatter(model_df, x=xvar, y=dep_var, trendline="ols", trendline_color_override=PRIMARY_HEX,
                                         width=900, height=500, labels={xvar: xvar, dep_var: dep_var})
                scatter_fig.update_traces(marker=dict(size=7, opacity=0.8))
                scatter_fig.update_layout(transition={"duration":300})
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.markdown("""
                    <div class='insight'
                    style='
                    background-color: #f5edff;
                    border-left: 5px solid #4D29A0;
                    padding: 10px 15px;
                    border-radius: 10px;
                    margin-top: 10px;
                    margin-bottom: 30px;
                    color: #3a0069;
                    font-family: "Poppins", sans-serif;
                    font-size: 16px;
                '>
                💡 Plot menampilkan titik observasi dan garis regresi OLS; lihat p-value koefisien untuk menentukan signifikansi.
                </div>""", unsafe_allow_html=True)

            # Auto kesimpulan
            concl = []
            concl.append(f"Model menjelaskan {rsq*100:.2f}% variasi pada <b>{dep_var}</b> (R² = {rsq:.4f}).")

            sig_vars = [v for v in indep_vars if pvals.get(v, 1.0) < 0.05]
            if sig_vars:
                concl.append(f"Variabel signifikan: <b>{', '.join(sig_vars)}</b> (p < 0.05). Fokus pada variabel ini untuk intervensi.")
            else:
                concl.append("Tidak ada variabel independen yang signifikan pada α = 0.05 — pertimbangkan variabel lain atau model non-linear.")

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='chart-title'>Kesimpulan Otomatis dari Regresi</div>", unsafe_allow_html=True)

            st.markdown(f"""
                <div class='insight'
                    style='
                        background-color: #f5edff;
                        border-left: 5px solid #4D29A0;
                        padding: 10px 15px;
                        border-radius: 10px;
                        margin-top: 10px;
                        margin-bottom: 30px;
                        color: #3a0069;
                        font-family: "Poppins", sans-serif;
                        font-size: 16px;
                        line-height: 1.6;
                    '>
                    {"<br>".join(concl)}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------
# Page: Kesimpulan
# ---------------------------
elif page == "🧩 Kesimpulan":
    st.subheader("🧩 Kesimpulan & Rekomendasi")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        <div class="card metric" style="
            background-color: #f5edff;
            padding: 10px 15px;
            border-radius: 10px;
            margin-top: 10px;
            margin-bottom: 30px;
            color: #3a0069;
            font-family: 'Poppins', sans-serif;
            font-size: 19px;
        ">
            <h4>Kesimpulan Umum:</h4>
            <p>
                Mahasiswa Gen Z menunjukkan tingkat kepuasan yang beragam terhadap jurusan yang mereka pilih. 
                Secara umum, motivasi belajar dan kesesuaian mata kuliah dengan minat menjadi faktor yang paling berkorelasi positif dengan kepuasan. 
                Sebaliknya, tingkat stres cenderung menurunkan motivasi dan kepuasan belajar.
                Hasil analisis juga mengindikasikan bahwa kepuasan tertinggi muncul pada mahasiswa yang merasa jurusannya relevan dengan tujuan karier dan pengalaman belajar yang bermakna. 
                Model regresi menunjukkan bahwa variabel motivasi dan relevansi kurikulum memiliki pengaruh signifikan terhadap kepuasan secara keseluruhan.
            </p>
        </div>
        """, unsafe_allow_html=True)


    with colB:
        st.markdown("""
        <div class="card metric" style="
            background-color: #f5edff;
            padding: 10px 15px;
            border-radius: 10px;
            margin-top: 10px;
            margin-bottom: 30px;
            color: #3a0069;
            font-family: 'Poppins', sans-serif;
            font-size: 19px;
        ">
            <h4>Rekomendasi:</h4>
            <p> 
                1. Perkuat relevansi kurikulum dengan kebutuhan dunia kerja melalui magang terutama yang diadakan oleh pemerintah, proyek kolaboratif dengan perusahaan industri, dan pembelajaran berbasis praktik yang relevan dengan saat di dunia kerja. <br>
                2. Kembangkan program peningkatan motivasi belajar, seperti mentoring, workshop karier, dan pengakuan prestasi mahasiswa.<br>
                3. Lakukan evaluasi rutin pada jurusan dengan tingkat kepuasan rendah untuk mengidentifikasi masalah spesifik, baik dari sisi pengajaran maupun lingkungan belajar.<br>
                4. Kurangi faktor stres akademik dengan manajemen beban tugas yang seimbang dan akses mudah ke layanan konseling.<br>
                5. Gunakan hasil analisis data (misalnya clustering dan regresi) untuk mendukung pengambilan keputusan strategis dalam peningkatan kualitas pendidikan dan kesejahteraan mahasiswa.
            <p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# FOOTER (small)
# ---------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#6b4b8a;font-size:12px'>Made with 💜 — Dashboard by Kelompok Escape</div>", unsafe_allow_html=True)
