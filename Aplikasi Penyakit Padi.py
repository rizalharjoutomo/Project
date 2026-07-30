import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import plotly.express as px
from datetime import datetime
import pandas as pd
import json
import io
import os

# ──────────────────────────────
# Konfigurasi Halaman
# ──────────────────────────────
st.set_page_config(
    page_title="🌾 Rice Disease Detection",
    #page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────
# Utilities (TETAP – logika tidak diubah)
# ──────────────────────────────
@st.cache_resource
def load_model(path="model_penyakit_padi_final2.keras"):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def load_class_names(default_order=None, fname="class_indices.json"):
    """
    Mengembalikan list nama kelas sesuai urutan pelatihan.
    """
    if os.path.exists(fname):
        try:
            with open(fname, "r") as f:
                class_indices = json.load(f)  # name -> index
            n_classes = len(class_indices)
            ordered = [None] * n_classes
            for name, idx in class_indices.items():
                ordered[idx] = name
            if any(x is None for x in ordered):
                raise ValueError("class_indices has gaps.")
            return ordered, True
        except Exception as e:
            st.warning(f"⚠️ Gagal membaca '{fname}': {e}. Menggunakan fallback.")
    return (default_order or ['bercak_daun','blast','bukan_padi','hawar_daun','normal','tungro']), False

def detect_input_size(model):
    try:
        shape = model.inputs[0].shape  # (None, H, W, C)
        h = int(shape[1]) if shape[1] is not None else 224
        w = int(shape[2]) if shape[2] is not None else 224
        return (w, h)
    except Exception:
        return (224, 224)

def model_has_rescaling(model):
    for layer in model.layers:
        name = layer.name.lower()
        if "rescaling" in name or layer.__class__.__name__.lower() == "rescaling":
            return True
    return False

def apply_preprocess(x, mode):
    """
    mode: ['auto', 'divide_255', 'vgg16', 'efficientnet', 'none']
    """
    if mode == "none":
        return x
    if mode == "divide_255":
        return x / 255.0
    if mode == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(x)
    if mode == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input(x)
    return x  # auto diputuskan di luar

# ──────────────────────────────
# Muat model
# ──────────────────────────────
model = load_model()
if model is None:
    st.stop()

input_size = detect_input_size(model)  # (W,H)
has_rescale = model_has_rescaling(model)

# ──────────────────────────────
# CSS & Tema (Tampilan saja)
# ──────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* Palet serba hitam */
:root{
  --bg1:#000000;
  --glass:#0a0a0a;            /* kontainer/card */
  --border:#1f2937;           /* garis pemisah */
  --text:#e5e7eb;             /* teks utama */
  --muted:#9ca3af;            /* teks sekunder */
}

/* Font & background */
html, body, .stApp { font-family:'Poppins',sans-serif; }
.stApp { background:#000000 !important; }

/* Header & Sidebar benar-benar hitam */
[data-testid="stHeader"] { background-color:#000000 !important; }
[data-testid="stHeader"] * { color: var(--text) !important; }
[data-testid="stSidebar"] { background-color:#000000 !important; color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"]{
  background:#0a0a0a; border:1px dashed #2b2b2b; color:var(--text);
}


/* Container */
.block-container{ max-width:1200px; padding-top:1.1rem; padding-bottom:1.1rem; }
.block-container{ padding-top: 2rem !important; }

/* Hero: hitam polos */
.hero{
  position:relative; padding:1.4rem 1.6rem; border-radius:18px;
  border:1px solid var(--border); background:#000000; color:var(--text);
  box-shadow:0 10px 30px rgba(0,0,0,.6);
}
.hero h1{ margin:0; color:var(--text); font-size:2.1rem; letter-spacing:.2px; }
.hero p{ margin:.3rem 0 0; color:var(--muted); }
.badges{ display:flex; gap:.5rem; margin-top:.8rem; flex-wrap:wrap; }
.badge{
  display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .6rem;
  border:1px solid var(--border); background:#0a0a0a; color:var(--text);
  border-radius:999px; font-size:.82rem; font-weight:600;
}
.hero{
  margin-top: .8rem !important;
  border-color: #2b2b2b !important; /* lebih kontras di atas bg hitam */
}

/* Cards hitam */
.card{
  background:#0a0a0a; border:1px solid var(--border); border-radius:16px;
  padding:1rem; color:var(--text); box-shadow:0 6px 20px rgba(0,0,0,.55);
}
.card h3,.card h4{ color:var(--text); margin-top:0; }

/* Prediction card */
.prediction{
  color:white; border-radius:16px; padding:1.2rem; text-align:center;
  box-shadow:0 10px 30px rgba(0,0,0,.7); background:#000000 !important;
}
.prediction .title{ font-size:1.4rem; font-weight:800; margin-bottom:.15rem; }
.prediction .sub{ opacity:.95; }

/* Confidence bar */
.conf-wrap{ margin-top:.6rem; text-align:left; }
.conf-bar{ height:12px; background:#1f2937; border-radius:999px; overflow:hidden; }
.conf-fill{ height:100%; background:linear-gradient(90deg, #666, #fff); width:0%; transition:width .6s ease; }
.conf-text{ font-size:.9rem; color:white; margin-top:.35rem; opacity:.95; }

/* Uploader tip */
.upload-tip{ border:1px dashed #2b2b2b; background:#0d0d0d; color:var(--text);
  border-radius:14px; padding:.85rem; text-align:center; margin-bottom:.6rem; }

/* Buttons */
.stButton>button, .stDownloadButton>button{
  background:#0a0a0a; color:#e5e7eb; border:1px solid var(--border);
  border-radius:10px; padding:.55rem .95rem; transition:transform .05s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  filter:brightness(1.1); transform:translateY(-1px);
}

/* Expanders */
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:14px; overflow:hidden; background:#0a0a0a; }
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] svg{ color:var(--text); }

/* Charts drop shadow */
.vega-embed, .plot-container{ filter:drop-shadow(0 2px 12px rgba(0,0,0,.6)); }

/* Footer */
.footer{
  text-align:center; color:var(--muted); border:1px solid var(--border);
  border-radius:14px; padding:1rem; background:#0a0a0a;
}
</style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────
# Sidebar (tampilan disempurnakan, fungsi sama)
# ──────────────────────────────
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.success("✅ Model berhasil dimuat")
    st.markdown("### ⚙️ Pengaturan")

    prep_mode = st.selectbox(
        "Preprocessing",
        [
            "Auto (detect)",
            "Divide by 255",
            "VGG16 preprocess_input",
            "EfficientNet preprocess_input",
            "None",
        ],
        index=4,  # tetap seperti kode asli
        help=(
            "Auto: jika model punya layer Rescaling, input tidak dinormalisasi di sini. "
            "Jika tidak, akan dibagi 255. Pilih manual jika Anda yakin pipeline training-nya berbeda."
        ),
    )

    confidence_threshold = st.slider("Ambang Confidence", 0.0, 1.0, 0.7, 0.05)

    use_enhancement_for_prediction = st.checkbox(
        "Gunakan gambar hasil enhancement untuk prediksi (tidak disarankan)", value=False
    )

    st.markdown("---")
    st.markdown("### 📁 Class Mapping")
    default_class_order = ['bercak_daun','blast','bukan_padi','hawar_daun','normal','tungro']
    class_names, loaded_from_json = load_class_names(default_order=default_class_order)

    if loaded_from_json:
        st.info("🔖 Urutan kelas diambil dari `class_indices.json` (sesuai training).")
    else:
        st.warning(
            "⚠️ `class_indices.json` tidak ditemukan. Menggunakan urutan fallback. "
            "Sebaiknya simpan `class_indices.json` dari training agar mapping tepat."
        )

    uploaded_ci = st.file_uploader("Opsional: unggah class_indices.json", type=["json"], key="ci_up")
    if uploaded_ci is not None:
        try:
            data = json.load(uploaded_ci)
            tmp_order = [None]*len(data)
            for name, idx in data.items():
                tmp_order[idx] = name
            if any(x is None for x in tmp_order):
                raise ValueError("class_indices.json tidak valid (ada index kosong).")
            class_names = tmp_order
            with open("class_indices.json", "w") as f:
                json.dump(data, f)
            st.success("✅ class_indices.json dimuat & disimpan. Urutan kelas diperbarui.")
        except Exception as e:
            st.error(f"❌ Gagal memproses class_indices.json: {e}")

    st.markdown("---")
    st.markdown("### 🔎 Tampilan")
    show_probabilities = st.checkbox("Tampilkan semua probabilitas", True)
    show_disease_info = st.checkbox("Tampilkan info penyakit", True)
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────
# Info Penyakit (Tetap)
# ──────────────────────────────
DISEASE_INFO = {
    'bercak_daun': {
        'title': 'Bercak Daun (Brown Spot)',
        'description': 'Penyakit jamur oleh Bipolaris oryzae. Muncul bercak coklat oval dengan halo kuning.',
        'symptoms': ['Bercak coklat oval', 'Halo kuning di sekitar bercak', 'Penurunan vigor tanaman'],
        'treatment': ['Fungisida kontak', 'Perbaikan nutrisi K', 'Pengelolaan kelembaban'],
        'severity': 'Sedang',
        'color': '#d35400',
        'prevention': 'Pemupukan seimbang (terutama K), sanitasi lahan, varietas toleran'
    },
    'blast': {
        'title': 'Blast',
        'description': 'Penyakit jamur (Magnaporthe oryzae). Bercak berbentuk belah ketupat dengan pusat abu-abu.',
        'symptoms': ['Bercak belah ketupat', 'Pusat bercak abu-abu', 'Tepi coklat'],
        'treatment': ['Fungisida sistemik', 'Pengaturan jarak tanam', 'Hindari kelebihan N'],
        'severity': 'Sangat Tinggi',
        'color': '#8e44ad',
        'prevention': 'Varietas tahan, drainase baik, pemupukan N berimbang'
    },
    'bukan_padi': {
        'title': 'Bukan Padi',
        'description': 'Gambar bukan daun padi (out-of-distribution).',
        'symptoms': ['Ciri tidak cocok dengan daun padi'],
        'treatment': ['Ambil ulang gambar pada daun padi', 'Pastikan fokus & pencahayaan baik'],
        'severity': 'N/A',
        'color': '#34495e',
        'prevention': 'Gunakan dataset/gambar yang sesuai domain'
    },
    'hawar_daun': {
        'title': 'Hawar Daun (BLB)',
        'description': 'Penyakit bakteri oleh Xanthomonas oryzae pv. oryzae. Gejala klorosis dan hawar dari ujung daun.',
        'symptoms': ['Lesi memanjang kekuningan', 'Daun mengering dari ujung', 'Terkadang eksudat bakteri'],
        'treatment': ['Varietas tahan', 'Pengelolaan air', 'Aplikasi bakterisida terarah'],
        'severity': 'Tinggi',
        'color': '#e67e22',
        'prevention': 'Rotasi, sanitasi, kontrol gulma, irigasi baik'
    },
    'normal': {
        'title': 'Daun Sehat',
        'description': 'Daun padi normal tanpa gejala penyakit.',
        'symptoms': ['Warna hijau segar', 'Tidak ada bercak/hawar', 'Pertumbuhan normal'],
        'treatment': ['Perawatan rutin', 'Pemupukan teratur', 'Monitoring berkala'],
        'severity': 'Tidak Ada',
        'color': '#27ae60',
        'prevention': 'Pemeliharaan optimal, monitoring rutin'
    },
    'tungro': {
        'title': 'Tungro',
        'description': 'Penyakit virus yang ditularkan wereng hijau (RTBV/RTSV).',
        'symptoms': ['Daun menguning-oranye', 'Pertumbuhan kerdil', 'Helaian daun menegang'],
        'treatment': ['Tanam serempak', 'Pengendalian wereng', 'Varietas tahan'],
        'severity': 'Tinggi',
        'color': '#c0392b',
        'prevention': 'Tanam serempak, sanitasi, pengendalian vektor'
    }
}

# ──────────────────────────────
# Header (Hero)
# ──────────────────────────────
st.markdown(
    f"""
<div class="hero">
  <h1>🌾 Smart Rice Disease Detection</h1>
  <p>Monitoring kesehatan tanaman berbasis AI • Input size: {input_size[0]}×{input_size[1]}</p>
  <div class="badges">
    <span class="badge">✅ Model: <strong>Loaded</strong></span>
    <span class="badge">🧪 Preprocess: <strong>{'Auto/Manual'}</strong></span>
    <span class="badge">🖼️ Format: <strong>RGB</strong></span>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

st.markdown("\n")

# ──────────────────────────────
# Konten Utama 
# ──────────────────────────────
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Unggah Gambar Daun Padi")
    st.markdown("<div class='upload-tip'>Tarik & letakkan berkas Anda di sini atau klik untuk memilih (jpg/jpeg/png)</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Pilih file gambar (jpg/jpeg/png)",
        type=["jpg", "jpeg", "png"],
        help="Upload gambar daun padi yang jelas untuk deteksi penyakit"
    )

    original_image = None
    enhanced_image = None

    if uploaded_file:
        original_image = Image.open(uploaded_file).convert('RGB')
        st.markdown("#### 🖼️ Pratinjau Gambar Asli")
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        with st.expander("🎨 Opsi Enhancement (hanya tampilan)"):
            enhance_contrast = st.slider("Kontras", 0.5, 2.0, 1.0, 0.1)
            enhance_brightness = st.slider("Kecerahan", 0.5, 2.0, 1.0, 0.1)
            enhance_sharpness = st.slider("Kejelasan", 0.5, 2.0, 1.0, 0.1)

            enhanced_image = original_image.copy()
            if enhance_contrast != 1.0:
                enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(enhance_contrast)
            if enhance_brightness != 1.0:
                enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(enhance_brightness)
            if enhance_sharpness != 1.0:
                enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(enhance_sharpness)

            if any([enhance_contrast != 1.0, enhance_brightness != 1.0, enhance_sharpness != 1.0]):
                st.image(enhanced_image, caption="Enhanced Image (tampilan)", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Statistik Cepat (demo)")
    disease_stats = pd.DataFrame({
        'Disease': class_names,
        'Percentage': np.random.randint(5, 20, size=len(class_names))
    })
    fig_pie = px.pie(disease_stats, values='Percentage', names='Disease', title="Distribusi Penyakit (contoh)")
    fig_pie.update_traces(hole=.45, textposition='inside')
    fig_pie.update_layout(
        height=320, showlegend=True, margin=dict(l=10,r=10,t=40,b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e5e7eb'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────
# Prediksi (UI saja – logika asli dipertahankan)
# ──────────────────────────────
if (original_image is not None) and (model is not None):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Hasil Analisis")

    with st.spinner("🤖 Menganalisis gambar..."):
        img_for_pred = enhanced_image if (use_enhancement_for_prediction and enhanced_image is not None) else original_image
        img_resized = img_for_pred.resize(input_size)
        x = tf.keras.preprocessing.image.img_to_array(img_resized)  # float32 [0..255]
        x = np.expand_dims(x, axis=0)

        auto_mode = "none"  # sesuai kode asli
        chosen = auto_mode if prep_mode.startswith("Auto") else {
            "Divide by 255": "divide_255",
            "VGG16 preprocess_input": "vgg16",
            "EfficientNet preprocess_input": "efficientnet",
            "None": "none"
        }[prep_mode]

        x = apply_preprocess(x, chosen)
        preds = model.predict(x, verbose=0)[0]  # (num_classes,)
        if preds.ndim != 1 or len(preds) != len(class_names):
            st.error("❌ Output model tidak cocok dengan jumlah kelas. Pastikan class_indices.json sesuai.")
            st.stop()

        pred_idx = int(np.argmax(preds))
        predicted_class = class_names[pred_idx]
        confidence = float(np.max(preds))

    info = DISEASE_INFO.get(predicted_class, {
        'title': predicted_class,
        'description': '-',
        'symptoms': [],
        'treatment': [],
        'severity': '-',
        'color': '#2c3e50',
        'prevention': '-'
    })
    disease_color = info['color']

    # Kartu Prediksi estetik
    st.markdown(
        f"""
        <div class=\"prediction\" style=\"background: linear-gradient(135deg, {disease_color} 0%, {disease_color}e6 100%);\"> 
            <div class=\"title\">{info['title']}</div>
            <div class=\"sub\">Confidence: {confidence:.1%}</div>
            <div class=\"conf-wrap\">
                <div class=\"conf-bar\"><div class=\"conf-fill\" style=\"width:{confidence*100:.2f}%\"></div></div>
                <div class=\"conf-text\">Ambang saat ini: {confidence_threshold:.0%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Indikator confidence asli (tetap)
    if confidence >= confidence_threshold:
        st.success(f"✅ Kepercayaan tinggi ({confidence:.1%})")
    else:
        st.warning(f"⚠️ Kepercayaan rendah ({confidence:.1%}). Pertimbangkan ambil ulang gambar.")

    # Distribusi probabilitas (dipercantik)
    if st.session_state.get("show_probabilities_flag", show_probabilities):
        st.markdown("#### 📊 Distribusi Probabilitas")
        prob_df = pd.DataFrame({
            'Disease': class_names,
            'Probability (%)': preds * 100.0
        })
        color_map = {name: DISEASE_INFO.get(name, {'color': '#7f8c8d'})['color'] for name in class_names}
        fig_bar = px.bar(
            prob_df, x='Disease', y='Probability (%)',
            color='Disease', color_discrete_map=color_map,
            title="Probabilitas Prediksi per Kelas"
        )
        fig_bar.update_layout(
            height=420, showlegend=False, margin=dict(l=10,r=10,t=50,b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e5e7eb'
        )
        fig_bar.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    # Informasi penyakit (tampilan baru)
    # ──────────────────────────────
    if show_disease_info:
        st.markdown("#### 📋 Informasi Penyakit")
        # buka kartu tunggal
        st.markdown(
            """
            <div class="card" style="background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02)); padding:1rem;">
            """,
            unsafe_allow_html=True,
        )

        # gambar di atas (pakai gambar yang dipakai untuk prediksi)
        st.markdown(
            "<div style='border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:.75rem;'>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # isi informasi dalam satu kolom/box
        st.markdown(
            f"""
            <h4 style="color:{info['color']};">🔬 Deskripsi</h4>
            <p style="color:#e5e7eb;">{info['description']}</p>

            <h4 style="color:{info['color']};">🎯 Keparahan</h4>
            <span class="badge" style="background:{info['color']}; color:white; border-color:transparent;">{info['severity']}</span>

            <h4 style="color:{info['color']}; margin-top:1rem;">🩺 Gejala</h4>
            """,
            unsafe_allow_html=True,
        )
        for s in info['symptoms']:
            st.markdown(f"- {s}")

        st.markdown(
            f"""
            <h4 style="color:{info['color']}; margin-top:1rem;">💊 Penanganan</h4>
            """,
            unsafe_allow_html=True,
        )
        for t in info['treatment']:
            st.markdown(f"- {t}")

        st.markdown(
            f"""
            <h4 style="color:{info['color']}; margin-top:1rem;">🛡️ Pencegahan</h4>
            <p style="color:#e5e7eb;">{info['prevention']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # Unduh laporan (tampilan sama, sedikit rapikan)
    st.markdown("#### 📄 Unduh Laporan")
    report_data = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Predicted Class': predicted_class,
        'Readable Title': info['title'],
        'Confidence': f"{confidence:.2%}",
        'Severity': info['severity'],
        'Description': info['description'],
        'Preprocessing Mode': chosen,
        'Input Size': f"{input_size[0]}x{input_size[1]}",
        'Has Rescaling in Model': has_rescale
    }
    report_df = pd.DataFrame([report_data])
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Analysis Report (CSV)",
        data=csv,
        file_name=f"rice_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────
# Footer
# ──────────────────────────────
st.markdown(
    """
<div class="footer">
  <h3 style="margin:.2rem 0; color:#e5e7eb;">🌾 Smart Rice Disease Detection System</h3>
  <p>Memberdayakan petani dengan monitoring kesehatan tanaman berbasis AI</p>
  <p>© 2025 • Dibuat dengan ❤️ untuk pertanian berkelanjutan</p>
</div>
    """,
    unsafe_allow_html=True,
)
