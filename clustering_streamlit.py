import streamlit as st
import pandas as pd
import joblib 
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO # Digunakan untuk memuat konten biner dari URL

# --- 🎯 GANTI INI DENGAN URL FOLDER RAW GITHUB ANDA ---
# Contoh: https://raw.githubusercontent.com/username/repo/branch/folder_path/
MODEL_BASE_URL = "https://github.com/atikareski/finalDataMining_AtikaReski/raw/refs/heads/main/models/" 
# --------------------------------------------------------

# --- KONFIGURASI MODEL YANG DIMUAT ---
K_FIXED = 3 
SPENDING_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# --- 1. Muat Model dan Data (Caching) ---
@st.cache_resource
def load_and_preprocess_models():
    """Mengunduh dan memuat semua model dan data historis PCA dari GitHub."""
    
    # Fungsi pembantu untuk mengunduh dan memuat file PKL
    def fetch_model(filename):
        url = MODEL_BASE_URL + filename
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Muat konten biner
                return joblib.load(BytesIO(response.content))
            else:
                st.error(f"Gagal mengunduh {filename}. Status: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error memuat {filename}: {e}")
            return None

    # Muat semua objek yang disimpan
    scaler = fetch_model("scaler.pkl")
    model_logistic = fetch_model("model_logistic.pkl")
    pca = fetch_model("pca.pkl")
    pca_data_historis = fetch_model("pca_data_historis.pkl")
    
    # Ambil rata-rata dari scaler untuk input default (X_train mean)
    # Kita membuat DataFrame dummy hanya untuk mendapatkan kolom spending_cols
    if scaler is not None:
        X_means = pd.Series(scaler.mean_, index=SPENDING_COLS).round(0).astype(int)
    else:
        X_means = None

    if scaler is None or model_logistic is None or pca is None or pca_data_historis is None:
        st.error("Satu atau lebih file model (.pkl) gagal dimuat. Periksa kembali URL dan akses file.")
        st.stop()
        
    return scaler, model_logistic, pca, pca_data_historis, X_means

# Jalankan pemuatan model
scaler, model_logistic, pca, pca_data_historis, X_means = load_and_preprocess_models()

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("Aplikasi Prediksi Segmen Pelanggan Baru (K=3)")
st.caption("Model Logistik Regression memprediksi kluster berdasarkan model yang telah dilatih.")

# --- Fungsi Visualisasi PCA (Termasuk Pelanggan Baru) ---
def plot_pca_clusters(pca_data_historis, pca_obj, new_point=None, predicted_cluster=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Data Basis (Pelanggan Lama)
    scatter = ax.scatter(
        pca_data_historis['PC1'], 
        pca_data_historis['PC2'], 
        c=pca_data_historis['Cluster'],
        cmap='viridis', 
        marker='o', 
        s=50, 
        alpha=0.6,
        label='Data Historis'
    )
    
    # Plot Pelanggan Baru (Jika Ada)
    if new_point is not None:
        ax.scatter(
            new_point[0, 0], 
            new_point[0, 1], 
            color='red', 
            marker='*', 
            s=500, 
            edgecolors='black', 
            linewidth=1.5,
            label=f'Pelanggan Baru (Kluster {predicted_cluster})'
        )
        ax.annotate(
            'Pelanggan Baru', 
            (new_point[0, 0], new_point[0, 1]),
            textcoords="offset points", 
            xytext=(10, 10), 
            ha='center', 
            fontsize=12, 
            color='red'
        )
        
    ax.set_title(f'Peta Segmentasi Pelanggan (K={K_FIXED})', fontsize=16)
    ax.set_xlabel('Faktor Kebutuhan Pokok Ritel (PC1)', fontsize=12)
    ax.set_ylabel('Faktor Bahan Baku Segar & Khusus (PC2)', fontsize=12)
    
    legend1 = ax.legend(*scatter.legend_elements(), title="Kluster", loc="lower left", title_fontsize=12, fontsize=10)
    ax.add_artist(legend1)
    
    if new_point is not None:
        ax.legend(loc="upper right")
        
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

# --- Layout Aplikasi dan Input ---

st.sidebar.header("Uji Prediksi Pelanggan Baru")
st.sidebar.markdown("Masukkan pengeluaran tahunan (Rp):")

input_values = {}
for col_name in SPENDING_COLS:
    default_mean = X_means[col_name] if X_means is not None else 5000 # Fallback jika mean tidak termuat
    input_values[col_name] = st.sidebar.number_input(
        f'{col_name} (Rata-rata: {default_mean:,})', 
        min_value=0, 
        value=default_mean,
        key=f'input_{col_name}'
    )

predict_button = st.sidebar.button("Prediksi Segmen")

# --- Tampilan Hasil Statis ---
st.header("1. Peta Segmentasi Pelanggan Historis")
col_pca_display, col_results_display = st.columns([2, 1])

# Menampilkan PCA Plot Awal
with col_pca_display:
    st.subheader("Peta Segmentasi (PCA) Data Historis")
    fig_pca = plot_pca_clusters(pca_data_historis, pca)
    pca_plot_area = st.pyplot(fig_pca)

# Menampilkan Tinjauan Segmen
with col_results_display:
    st.subheader("Tinjauan Segmen Historis")
    # Karena kita tidak melatih model di sini, kita gunakan value_counts dari data PCA historis
    st.dataframe(pca_data_historis['Cluster'].value_counts().rename("Jumlah Pelanggan").to_frame(), use_container_width=True)
    st.info("Tekan tombol 'Prediksi Segmen' di sidebar untuk menguji pelanggan baru!")


# --- Logika Prediksi dan Pembaruan Plot ---
if predict_button:
    # 1. Persiapan Data Input
    new_customer_data = pd.DataFrame([input_values])
    new_customer_data = new_customer_data[SPENDING_COLS] 
    
    X_new = new_customer_data.values
    
    # 2. Standardisasi data input secara MANUAL
    # Menggunakan mean_ dan scale_ dari scaler yang dimuat
    new_customer_scaled = (X_new - scaler.mean_) / scaler.scale_
    
    # 3. Prediksi Kluster & Probabilitas
    prediction = model_logistic.predict(new_customer_scaled)
    prediction_proba = model_logistic.predict_proba(new_customer_scaled)[0]
    predicted_cluster = prediction[0]
    
    # 4. Transformasi ke ruang PCA untuk plot
    new_point_pca = pca.transform(new_customer_scaled)
    
    # 5. Update Plot PCA dengan Pelanggan Baru
    with col_pca_display:
        st.subheader("Peta Segmentasi Pelanggan (PCA) - Hasil Prediksi")
        fig_updated = plot_pca_clusters(pca_data_historis, pca, new_point_pca, predicted_cluster)
        pca_plot_area.pyplot(fig_updated) # Memperbarui plot

    # 6. Tampilkan Hasil Prediksi dan Tindakan Bisnis
    with col_results_display:
        st.subheader("3. Hasil Prediksi")
        st.success(f"Segmen Diprediksi: **Kluster {predicted_cluster}**")
        
        st.markdown("##### Probabilitas Keyakinan Model:")
        proba_df = pd.DataFrame(
            {'Kluster': [f'Kluster {i}' for i in range(K_FIXED)], 
             'Probabilitas': prediction_proba.round(4)
            }
        ).sort_values(by='Probabilitas', ascending=False)
        st.dataframe(proba_df, hide_index=True)
        
        st.markdown("---")
        st.markdown("##### Rekomendasi Tindakan Bisnis:")
        
        if predicted_cluster == 2:
            st.warning("Kluster 2 (Super-Premium): Alihkan segera ke tim Key Account (Akun Kunci) dengan penawaran eksklusif.")
        elif predicted_cluster == 0:
            st.info("Kluster 0 (Ritel): Targetkan dengan diskon volume pada produk Sembako dan produk berumur panjang.")
        else:
            st.info("Kluster 1 (Restoran): Fokus pada kualitas produk Fresh dan logistik pasokan yang cepat dan andal.")
