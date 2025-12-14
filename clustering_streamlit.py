import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings

# --- 🎯 GANTI INI DENGAN URL MENTAH (RAW) GITHUB ANDA ---
GITHUB_RAW_URL = "https://raw.githubusercontent.com/atikareski/finalDataMining_AtikaReski/refs/heads/main/Wholesale%20customers%20data.csv"
# --------------------------------------------------------

# Definisikan K Optimal yang Ditetapkan
K_FIXED = 3 
# Warna untuk Kluster (akan digunakan di plot PCA)
CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("Aplikasi Prediksi Segmen Pelanggan Baru (K=3)")
st.caption("Model Logistik Regression otomatis mengklasifikasikan pelanggan baru berdasarkan pola pengeluaran.")

# --- 1. Muat Data dan Latih Model (Caching) ---
@st.cache_resource
def train_models(url):
    # Memuat dan Pra-pemrosesan
    try:
        df_base = pd.read_csv(url) 
    except Exception as e:
        st.error(f"Gagal memuat data dari URL. Error: {e}")
        return None, None, None, None, None, None, None

    spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X = df_base[spending_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Nonaktifkan ConvergenceWarning
    warnings.filterwarnings('ignore', category=UserWarning) 

    # K-Means Clustering (K=3)
    kmeans = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
    df_clustered = df_base.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_spending_means = df_clustered.groupby('Cluster')[spending_cols].mean().round(2)
    
    # Regresi Logistik (Klasifikasi)
    X_logreg = X_scaled
    Y_logreg = df_clustered['Cluster']
    X_train, X_test, Y_train, Y_test = train_test_split(X_logreg, Y_logreg, test_size=0.3, random_state=42) 
    
    model_logistic = LogisticRegression(
        random_state=42, 
        solver='lbfgs',      
        max_iter=10000,        
        multi_class='multinomial',
        tol=0.0001
    )
    model_logistic.fit(X_train, Y_train)
    accuracy = accuracy_score(Y_test, model_logistic.predict(X_test))
    
    # PCA untuk Visualisasi
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    
    return df_clustered, cluster_spending_means, model_logistic, accuracy, pca, pca_data, scaler

# Jalankan pelatihan model dan dapatkan hasilnya
df_clustered, cluster_spending_means, model_logistic, accuracy, pca, pca_data, scaler = train_models(GITHUB_RAW_URL)

if df_clustered is None:
    st.stop()
    
spending_cols = df_clustered.columns[4:-1].tolist() # Ambil kolom pengeluaran

# --- Fungsi Visualisasi PCA (Termasuk Pelanggan Baru) ---
def plot_pca_clusters(pca_data, df_clustered, new_point=None, predicted_cluster=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Data Basis (Pelanggan Lama)
    scatter = ax.scatter(
        pca_data[:, 0], 
        pca_data[:, 1], 
        c=df_clustered['Cluster'],
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

# --- Layout Aplikasi ---

st.sidebar.header("Uji Prediksi Pelanggan Baru")
st.sidebar.markdown("Masukkan pengeluaran tahunan (Rp):")

# Input Interaktif
input_values = {}
for col_name in spending_cols:
    default_mean = int(df_clustered[col_name].mean())
    input_values[col_name] = st.sidebar.number_input(
        f'{col_name} (Rata-rata: {default_mean:,})', 
        min_value=0, 
        value=default_mean,
        key=f'input_{col_name}'
    )

# Tombol Prediksi
predict_button = st.sidebar.button("Prediksi Segmen")

# --- Tampilan Hasil Statis ---
st.header("1. Hasil Analisis Inti")

col_info, col_mean = st.columns([1, 2])

with col_info:
    st.metric(label="Akurasi Model Klasifikasi", value=f"{accuracy*100:.2f} %")
    st.markdown("Model Logistik sangat andal dalam memprediksi keanggotaan kluster.")
    st.subheader("Ukuran Segmen")
    st.dataframe(df_clustered['Cluster'].value_counts().rename("Jumlah Pelanggan").to_frame(), use_container_width=True)

with col_mean:
    st.subheader("Profil Pengeluaran Rata-rata per Segmen")
    st.dataframe(cluster_spending_means, use_container_width=True)

st.divider()

# --- Tampilan Plot PCA Default ---
col_pca_display, col_results_display = st.columns([2, 1])

with col_pca_display:
    st.subheader("2. Peta Segmentasi Pelanggan (PCA)")
    fig_pca = plot_pca_clusters(pca_data, df_clustered)
    pca_plot_area = st.pyplot(fig_pca)

# --- Logika Prediksi dan Pembaruan Plot ---
if predict_button:
    # 1. Persiapan Data Input
    new_customer_data = pd.DataFrame([input_values])
    new_customer_scaled = scaler.transform(new_customer_data)
    
    # 2. Prediksi Kluster & Probabilitas
    prediction = model_logistic.predict(new_customer_scaled)
    prediction_proba = model_logistic.predict_proba(new_customer_scaled)[0]
    predicted_cluster = prediction[0]
    
    # 3. Transformasi ke ruang PCA untuk plot
    new_point_pca = pca.transform(new_customer_scaled)
    
    # 4. Update Plot PCA dengan Pelanggan Baru
    with col_pca_display:
        st.subheader("Peta Segmentasi Pelanggan (PCA)")
        fig_updated = plot_pca_clusters(pca_data, df_clustered, new_point_pca, predicted_cluster)
        pca_plot_area.pyplot(fig_updated) # Memperbarui plot

    # 5. Tampilkan Hasil Prediksi dan Tindakan Bisnis
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
            st.warning("Kluster 2 (Super-Premium): Alihkan segera ke tim Key Account untuk layanan dan penawaran eksklusif.")
        elif predicted_cluster == 0:
            st.info("Kluster 0 (Ritel): Targetkan dengan diskon volume pada produk Sembako dan produk berumur panjang.")
        else:
            st.info("Kluster 1 (Restoran): Fokus pada kualitas produk Fresh dan logistik pasokan yang cepat dan andal.")
else:
    # Tampilkan hasil kluster di kolom hasil saat belum ada prediksi
    with col_results_display:
        st.subheader("Tinjauan Segmen")
        st.info("Tekan tombol 'Prediksi Segmen' di sidebar untuk menguji pelanggan baru!")
