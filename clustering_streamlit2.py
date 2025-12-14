import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

GITHUB_RAW_URL = "https://raw.githubusercontent.com/atikareski/finalDataMining_AtikaReski/refs/heads/main/Wholesale%20customers%20data.csv"
# --------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Aplikasi Segmentasi Pelanggan Interaktif (K-Means)")
st.write("Gunakan slider di bawah untuk melihat bagaimana perubahan jumlah kluster (K) memengaruhi segmentasi pelanggan.")

# --- 1. Muat Data dan Persiapan ---
@st.cache_data
def load_and_preprocess_data(url):
    try:
        df = pd.read_csv(url)
        spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        X = df[spending_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return df, X_scaled, spending_cols
    except Exception as e:
        st.error(f"Gagal memuat data dari URL. Pastikan URL GitHub Raw Anda benar. Error: {e}")
        return pd.DataFrame(), None, None

df_original, X_scaled, spending_cols = load_and_preprocess_data(GITHUB_RAW_URL)

if df_original.empty or X_scaled is None:
    st.stop()

# --- TAMPILAN DATA AWAL (PERMINTAAN ANDA) ---
st.header("1. Data Awal")
st.dataframe(df_original.head())
# ---------------------------------------------

# --- 2. Fungsi Analisis Utama yang Bergantung pada K ---
def run_clustering_analysis(k, X_scaled, df_base, spending_cols):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_base['Cluster'] = kmeans.fit_predict(X_scaled)
   
    if k > 1:
        score = silhouette_score(X_scaled, df_base['Cluster'])
    else:
        score = 0
 
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df_base['Cluster']

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['Cluster'],
        cmap='viridis',
        marker='o',
        s=50,
        alpha=0.8
    )

    ax.set_title(f'Visualisasi Kluster Pelanggan (K={k}) | Skor Siluet: {score:.4f}', fontsize=16)
    ax.set_xlabel('Faktor Kebutuhan Pokok Ritel (PC1)', fontsize=12) 
    ax.set_ylabel('Faktor Bahan Baku Segar & Khusus (PC2)', fontsize=12) 
    
    legend1 = ax.legend(*scatter.legend_elements(), 
                        title="Kluster", 
                        loc="lower left", 
                        title_fontsize=12,
                        fontsize=10)
    ax.add_artist(legend1)
    ax.grid(True, linestyle='--', alpha=0.6)

    cluster_spending_means = df_base.groupby('Cluster')[spending_cols].mean().round(2)
    cluster_size = df_base['Cluster'].value_counts().sort_index()
    
    return fig, cluster_spending_means, cluster_size, score

# --- 3. Hitung K Optimal (Awal) dan Siapkan Slider ---
@st.cache_data
def calculate_optimal_k(X_scaled):
    silhouette_results = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_results.append({'k': k, 'Skor Siluet': score})
    
    silhouette_df = pd.DataFrame(silhouette_results)
    k_optimal_default = silhouette_df.loc[silhouette_df['Skor Siluet'].idxmax()]['k']
    return int(k_optimal_default), silhouette_df

k_optimal_default, silhouette_df = calculate_optimal_k(X_scaled)

# --- 4. Tampilkan Widget dan Hasil ---

st.sidebar.header("Kontrol Kluster")
selected_k = st.sidebar.slider(
    'Pilih Jumlah Kluster (K):',
    min_value=2, 
    max_value=10, 
    value=k_optimal_default,
    step=1
)

# Jalankan analisis dengan K yang dipilih
fig, cluster_spending_means, cluster_size, current_score = run_clustering_analysis(
    selected_k, 
    X_scaled, 
    df_original.copy(), 
    spending_cols
)

# --- Tampilkan Hasil Utama ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"Hasil Klusterisasi (K={selected_k})")
    st.subheader(f"Skor Siluet K={selected_k}: {current_score:.4f}")
    st.pyplot(fig)

with col2:
    st.header("Analisis Segmen")
    st.subheader("Rata-rata Pengeluaran Berdasarkan Segmen")
    st.dataframe(cluster_spending_means, use_container_width=True)
    
    st.subheader("Ukuran Setiap Segmen Pelanggan")
    st.dataframe(cluster_size.rename("Jumlah Pelanggan").to_frame(), use_container_width=True)

    st.subheader("Perbandingan K Optimal")
    st.dataframe(silhouette_df.style.format(precision=4), hide_index=True, use_container_width=True)
