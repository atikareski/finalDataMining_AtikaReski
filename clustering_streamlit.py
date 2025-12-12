import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("Aplikasi Segmentasi Pelanggan (K-Means Clustering)")
st.write("Aplikasi ini melakukan segmentasi pelanggan grosir menggunakan K-Means dan Skor Siluet, serta memvisualisasikan hasilnya dengan PCA.")

# --- 1. Muat Data dan Persiapan ---
@st.cache_data # Streamlit cache data agar loading hanya terjadi sekali
def load_data(file_name):
    # Asumsi file berada di lokasi yang dapat diakses oleh Streamlit
    try:
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        st.error(f"File '{file_name}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return pd.DataFrame()

FILE_NAME = "https://raw.githubusercontent.com/atikareski/finalDataMining_AtikaReski/refs/heads/main/Wholesale%20customers%20data.csv"
df_original = load_data(FILE_NAME)

if not df_original.empty:
    st.header("1. Data Awal")
    st.dataframe(df_original.head())

    # --- Persiapan Data (Standardisasi) ---
    spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X = df_original[spending_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2. Tentukan k Optimal (Skor Siluet) ---
    st.header("2. Penentuan Jumlah Kluster Optimal (K)")
    
    silhouette_results = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_results.append({'k': k, 'Skor Siluet': score})

    silhouette_df = pd.DataFrame(silhouette_results)
    
    st.subheader("Hasil Perhitungan Skor Siluet Rata-rata")
    st.dataframe(silhouette_df, hide_index=True, use_container_width=True, height=350)

    k_optimal = silhouette_df.loc[silhouette_df['Skor Siluet'].idxmax()]['k']
    st.success(f"Skor Siluet tertinggi adalah **{silhouette_df['Skor Siluet'].max():.4f}** pada k = **{int(k_optimal)}**. Kami menggunakan k={int(k_optimal)} untuk clustering.")

    # --- 3. Terapkan K-Means Clustering dengan k optimal ---
    kmeans = KMeans(n_clusters=int(k_optimal), random_state=42, n_init=10)
    df_original['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # --- 4. Reduksi Dimensi untuk Visualisasi (PCA) ---
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df_original['Cluster']

    # --- 5. Visualisasi Kluster pada Plot 2D ---
    st.header("3. Visualisasi Kluster Pelanggan (PCA)")

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot data points
    scatter = ax.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['Cluster'],
        cmap='viridis',
        marker='o',
        s=50,
        alpha=0.8
    )

    # Menggunakan label sumbu yang lebih deskriptif
    ax.set_title(f'Visualisasi Kluster Pelanggan Berdasarkan Pola Pembelian (k={int(k_optimal)})', fontsize=16)
    ax.set_xlabel('Faktor Kebutuhan Pokok Ritel (PC1)', fontsize=12) # Label interpretatif
    ax.set_ylabel('Faktor Bahan Baku Segar & Khusus (PC2)', fontsize=12) # Label interpretatif
    
    # Menambahkan legenda
    legend1 = ax.legend(*scatter.legend_elements(), 
                        title="Kluster", 
                        loc="lower left", 
                        title_fontsize=12,
                        fontsize=10)
    ax.add_artist(legend1)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Menampilkan plot di Streamlit
    st.pyplot(fig)

    # --- 6. Profil Kluster (Rata-rata Pengeluaran) ---
    st.header("4. Profil Kluster (Rata-rata Pengeluaran)")
    cluster_spending_means = df_original.groupby('Cluster')[spending_cols].mean().round(2)
    
    st.write("Tabel di bawah menunjukkan pola pengeluaran rata-rata tahunan (dalam mata uang lokal) untuk setiap segmen pelanggan yang teridentifikasi.")
    st.dataframe(cluster_spending_means, use_container_width=True)
    
    # Tambahan: Menampilkan Ukuran Kluster untuk konteks
    st.subheader("Ukuran Setiap Kluster")
    cluster_size = df_original['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_size.rename("Jumlah Pelanggan").to_frame(), use_container_width=True)

    st.markdown("---")

    st.info("Berdasarkan analisis ini, Kluster 0 adalah Ritel Volume Tinggi, Kluster 1 adalah Horeca Standar, dan Kluster 2 adalah Pembeli Eksklusif Volume Sangat Tinggi.")
