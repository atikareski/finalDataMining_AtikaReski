import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("Aplikasi Segmentasi Pelanggan (K-Means Clustering)")
st.write("Aplikasi ini melakukan segmentasi pelanggan grosir menggunakan K-Means dan Silhouette Score, serta memvisualisasikan hasilnya dengan PCA.")

@st.cache_data
def load_data(file_name):
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

    spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X = df_original[spending_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.header("2. Penentuan Jumlah Kluster Optimal (K)")
    
    silhouette_results = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_results.append({'k': k, 'Silhouette Score': score})

    silhouette_df = pd.DataFrame(silhouette_results)
    
    st.subheader("Hasil Perhitungan Silhouette Score Rata-rata")
    st.dataframe(silhouette_df, hide_index=True, use_container_width=True, height=350)

    k_optimal = silhouette_df.loc[silhouette_df['Skor Siluet'].idxmax()]['k']
    st.success(f"Silhouette Score tertinggi adalah **{silhouette_df['Skor Siluet'].max():.4f}** pada k = **{int(k_optimal)}**. Menggunakan k={int(k_optimal)} untuk clustering.")

    kmeans = KMeans(n_clusters=int(k_optimal), random_state=42, n_init=10)
    df_original['Cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df_original['Cluster']

    st.header("3. Visualisasi Kluster Pelanggan (PCA)")

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

    ax.set_title(f'Visualisasi Kluster Pelanggan Berdasarkan Pola Pembelian (k={int(k_optimal)})', fontsize=16)
    ax.set_xlabel('Faktor Kebutuhan Pokok Ritel (PC1)', fontsize=12)
    ax.set_ylabel('Faktor Bahan Baku Segar & Khusus (PC2)', fontsize=12)

    legend1 = ax.legend(*scatter.legend_elements(), 
                        title="Kluster", 
                        loc="lower left", 
                        title_fontsize=12,
                        fontsize=10)
    ax.add_artist(legend1)
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

    st.header("4. Profil Kluster (Rata-rata Pengeluaran)")
    cluster_spending_means = df_original.groupby('Cluster')[spending_cols].mean().round(2)
    
    st.write("Tabel di bawah menunjukkan pola pengeluaran rata-rata tahunan (dalam mata uang lokal) untuk setiap segmen pelanggan yang teridentifikasi.")
    st.dataframe(cluster_spending_means, use_container_width=True)

    st.subheader("Ukuran Setiap Kluster")
    cluster_size = df_original['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_size.rename("Jumlah Pelanggan").to_frame(), use_container_width=True)

    st.markdown("### Kesimpulan")
    st.markdown("""
        Analisis berhasil mengelompokkan pelanggan menjadi **3 segmen strategis** berdasarkan kebiasaan belanja:
        
        | Segmen | Perilaku Kunci | Strategi Bisnis |
        | :--- | :--- | :--- |
        | **Toko Ritel & Kebutuhan Pokok (Kluster 0)** | Pengeluaran tinggi pada Sembako, Susu, dan Detergents/Kertas. | Fokus pada **Diskon Volume** dan efisiensi logistik. |
        | **Pemilik Restoran atau Katering (Kluster 1)** | Pengeluaran tinggi pada Produk **Segar (Fresh)** dengan jumlah tinggi. | Fokus pada **Kualitas Bahan Baku** dan konsistensi pasokan. |
        | **Pembeli Eksklusif Super-Premium (Kluster 2)** | Pengeluaran ekstrem pada **Frozen** dan **Delicassen**. | Perlakukan sebagai Akun Kunci (Key Account) memprioritaskan layanan eksklusif dan retensi. |
    """)



