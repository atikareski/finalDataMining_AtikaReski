import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# --- 🎯 GANTI INI DENGAN URL MENTAH (RAW) GITHUB ANDA ---
# Ganti ini agar Streamlit dapat memuat data dari sumber publik
GITHUB_RAW_URL = "https://raw.githubusercontent.com/atikareski/finalDataMining_AtikaReski/refs/heads/main/Wholesale%20customers%20data.csv"
# --------------------------------------------------------

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("Segmentasi Pelanggan (K-Means) & Otomatisasi Klasifikasi (LogReg)")
st.caption("Aliran Kerja: Clustering (Unsupervised) -> Classification (Supervised)")

# --- 1. Muat Data dan Standardisasi ---
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

# --- 2. Tentukan k Optimal (Skor Siluet) ---
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
    k_optimal = int(silhouette_df.loc[silhouette_df['Skor Siluet'].idxmax()]['k'])
    return k_optimal, silhouette_df

k_optimal, silhouette_df = calculate_optimal_k(X_scaled)

# --- Tampilan Utama Streamlit ---
st.header("1. Data dan K Optimal")
col_data, col_k = st.columns([1, 1])

with col_data:
    st.subheader("Data Awal (5 Baris Teratas)")
    st.dataframe(df_original.head(), use_container_width=True)

with col_k:
    st.subheader("Penentuan Jumlah Kluster (K)")
    st.dataframe(silhouette_df.style.format(precision=4), hide_index=True, use_container_width=True)
    st.success(f"Skor Siluet tertinggi pada K = **{k_optimal}**.")

# --- 3. Terapkan K-Means & Profil Kluster ---
@st.cache_data
def run_clustering(k, X_scaled, df_base, spending_cols):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_base['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_spending_means = df_base.groupby('Cluster')[spending_cols].mean().round(2)
    return df_base, cluster_spending_means

df_clustered, cluster_spending_means = run_clustering(k_optimal, X_scaled, df_original.copy(), spending_cols)

st.header(f"2. Hasil K-Means Clustering (K={k_optimal})")
st.subheader("Profil Kluster (Rata-rata Pengeluaran)")
st.dataframe(cluster_spending_means, use_container_width=True)

# --- 4. & 5. Visualisasi Kluster (PCA) ---
@st.cache_data
def plot_pca_clusters(df_clustered, X_scaled, k_optimal):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df_clustered['Cluster']

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'],
        cmap='viridis', marker='o', s=50, alpha=0.8
    )
    
    ax.set_title(f'Visualisasi Kluster Pelanggan Menggunakan PCA (k={k_optimal})', fontsize=16)
    ax.set_xlabel('Faktor Kebutuhan Pokok Ritel (PC1)', fontsize=12)
    ax.set_ylabel('Faktor Bahan Baku Segar & Khusus (PC2)', fontsize=12)
    ax.legend(*scatter.legend_elements(), title="Kluster", loc="lower left", title_fontsize=12, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

fig_pca = plot_pca_clusters(df_clustered, X_scaled, k_optimal)
st.subheader("Visualisasi Kluster (PCA)")
st.pyplot(fig_pca)


# ----------------------------------------------------------------------
st.header("3. Klasifikasi Lanjutan: Regresi Logistik")
st.markdown("Model Regresi Logistik digunakan untuk memprediksi kluster pelanggan baru, mengotomatisasi segmentasi.")

# --- 6. Model Regresi Logistik Multinomial ---
@st.cache_data
def run_logistic_regression(X_scaled, Y_logreg, spending_cols, k_optimal):
    # Ganti baris ini:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_logreg, test_size=0.3, random_state=42
    )

    # Menggunakan multi_class='auto' (yang defaultnya adalah multinomial jika lebih dari 2 kelas)
    model_logistic = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
    model_logistic.fit(X_train, Y_train)

    Y_pred = model_logistic.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # zero_division=1 untuk menghindari peringatan jika ada kluster yang tidak diprediksi
    report = classification_report(Y_test, Y_pred, target_names=[f'Kluster {i}' for i in range(k_optimal)], output_dict=True, zero_division=1)
    
    coef_df = pd.DataFrame(model_logistic.coef_, columns=spending_cols, index=[f'Koefisien Kluster {i}' for i in range(k_optimal)])
    
    return accuracy, report, coef_df

accuracy, report, coef_df = run_logistic_regression(X_scaled, df_clustered['Cluster'], spending_cols, k_optimal)

col_acc, col_rep = st.columns([1, 2])

with col_acc:
    st.subheader("Akurasi Model")
    st.metric(label="Akurasi Klasifikasi", value=f"{accuracy*100:.2f} %")

with col_rep:
    st.subheader("Laporan Klasifikasi per Kluster")
    st.dataframe(pd.DataFrame(report).transpose().iloc[:-3].style.format(precision=4), use_container_width=True)


# --- 7. Visualisasi Koefisien Regresi Logistik ---
def plot_logreg_coefficients(coef_df, k_optimal, spending_cols):
    fig, axes = plt.subplots(k_optimal, 1, figsize=(12, k_optimal * 4), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for i in range(k_optimal):
        cluster_name = f'Kluster {i}'
        ax = axes[i]
        coef = coef_df.iloc[i]
        
        ax.bar(coef.index, coef.values, color=np.where(coef.values > 0, colors[i], 'red'))
        
        ax.set_title(f'Bobot Fitur untuk Memprediksi {cluster_name}', fontsize=14)
        ax.set_ylabel('Koefisien LogReg', fontsize=10)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        
        if i == k_optimal - 1:
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.tick_params(axis='x', labelbottom=False)

    plt.suptitle('Visualisasi Koefisien Regresi Logistik (Bobot Fitur)', fontsize=16, fontweight='bold')
    return fig

fig_logreg = plot_logreg_coefficients(coef_df, k_optimal, spending_cols)
st.subheader("4. Bobot Fitur (Koefisien Logistik)")
st.pyplot(fig_logreg)

st.markdown("""
### Interpretasi Visualisasi Koefisien:
Visualisasi ini menunjukkan 'resep' untuk setiap kluster:
* **Batang Positif:** Fitur ini **meningkatkan** peluang pelanggan masuk ke kluster tersebut.
* **Batang Negatif:** Fitur ini **menurunkan** peluang pelanggan masuk ke kluster tersebut (karena itu adalah ciri khas kluster lain).
""")
