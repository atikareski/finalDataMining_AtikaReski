import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# --- 🎯 GANTI INI DENGAN URL MENTAH (RAW) GITHUB ANDA ---
GITHUB_RAW_URL = "GANTI_DENGAN_URL_RAW_GITHUB_ANDA"
# --------------------------------------------------------

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("Aplikasi Otomatisasi Segmentasi Pelanggan")
st.caption("Fokus: Memeriksa perilaku pelanggan baru dan memprediksi segmen klusternya.")

# Definisikan K Optimal (berdasarkan hasil analisis sebelumnya)
K_FIXED = 3 # Kita gunakan 3, yang merupakan K optimal dari data ini

# --- 1. Muat Data dan Standardisasi (Caching) ---
@st.cache_data
def load_and_preprocess_data(url):
    try:
        df = pd.read_csv(url)
        spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        X = df[spending_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return df, X_scaled, spending_cols, scaler
    except Exception as e:
        st.error(f"Gagal memuat data dari URL. Pastikan URL GitHub Raw Anda benar. Error: {e}")
        return pd.DataFrame(), None, None, None

df_original, X_scaled, spending_cols, scaler = load_and_preprocess_data(GITHUB_RAW_URL)

if df_original.empty or X_scaled is None:
    st.stop()

# --- 2. Latih Model K-Means & Logistik (Caching) ---
@st.cache_resource
def train_models(k, X_scaled, df_base, spending_cols):
    # K-Means Clustering (K=3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clustered = df_base.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_spending_means = df_clustered.groupby('Cluster')[spending_cols].mean().round(2)
    
    # Regresi Logistik untuk Prediksi
    X_logreg = X_scaled
    Y_logreg = df_clustered['Cluster']
    
    # Membagi data (tanpa stratify karena Kluster 2 sangat kecil)
    X_train, X_test, Y_train, Y_test = train_test_split(X_logreg, Y_logreg, test_size=0.3, random_state=42) 
    
    # Latih model Logistik
    model_logistic = LogisticRegression(
        random_state=42, 
        solver='lbfgs', 
        max_iter=1000, 
        multi_class='multinomial' # Perbaikan error
    )
    model_logistic.fit(X_train, Y_train)
    
    # Hitung akurasi model
    accuracy = accuracy_score(Y_test, model_logistic.predict(X_test))
    
    # Koefisien untuk Visualisasi
    coef_df = pd.DataFrame(model_logistic.coef_, columns=spending_cols, index=[f'Koefisien Kluster {i}' for i in range(k)])
    
    return cluster_spending_means, model_logistic, accuracy, coef_df, df_clustered

cluster_spending_means, model_logistic, accuracy, coef_df, df_clustered = train_models(
    K_FIXED, X_scaled, df_original, spending_cols
)

# --- Fungsi Visualisasi Koefisien (Plot Batang) ---
def plot_logreg_coefficients(coef_df, k):
    fig, axes = plt.subplots(k, 1, figsize=(12, k * 3.5), sharex=True)
    if k == 2:
        axes = [axes] 
    
    plt.subplots_adjust(hspace=0.5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for i in range(k):
        cluster_name = f'Kluster {i}'
        ax = axes[i]
        coef_row = coef_df.loc[f'Koefisien Kluster {i}']
        
        ax.bar(coef_row.index, coef_row.values, color=np.where(coef_row.values > 0, colors[i % len(colors)], 'red'))
        
        ax.set_title(f'Bobot Fitur untuk Memprediksi {cluster_name}', fontsize=14)
        ax.set_ylabel('Koefisien LogReg', fontsize=10)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        
        if i == k - 1:
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.tick_params(axis='x', labelbottom=False)

    plt.suptitle('Koefisien Regresi Logistik (Pola Khas Setiap Kluster)', fontsize=16, fontweight='bold')
    return fig

# --- 3. TAMPILAN HASIL ANALISIS (Bagian Statis) ---
st.header(f"1. Analisis Segmentasi Kunci (K={K_FIXED})")

col_info, col_mean = st.columns([1, 2])

with col_info:
    st.metric(label="Akurasi Model Prediksi", value=f"{accuracy*100:.2f} %")
    st.caption("Akurasi ini menunjukkan seberapa baik model Logistik dapat mengklasifikasikan pelanggan ke kluster yang benar.")
    st.subheader("Ukuran Segmen")
    st.dataframe(df_clustered['Cluster'].value_counts().rename("Jumlah Pelanggan").to_frame(), use_container_width=True)

with col_mean:
    st.subheader("Profil Pengeluaran Rata-rata per Segmen")
    st.dataframe(cluster_spending_means, use_container_width=True)
    st.markdown("Interpretasi: Kluster 2 memiliki pengeluaran rata-rata tertinggi, Kluster 0 fokus pada Sembako, Kluster 1 fokus pada Segar.")

# Visualisasi Bobot Fitur (Koefisien LogReg)
st.header("2. Pola Perilaku Utama (Bobot Fitur)")
st.subheader("Fitur Kunci yang Digunakan Model untuk Mengklasifikasi")
st.markdown("Plot batang menunjukkan produk mana yang harus dicari untuk mengidentifikasi setiap segmen:")
fig_logreg = plot_logreg_coefficients(coef_df, K_FIXED)
st.pyplot(fig_logreg)


# --- 4. PREDIKSI PELANGGAN BARU (Fokus Utama Aplikasi) ---
st.header("3. Uji Pelanggan Baru dan Prediksi Segmen")
st.markdown("**Masukkan pengeluaran tahunan pelanggan baru untuk segera mengetahui segmen kluster mereka.**")

# Sidebar untuk Input
st.sidebar.header("Input Pelanggan Baru")
st.sidebar.markdown("Masukkan data pengeluaran (misal: dalam IDR):")

input_values = {}
input_cols = st.sidebar.columns(1) # Membuat kolom input vertikal di sidebar

for col_name in spending_cols:
    default_mean = int(df_original[col_name].mean())
    with input_cols[0]: 
        input_values[col_name] = st.number_input(
            f'{col_name} (Rata-rata: {default_mean:,})', 
            min_value=0, 
            value=default_mean,
            key=f'input_{col_name}'
        )

# Tombol Prediksi
if st.sidebar.button("Prediksi Kluster Pelanggan"):
    # 1. Ubah input menjadi DataFrame
    new_customer_data = pd.DataFrame([input_values])
    
    # 2. Standardisasi data input
    new_customer_scaled = scaler.transform(new_customer_data)
    
    # 3. Prediksi kluster
    prediction = model_logistic.predict(new_customer_scaled)
    prediction_proba = model_logistic.predict_proba(new_customer_scaled)[0]
    predicted_cluster = prediction[0]
    
    st.subheader("Hasil Prediksi Segmen")
    st.success(f"Pelanggan Baru Ini Adalah Kluster **{predicted_cluster}**.")
    
    st.markdown("##### Probabilitas Keanggotaan Kluster:")
    proba_df = pd.DataFrame(
        {'Kluster': [f'Kluster {i}' for i in range(K_FIXED)], 
         'Probabilitas': prediction_proba.round(4)
        }
    ).sort_values(by='Probabilitas', ascending=False)
    
    st.dataframe(proba_df, hide_index=True)
    
    st.markdown("---")
    st.markdown("**Tindakan Bisnis:**")
    if predicted_cluster == 2:
        st.warning("Kluster 2 (Super-Premium): Alihkan segera ke tim Key Account (Akun Kunci) dengan penawaran eksklusif.")
    elif predicted_cluster == 0:
        st.info("Kluster 0 (Ritel): Targetkan dengan diskon volume untuk Sembako dan Milk.")
    else:
        st.info("Kluster 1 (Restoran): Fokus pada kualitas produk Fresh dan logistik cepat.")
