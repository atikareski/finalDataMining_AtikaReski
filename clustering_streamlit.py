import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# --- 🎯 GANTI INI DENGAN URL MENTAH (RAW) GITHUB ANDA ---
GITHUB_RAW_URL = "https://raw.githubusercontent.com/atikareski/finalDataMining_AtikaReski/refs/heads/main/Wholesale%20customers%20data.csv"
# --------------------------------------------------------

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("Segmentasi Pelanggan (K-Means) & Klasifikasi Interaktif")
st.caption("Aplikasi ini memungkinkan Anda memilih jumlah kluster (K) dan menguji prediksi pelanggan baru.")

# --- 1. Muat Data dan Standardisasi ---
# @st.cache_data memastikan fungsi hanya berjalan sekali
@st.cache_data
def load_and_preprocess_data(url):
    try:
        df = pd.read_csv(url)
        spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        X = df[spending_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Simpan scaler untuk digunakan nanti dalam prediksi pelanggan baru
        return df, X_scaled, spending_cols, scaler
    except Exception as e:
        st.error(f"Gagal memuat data dari URL. Pastikan URL GitHub Raw Anda benar. Error: {e}")
        return pd.DataFrame(), None, None, None

df_original, X_scaled, spending_cols, scaler = load_and_preprocess_data(GITHUB_RAW_URL)

if df_original.empty or X_scaled is None:
    st.stop()

# --- 2. Hitung K Optimal (Awal) ---
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
    k_optimal_default = int(silhouette_df.loc[silhouette_df['Skor Siluet'].idxmax()]['k'])
    return k_optimal_default, silhouette_df

k_optimal_default, silhouette_df = calculate_optimal_k(X_scaled)

# --- SIDEBAR INTERAKTIF: Pemilihan K ---
st.sidebar.header("Kontrol Interaktif")
selected_k = st.sidebar.slider(
    '1. Pilih Jumlah Kluster (K):',
    min_value=2, 
    max_value=10, 
    value=k_optimal_default, 
    step=1
)
st.sidebar.markdown(f"**K Optimal Rekomendasi:** {k_optimal_default}")
st.sidebar.dataframe(silhouette_df[['k', 'Skor Siluet']].set_index('k'), use_container_width=True)

# --- Fungsi Utama Analisis (Bergantung pada K) ---
def run_full_analysis(k, X_scaled, df_base, spending_cols):
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clustered = df_base.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_spending_means = df_clustered.groupby('Cluster')[spending_cols].mean().round(2)
    
    # PCA & Visualisasi
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df_clustered['Cluster']

    fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
    scatter = ax_pca.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'],
                             cmap='viridis', marker='o', s=50, alpha=0.8)
    ax_pca.set_title(f'Visualisasi Kluster Pelanggan (K={k})', fontsize=16)
    ax_pca.set_xlabel('Faktor Kebutuhan Pokok Ritel (PC1)', fontsize=12)
    ax_pca.set_ylabel('Faktor Bahan Baku Segar & Khusus (PC2)', fontsize=12)
    ax_pca.legend(*scatter.legend_elements(), title="Kluster", loc="lower left", title_fontsize=12, fontsize=10)
    
    # Regresi Logistik
    X_logreg = X_scaled
    Y_logreg = df_clustered['Cluster']
    X_train, X_test, Y_train, Y_test = train_test_split(X_logreg, Y_logreg, test_size=0.3, random_state=42, stratify=Y_logreg)
    model_logistic = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
    model_logistic.fit(X_train, Y_train)
    
    Y_pred = model_logistic.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    coef_df = pd.DataFrame(model_logistic.coef_, columns=spending_cols, index=[f'Koefisien Kluster {i}' for i in range(k)])

    return df_clustered, cluster_spending_means, fig_pca, model_logistic, accuracy, coef_df

# Jalankan analisis dengan K yang dipilih
df_clustered, cluster_spending_means, fig_pca, model_logistic, accuracy, coef_df = run_full_analysis(
    selected_k, X_scaled, df_original.copy(), spending_cols
)

# --- TAMPILAN HASIL KLUSTER ---
st.header(f"2. Hasil K-Means Clustering (K={selected_k})")
col_mean, col_pca = st.columns([1, 2])

with col_mean:
    st.subheader("Profil Pengeluaran")
    st.dataframe(cluster_spending_means, use_container_width=True)
    st.subheader("Ukuran Segmen")
    st.dataframe(df_clustered['Cluster'].value_counts().rename("Jumlah Pelanggan").to_frame(), use_container_width=True)

with col_pca:
    st.subheader("Peta Segmentasi (PCA)")
    st.pyplot(fig_pca)

# --- TAMPILAN REGRESI LOGISTIK ---
st.header("3. Klasifikasi & Prediksi")
st.metric(label=f"Akurasi Model Regresi Logistik (K={selected_k})", value=f"{accuracy*100:.2f} %")

# Visualisasi Koefisien LogReg
def plot_logreg_coefficients(coef_df, k):
    fig, axes = plt.subplots(k, 1, figsize=(12, k * 3.5), sharex=True)
    if k == 2: # Penanganan khusus jika k=2 karena axes hanya 1D
        axes = [axes] 
    
    plt.subplots_adjust(hspace=0.5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i in range(k):
        cluster_name = f'Kluster {i}'
        ax = axes[i]
        coef = coef_df.iloc[i]
        ax.bar(coef.index, coef.values, color=np.where(coef.values > 0, colors[i % len(colors)], 'red'))
        ax.set_title(f'Bobot Fitur untuk Memprediksi {cluster_name}', fontsize=14)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        if i == k - 1:
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.tick_params(axis='x', labelbottom=False)

    plt.suptitle('Bobot Fitur Regresi Logistik (Pola Khas Setiap Kluster)', fontsize=16, fontweight='bold')
    return fig

fig_logreg = plot_logreg_coefficients(coef_df, selected_k)
st.subheader("Visualisasi Koefisien Logistik")
st.pyplot(fig_logreg)

# --- 4. PREDIKSI PELANGGAN BARU INTERAKTIF ---
st.header("4. Uji Prediksi Pelanggan Baru")
st.sidebar.header("Uji Pelanggan Baru")
st.sidebar.markdown("Masukkan pengeluaran tahunan (satuan yang sama dengan data asli):")

input_values = {}
input_cols = st.sidebar.columns(3)

for i, col_name in enumerate(spending_cols):
    # Menggunakan slider atau number_input yang sesuai dengan skala data
    default_mean = int(df_original[col_name].mean())
    with input_cols[i % 3]: # Membagi input ke 3 kolom di sidebar
        input_values[col_name] = st.number_input(
            f'{col_name} (Rata-rata: {default_mean:,})', 
            min_value=0, 
            value=default_mean,
            key=f'input_{col_name}'
        )

# Fungsi Prediksi
if st.sidebar.button("Prediksi Kluster"):
    # Ubah input menjadi DataFrame
    new_customer_data = pd.DataFrame([input_values])
    
    # 1. Standardisasi data input menggunakan scaler yang dilatih pada data asli
    new_customer_scaled = scaler.transform(new_customer_data)
    
    # 2. Prediksi kluster menggunakan model LogReg yang sudah dilatih
    prediction = model_logistic.predict(new_customer_scaled)
    prediction_proba = model_logistic.predict_proba(new_customer_scaled)[0]
    
    predicted_cluster = prediction[0]
    
    st.subheader("Hasil Prediksi:")
    st.success(f"Pelanggan Baru Diprediksi Berada di Kluster **{predicted_cluster}**.")
    
    st.markdown("##### Probabilitas Keanggotaan Kluster:")
    proba_df = pd.DataFrame(
        {'Kluster': [f'Kluster {i}' for i in range(selected_k)], 
         'Probabilitas': prediction_proba.round(4)
        }
    ).sort_values(by='Probabilitas', ascending=False)
    
    st.dataframe(proba_df, hide_index=True)
