import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import warnings
import sys

# --- KONFIGURASI ---
FILE_PATH = "D:/COLLEGE!!/FIFTH SEMESTER/DATA MINING/DataMiningTeori/Wholesale customers data.csv" 
K_FIXED = 3 # K yang Dipilih
warnings.filterwarnings('ignore', category=UserWarning) 

# --- 1. Muat Data dan Standardisasi ---
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: Gagal memuat file. Harap pastikan file ada di path: {FILE_PATH}")
    sys.exit()

# Menampilkan data awal
print("--- 1. Data Awal (5 Baris Teratas) ---")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[spending_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Evaluasi dan K-Means Clustering ---

# PERHITUNGAN SILHOUETTE SCORE UNTUK K=2 HINGGA K=10
silhouette_scores = {}
print("\n--- 2. Evaluasi Silhouette Score untuk K-Means (K=2 hingga K=10) ---")
for k in range(2, 11):
    kmeans_eval = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans_eval.fit_predict(X_scaled)
    if len(np.unique(cluster_labels)) > 1:
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores[k] = score
        print(f"K={k}: {score:.4f}")

# K-Means FINAL (K=3)
kmeans = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Menampilkan hasil klustering (profil rata-rata)
cluster_spending_means = df.groupby('Cluster')[spending_cols].mean()
cluster_counts = df['Cluster'].value_counts().sort_index().to_frame('Jumlah Sampel')
print(f"\n--- 3. Hasil Klustering (Profil Rata-rata K={K_FIXED}) ---")
print(cluster_spending_means.to_markdown(numalign="left", stralign="left", floatfmt=".2f"))
print("\n--- Distribusi Kluster ---")
print(cluster_counts.to_markdown(numalign="left", stralign="left"))


# --- 4. Visualisasi Hasil Klustering (PCA) ---
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster']

print("\n--- 4. Visualisasi Hasil Klustering (PCA Plot) ---")

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    pca_df['PC1'], 
    pca_df['PC2'], 
    c=pca_df['Cluster'],
    cmap='viridis', 
    marker='o', 
    s=50, 
    alpha=0.6
)
ax.set_title(f'Peta Segmentasi Pelanggan (PCA, K={K_FIXED})')
ax.set_xlabel('Principal Component 1 (PC1)')
ax.set_ylabel('Principal Component 2 (PC2)')
ax.legend(*scatter.legend_elements(), title="Kluster")
plt.show() # Tampilkan plot


# --- 5. Pelatihan Regresi Logistik DENGAN OVERSAMPLING ---
print("\n--- 5. Penambahan Data (Oversampling) dan Pelatihan Model Prediksi ---")

X_logreg_original = X_scaled
Y_logreg_original = df['Cluster']

# Menjelaskan Oversampling
print("Kluster 2 (Super-Premium) memiliki jumlah sampel yang sangat sedikit (minoritas) di data asli.")
print("Dilakukan Oversampling (penggandaan data) Kluster 2 untuk menyeimbangkan data pelatihan,")
print("agar model Regresi Logistik mampu memprediksi Kluster 2 dengan keyakinan yang memadai.")

# IMPLEMENTASI OVERSAMPLING (Memperkuat Kluster 2)
kluster_2_index = Y_logreg_original[Y_logreg_original == 2].index
X_kluster_2 = X_logreg_original[kluster_2_index, :] 
Y_kluster_2 = Y_logreg_original.loc[kluster_2_index]

if len(Y_kluster_2) > 0:
    multiplier = 20 # Gandakan 20x
    X_oversampled_2 = np.tile(X_kluster_2, (multiplier, 1))
    Y_oversampled_2 = pd.concat([Y_kluster_2] * multiplier, ignore_index=True)
    
    # Gabungkan
    X_mayoritas = np.delete(X_logreg_original, kluster_2_index, axis=0)
    Y_mayoritas = Y_logreg_original.drop(kluster_2_index)

    X_new_logreg = np.concatenate((X_mayoritas, X_oversampled_2), axis=0)
    Y_new_logreg = pd.concat((Y_mayoritas, Y_oversampled_2))
    print(f"DEBUG: Kluster 2 digandakan {multiplier}x. Total sampel baru: {len(Y_new_logreg)}")
else:
    X_new_logreg = X_logreg_original
    Y_new_logreg = Y_logreg_original

# Split data (menggunakan data yang sudah diseimbangkan)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_new_logreg, 
    Y_new_logreg, 
    test_size=0.3, 
    random_state=42, 
    stratify=Y_new_logreg if len(Y_new_logreg.unique()) > 1 else None 
)

model_logistic = LogisticRegression(
    random_state=42,
    solver='lbfgs',
    max_iter=10000, 
    tol=0.0001
)
model_logistic.fit(X_train, Y_train)

# --- 6. Menampilkan Hasil Klasifikasi (Evaluasi Regresi Logistik) ---
Y_pred = model_logistic.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
existing_labels = Y_test.unique() 
existing_labels.sort()
filtered_target_names = [f'Kluster {i}' for i in existing_labels]

report = classification_report(
    Y_test, 
    Y_pred, 
    labels=existing_labels, 
    target_names=filtered_target_names, 
    output_dict=True, 
    zero_division=1
)

print(f"\n--- 6. Hasil Klasifikasi Regresi Logistik (Setelah Oversampling) ---")
print(f"Akurasi Model Regresi Logistik: {accuracy:.4f}")
print("\nLaporan Klasifikasi:")
print(pd.DataFrame(report).transpose().to_markdown(floatfmt='.4f'))

# --- LANGKAH PENTING: MENYIMPAN HANYA MODEL YANG DIBUTUHKAN ---

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model_logistic, 'model_logistic.pkl') 
joblib.dump(pca, 'pca.pkl')
joblib.dump(pca_df, 'pca_data_historis.pkl') 

print("\n=======================================================")
print("--- PELATIHAN DAN PENYIMPANAN MODEL SELESAI ---")
print("Anda harus MENGGANTI 4 file PKL di GitHub Anda sekarang.")
print("=======================================================")