import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("🔄 Memulai proses pembuatan dataset simulasi RFM...")

np.random.seed(42)
n_samples = 1000

recency = np.random.randint(1, 90, n_samples)  
frequency = np.random.randint(1, 15, n_samples) 
monetary = frequency * np.random.uniform(50000, 500000, n_samples) 

#! Logika untuk menentukan label "Churn" (1 = Churn, 0 = Setia)
#! Pelanggan cenderung churn JIKA: sudah lama tidak beli DAN frekuensinya sedikit

churn_labels = []
for r, f in zip(recency, frequency):
    prob_churn = 0.0
    if r > 30:
        prob_churn += 0.6  #! Risiko tinggi jika > 30 hari
    if f == 1:
        prob_churn += 0.2  #! Risiko tinggi jika baru beli 1x
    if r <= 14 and f >= 3:
        prob_churn = 0.05  # !Pelanggan setia yang baru beli, risiko sangat rendah
        
    # Tambahkan sedikit elemen acak (noise) agar AI belajar pola, bukan sekadar if-else
    prob_churn += np.random.uniform(-0.1, 0.1)
    
    # Tentukan final label (batas ambang 0.5)
    churn = 1 if prob_churn > 0.5 else 0
    churn_labels.append(churn)

# Buat DataFrame Pandas
df = pd.DataFrame({
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary,
    'Churn': churn_labels
})

print(f"✅ Dataset berhasil dibuat! Total data: {len(df)} pelanggan.")

# 2. PERSIAPAN PELATIHAN MODEL
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['Churn']

# Pisahkan data untuk training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Memulai pelatihan model Machine Learning (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Cek Akurasi Singkat
akurasi = model.score(X_test, y_test)
print(f"🎯 Pelatihan Selesai! Akurasi Model: {akurasi * 100:.2f}%")

# 3. MENYIMPAN ARTIFAK MODEL KE DALAM FILE (.pkl)
# Menyimpan otak AI
joblib.dump(model, 'rfm_churn_model.pkl')

# Menyimpan daftar kolom (agar API tahu fitur apa saja yang dibutuhkan)
model_columns = list(X.columns)
joblib.dump(model_columns, 'rfm_model_columns.pkl')

print("\n🎉 SUKSES! File 'rfm_churn_model.pkl' dan 'rfm_model_columns.pkl' telah dibuat di folder ini.")
print("Sekarang kamu bisa menyalakan ulang FastAPI kamu!")