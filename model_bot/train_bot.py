import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Dataset Latih: Daftarkan variasi pertanyaan Agung beserta label kategorinya
training_data = [
    # Kategori: Sapaan
    ("hi", "sapaan"),
    ("halo", "sapaan"),
    ("selamat pagi", "sapaan"),
    ("selamat siang", "sapaan"),
    ("p", "sapaan"),
    
    # Kategori: Tanya Omzet / Pendapatan
    ("berapa total pendapatan", "tanya_omzet"),
    ("omzet bulan ini berapa", "tanya_omzet"),
    ("total sales toko", "tanya_omzet"),
    ("cek keuntungan", "tanya_omzet"),
    ("pendapatan sekarang", "tanya_omzet"),
    
    # Kategori: Tanya Prakiraan / Prediksi
    ("bagaimana prediksi kedepan", "tanya_prakiraan"),
    ("prakiraan 7 hari ke depan", "tanya_prakiraan"),
    ("ramalan penjualan minggu depan", "tanya_prakiraan"),
    ("prediksi ai", "tanya_prakiraan"),
    ("omzet masa depan", "tanya_prakiraan"),
    
    # Kategori: Tanya Pelanggan / Churn
    ("siapa pelanggan yang mau kabur", "tanya_churn"),
    ("cek wawasan pelanggan", "tanya_churn"),
    ("pelanggan berisiko", "tanya_churn"),
    ("siapa yang perlu perhatian", "tanya_churn"),
    ("data churn rate", "tanya_churn"),
]

# Pisahkan pertanyaan (X) dan label jawaban (y)
X_train = [text for text, label in training_data]
y_train = [label for text, label in training_data]

# 2. Proses Ekstraksi Teks (Mengubah kata menjadi pembobotan angka)
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# 3. Latih Model menggunakan Logistic Regression
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# 4. Simpan hasilnya menjadi file .pkl open-source buatan Anda sendiri!
joblib.dump(model, "chatbot_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("🎉 Sukses! Model 'chatbot_model.pkl' & 'vectorizer.pkl' berhasil dibuat.")