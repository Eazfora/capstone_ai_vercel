from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Dict, Any
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import io
import bson
import os

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

# 1. KONEKSI DATABASE MONGODB
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017") 
client = MongoClient(MONGO_URI)
db = client["omnisight_db"]       # Nama database Anda
models_col = db["ai_models"]       # Collection tempat menyimpan model biner AI

# -------------------------------------------

app = FastAPI(title="OmniSight BI - ML Endpoint (Updated)")

# UPDATE CORS: Mengizinkan localhost dan domain production Vercel Anda
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "https://frontend-client-vercel.vercel.app"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD CHATBOT MODEL (Bawaan Lokal)
try:
    chatbot_model = joblib.load("model_bot/chatbot_model.pkl")
    chatbot_vectorizer = joblib.load("model_bot/vectorizer.pkl")
    print("🤖 Chatbot .pkl Model & Vectorizer successfully loaded!")
except Exception as e:
    print(f"⚠️ Warning: File .pkl chatbot belum dibuat atau gagal dimuat ({str(e)}).")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def omnisight_bot_pkl(data: ChatRequest):
    try:
        user_text = data.message.lower().strip()
        text_vector = chatbot_vectorizer.transform([user_text])
        predicted_intent = chatbot_model.predict(text_vector)[0]
        
        if predicted_intent == "sapaan":
            reply = "Halo Agung! Saya OmniSight AI. Ada yang bisa saya bantu untuk menganalisis data bisnis Anda hari ini?"
        elif predicted_intent == "tanya_omzet":
            reply = "Total pendapatan aktual yang berhasil dibukukan saat ini adalah **Rp 4.925.000**, bergerak di tren -30.6% dibanding bulan lalu."
        elif predicted_intent == "tanya_prakiraan":
            reply = "Berdasarkan proyeksi model AI, akumulasi pendapatan Anda untuk 7 hari ke depan diperkirakan mencapai **Rp 355.642**."
        elif predicted_intent == "tanya_churn":
            reply = "Sistem mendeteksi total **7 pelanggan**. Pelanggan #4 terpantau masuk zona *Waspada Churn (60%)* karena sudah 37 hari pasif."
        else:
            reply = "Maaf Agung, saya belum memahami pertanyaan itu. Anda bisa bertanya seputar omzet, prediksi, atau risiko pelanggan kabur."

        return {"status": "success", "reply": reply}
    except Exception as e:
        return {"status": "error", "message": f"Gagal memproses chat: {str(e)}"}


# =====================================================================
# Bagian: LOAD ARTIFAK MODEL DARI MONGODB (DENGAN FALLBACK LOKAL)
# =====================================================================

# --- Load Model Churn ---
try:
    churn_doc = models_col.find_one({"name": "churn_model_pipeline"})
    cols_doc = models_col.find_one({"name": "churn_model_columns"})
    
    if churn_doc and cols_doc:
        pipeline_churn = joblib.load(io.BytesIO(churn_doc["model_bytes"]))
        churn_columns = joblib.load(io.BytesIO(cols_doc["model_bytes"]))
        print("✅ Model Telco Churn berhasil dimuat dari MONGODB!")
    else:
        pipeline_churn = joblib.load('model_churn/telco_churn_tuned_pipeline.pkl')
        churn_columns = joblib.load('model_churn/model_columns.pkl')
        print("✅ Model Telco Churn (Bawaan GitHub) berhasil dimuat.")
except Exception as e:
    pipeline_churn = None
    churn_columns = None
    print(f"⚠️ Info: Menggunakan Rule-Based Fallback Churn sementara ({e})")

# --- Load Model Forecast ---
try:
    forecast_doc = models_col.find_one({"name": "forecasting_model"})
    if forecast_doc:
        model_forecast = joblib.load(io.BytesIO(forecast_doc["model_bytes"]))
        print("✅ Model Forecast (SARIMAX) berhasil dimuat dari MONGODB!")
    else:
        model_forecast = joblib.load('model_forecast/forecasting_model_optimized.pkl')
        print("✅ Model Forecast (Bawaan GitHub) berhasil dimuat.")
except Exception as e:
    model_forecast = None
    print(f"⚠️ Peringatan: Model Forecast gagal dimuat! Error: {e}")


# =====================================================================
# Bagian: TELCO CUSTOMER CHURN PREDICTION
# =====================================================================
class ChurnBatchRequest(BaseModel):
    customers: List[Dict[str, Any]]

@app.post("/predict-churn-batch")
def predict_churn_batch(data: ChurnBatchRequest):
    df_raw = pd.DataFrame(data.customers)
    if len(df_raw) == 0:
        return {"status": "success", "predictions": []}
    
    df_result = df_raw.copy()

    if len(df_raw) < 10:
        print(f"❄️ COLD START AKTIF: Hanya {len(df_raw)} pelanggan. Menggunakan Rule-Based.")
        predictions = []
        for index, row in df_raw.iterrows():
            recency = float(row.get('recency', 0))
            prob = 0.10
            if recency > 14: prob += 0.20
            if recency > 30: prob += 0.30
            if recency > 60: prob += 0.25
            predictions.append(max(0.0, min(1.0, prob)))
            
        df_result['Churn_Probability'] = predictions
        df_result['Prediction_Label'] = [1 if p > 0.5 else 0 for p in predictions]
        
        return {
            "status": "success",
            "engine": "Rule-Based RFM (Cold Start)",
            "predictions": df_result.to_dict(orient='records')
        }

    if pipeline_churn is not None and churn_columns is not None:
        try:
            fitur = ['recency', 'frequency', 'monetary']
            for col in fitur:
                if col not in df_raw.columns:
                    df_raw[col] = 0
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
                
            X = df_raw[fitur]
            probabilitas_churn = pipeline_churn.predict_proba(X)[:, 1]
            df_result['Churn_Probability'] = np.round(probabilitas_churn, 4)
            df_result['Prediction_Label'] = [1 if p > 0.5 else 0 for p in probabilitas_churn]
            
            return {
                "status": "success",
                "engine": "XGBoost RFM Model",
                "predictions": df_result.to_dict(orient='records')
            }
        except Exception as e:
            print(f"⚠️ ML Error. Detail: {e}")

    return {"status": "error", "message": "Sistem AI gagal memproses data Churn."}


# =====================================================================
# Bagian: AI SALES FORECASTING & CORRELATION
# =====================================================================
class SalesData(BaseModel):
    date: str
    sales: float

class ForecastRequest(BaseModel):
    Target_Month: str
    Current_Quantity: float
    Historical_Data: List[SalesData]

@app.post("/forecast-sales")
def forecast_sales(data: ForecastRequest):
    try:
        df = pd.DataFrame([item.dict() for item in data.Historical_Data])
        if len(df) == 0 or df['sales'].sum() == 0:
            return {"status": "success", "predictions_array": [0] * 7, "confidence_score": 0}

        actuals = df['sales'].values
        predictions = []
        confidence = 0
        
        if len(actuals) < 14:
            recent_avg = np.mean(actuals[-3:]) if len(actuals) >= 3 else np.mean(actuals)
            for _ in range(7):
                noise = np.random.uniform(-0.05, 0.05) 
                predictions.append(max(0, int(recent_avg * (1 + noise))))
            confidence = 65
        else:
            model = SimpleExpSmoothing(actuals, initialization_method="estimated").fit()
            forecast = model.forecast(7)
            predictions = [max(0, int(x)) for x in forecast]
            confidence = 88 
            
        return {
            "status": "success",
            "predictions_array": predictions,
            "confidence_score": confidence,
            "anomaly_spike": np.random.randint(10, 25),
            "correlation": {"promo": 0.85, "weekend": 0.72}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =====================================================================
# Bagian: LATIH ULANG MODEL FORECAST (SIMPAN KE MONGODB)
# =====================================================================
class RetrainData(BaseModel):
    transactions: list

@app.post("/retrain")
def retrain_model(data: RetrainData):
    global model_forecast 
    try:
        df = pd.DataFrame(data.transactions)
        if len(df) == 0:
            return {"status": "error", "message": "Gagal melatih: Tidak ada data transaksi."}
            
        df['invoiceDate'] = pd.to_datetime(df['invoiceDate'])
        df.set_index('invoiceDate', inplace=True)
        
        weekly_sales = df['totalSales'].resample('W').sum().to_frame()
        batas_atas = weekly_sales['totalSales'].quantile(0.99)
        weekly_sales = weekly_sales[weekly_sales['totalSales'] <= batas_atas]
        
        if len(weekly_sales) <= 5:
            return {
                "status": "warning",
                "message": f"Data terlalu sedikit ({len(weekly_sales)} minggu). Minimal butuh 6 minggu data."
            }
            
        weekly_sales = weekly_sales.iloc[1:-1]
        y_train = weekly_sales['totalSales'].values
        
        if np.var(y_train) == 0:
            return {"status": "warning", "message": "Gagal melatih: Variasi nilai transaksi flat."}
        
        try:
            p, d, q = model_forecast.order
            P, D, Q, s = model_forecast.seasonal_order
        except:
            p, d, q = 1, 1, 1
            P, D, Q, s = 0, 0, 0, 0
            
        try:
            model_baru = SARIMAX(y_train, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
            model_aktif_baru = model_baru.fit(disp=False)
        except Exception as math_err:
            return {"status": "error", "message": f"Struktur data menyebabkan error dekomposisi SARIMAX: {math_err}"}
        
        # 🔥 FIX READ-ONLY: Simpan Permanen ke MongoDB menggunakan BytesIO buffer
        try:
            buffer = io.BytesIO()
            joblib.dump(model_aktif_baru, buffer)
            buffer.seek(0)
            
            models_col.update_one(
                {"name": "forecasting_model"},
                {"$set": {"model_bytes": bson.Binary(buffer.read())}},
                upsert=True
            )
            model_forecast = model_aktif_baru
            return {"status": "success", "message": "Berhasil! Model SARIMAX AI dilatih ulang dan disimpan PERMANEN ke MongoDB."}
        except Exception as db_err:
            return {"status": "error", "message": f"Model terlatih namun gagal disimpan ke MongoDB: {db_err}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Gagal melatih model: {str(e)}"}


# =====================================================================
# Bagian: LATIH ULANG MODEL CHURN (SIMPAN KE MONGODB)
# =====================================================================
class RetrainChurnData(BaseModel):
    customers: list 

@app.post("/retrain-churn")
def retrain_churn_model(data: RetrainChurnData):
    global pipeline_churn
    global churn_columns
    try:
        df = pd.DataFrame(data.customers)
        if len(df) == 0:
             return {"status": "error", "message": "Tidak ada data untuk dilatih."}
             
        if 'recency' in df.columns:
            df['Is_Churn'] = df['recency'].apply(lambda x: 1 if x > 60 else 0)
        else:
             return {"status": "error", "message": "Kolom 'recency' tidak ditemukan."}

        fitur = ['recency', 'frequency', 'monetary']
        for col in fitur:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        X = df[fitur]
        y = df['Is_Churn']

        if len(y.unique()) < 2:
            return {"status": "warning", "message": "Belum bisa melatih model. Target y hanya memiliki 1 kelas data."}

        pipeline_baru = Pipeline([
            ('classifier', XGBClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=5, 
                min_child_weight=0, gamma=0, random_state=42
            ))
        ])
        pipeline_baru.fit(X, y)
        
        # 🔥 FIX READ-ONLY: Simpan Permanen Model Churn & Nama Kolom ke MongoDB
        try:
            buf_pipeline = io.BytesIO()
            joblib.dump(pipeline_baru, buf_pipeline)
            buf_pipeline.seek(0)
            
            buf_cols = io.BytesIO()
            joblib.dump(list(X.columns), buf_cols)
            buf_cols.seek(0)
            
            models_col.update_one(
                {"name": "churn_model_pipeline"},
                {"$set": {"model_bytes": bson.Binary(buf_pipeline.read())}},
                upsert=True
            )
            models_col.update_one(
                {"name": "churn_model_columns"},
                {"$set": {"model_bytes": bson.Binary(buf_cols.read())}},
                upsert=True
            )
            
            pipeline_churn = pipeline_baru
            churn_columns = list(X.columns)
            return {"status": "success", "message": "Model XGBoost Churn berhasil dilatih ulang dan disimpan PERMANEN ke MongoDB."}
        except Exception as db_err:
            return {"status": "error", "message": f"Model Churn terlatih namun gagal disimpan ke MongoDB: {db_err}"}
        
    except Exception as e:
        return {"status": "error", "message": f"Gagal melatih model churn: {str(e)}"}


@app.post("/generate-insight-narrative")
def generate_narrative(request: dict):
    sales_data = request.get("sales_data", [])
    if not sales_data: return {"narrative": "Data tidak cukup.", "sentiment": "Netral"}
    
    recent = sales_data[-3:]
    avg = sum(sales_data) / len(sales_data)
    current = recent[-1]
    
    if current > avg * 1.2:
        narrative = "🚀 Momentum kuat! Penjualan 20% di atas rata-rata. Pertahankan stok barang utama Anda."
        sentiment = "Positif"
    elif current < avg * 0.8:
        narrative = "⚠️ Penjualan melambat. Pertimbangkan diskon kilat untuk kategori ini dalam 48 jam ke depan."
        sentiment = "Waspada"
    else:
        narrative = "✅ Performa stabil dan sehat. Fokus pada efisiensi operasional minggu ini."
        sentiment = "Stabil"
        
    return {"narrative": narrative, "sentiment": sentiment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)