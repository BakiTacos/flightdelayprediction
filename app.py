import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Multi-Cluster Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Multi-Cluster Flight Delay Prediction App")
st.markdown("Aplikasi prediksi keterlambatan menggunakan pendekatan Multi-Model LightGBM berbasis Cluster Bandara.")
st.markdown("---")

# 2. Load Artefak Multi-Model
@st.cache_resource
def load_multi_model_artifacts():
    models = joblib.load("lgbm_cluster_models.pkl")         
    cluster_map = joblib.load("airport_cluster_mapping.pkl") 
    features = joblib.load("model_features.pkl")             
    return models, cluster_map, features

try:
    cluster_models, airport_cluster_mapping, expected_features = load_multi_model_artifacts()
    st.sidebar.success("✅ Semua Model Cluster & Pemetaan Berhasil Dimuat")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat komponen model: {e}")
    st.stop()

# --- DAFTAR KATEGORI MASTER YANG SEKARANG DIKUNCI 100% SAMA ---
CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'HA', 'EV', 'OO']
DEP_DAY_TYPE_OPTIONS = ['Night', 'Early_Morning', 'Morning', 'Midday', 'Afternoon', 'Evening']

BUSY_MONTHS = [7, 8, 10] 
CONGESTION_LOOKUP = {('JFK', 12): 45, ('LAX', 8): 60, ('ORD', 17): 75, ('ATL', 9): 90}
AIRPORT_OPTIONS = list(airport_cluster_mapping.keys()) if airport_cluster_mapping else ['LAX', 'JFK', 'ORD', 'ATL']
STATE_OPTIONS = ['Georgia', 'Illinois', 'Texas', 'Colorado', 'California', 'New York', 'Florida', 'North Carolina']
CITY_OPTIONS = ['Atlanta, GA', 'Chicago, IL', 'Dallas/Fort Worth, TX', 'Denver, CO', 'Los Angeles, CA', 'San Francisco, CA', 'New York, NY']

# 3. Form Input Pengguna
st.subheader("📊 Masukkan Informasi Penerbangan")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Jadwal & Waktu")
    month = st.slider("Bulan (Month)", min_value=1, max_value=12, value=6)
    day_of_week = st.slider("Hari dalam Seminggu (1=Senin, 7=Minggu)", min_value=1, max_value=7, value=3)
    crs_dep_time = st.number_input("Waktu Keberangkatan Terjadwal (Format HHMM, misal: 1530)", min_value=0, max_value=2359, value=1200, step=5)
    op_carrier_fl_num = st.number_input("Nomor Penerbangan (Flight Number)", min_value=1, max_value=9999, value=1234)
    
    st.markdown("### 🗺️ Jarak & Durasi")
    distance = st.number_input("Jarak Penerbangan (Distance in Miles)", min_value=10, max_value=10000, value=500)
    crs_elapsed_time = st.number_input("Durasi Terjadwal (CRS Elapsed Time in Minutes)", min_value=10, max_value=1000, value=120)

with col2:
    st.markdown("### ✈️ Maskapai & Lokasi")
    op_unique_carrier = st.selectbox("Maskapai (Carrier)", CARRIER_OPTIONS)
    origin = st.selectbox("Bandara Asal (Origin)", AIRPORT_OPTIONS, index=0)       
    origin_city_name = st.selectbox("Kota Asal (Origin City Name)", CITY_OPTIONS, index=4) 
    origin_state_nm = st.selectbox("Negara Bagian Asal (Origin State)", STATE_OPTIONS, index=4)
    
    st.markdown("---")
    dest = st.selectbox("Bandara Tujuan (Destination)", AIRPORT_OPTIONS, index=1 if len(AIRPORT_OPTIONS) > 1 else 0)    
    dest_city_name = st.selectbox("Kota Tujuan (Destination City Name)", CITY_OPTIONS, index=6) 
    dest_state_nm = st.selectbox("Negara Bagian Tujuan (Destination State)", STATE_OPTIONS, index=5)

st.markdown("---")

# 4. Prediction Execution
if st.button("🔮 Hitung Analisis & Prediksi Delay", type="primary", use_container_width=True):
    assigned_cluster = airport_cluster_mapping.get(origin, 0)
    model = cluster_models[assigned_cluster]
    
    # Feature Engineering
    dep_hour = int(crs_dep_time // 100) 
    is_busy_month = 1 if month in BUSY_MONTHS else 0
    
    if 0 <= dep_hour < 4: dep_day_type = 'Night'
    elif 4 <= dep_hour < 8: dep_day_type = 'Early_Morning'
    elif 8 <= dep_hour < 12: dep_day_type = 'Morning'
    elif 12 <= dep_hour < 16: dep_day_type = 'Midday'
    elif 16 <= dep_hour < 20: dep_day_type = 'Afternoon'
    else: dep_day_type = 'Evening'
        
    congestion_index = CONGESTION_LOOKUP.get((origin, dep_hour), 15)

    raw_input_data = {
        'month': int(month),
        'day_of_week': int(day_of_week),
        'op_unique_carrier': op_unique_carrier,
        'op_carrier_fl_num': float(op_carrier_fl_num),
        'crs_dep_time': int(crs_dep_time),
        'crs_elapsed_time': float(crs_elapsed_time),
        'distance': float(distance),
        'dep_hour': int(dep_hour),
        'is_busy_month': int(is_busy_month),
        'dep_day_type': dep_day_type,
        'congestion_index': int(congestion_index)
    }
    
    df_input = pd.DataFrame([raw_input_data])
    
    # Mengunci tipe data CategoricalDtype agar sinkron dengan levels master hasil training baru
    categories_dict = {
        'op_unique_carrier': CARRIER_OPTIONS,
        'dep_day_type': DEP_DAY_TYPE_OPTIONS
    }
    for col, categories in categories_dict.items():
        if col in df_input.columns:
            cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
            df_input[col] = df_input[col].astype(cat_type)

    # Safety Fill & Reorder Kolom
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[expected_features]
    
    # 5. Predict Execution
    try:
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        st.subheader("💡 Hasil Analisis Prediksi:")
        st.sidebar.info(f"📍 **Routing Status:** Bandara {origin} otomatis diproses menggunakan **Model Cluster {assigned_cluster}**.")

        if prediction == 1:
            st.error(f"⚠️ **Penerbangan Diprediksi DELAY** (Probabilitas Risiko Keterlambatan: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Penerbangan Diprediksi TEPAT WAKTU (ON TIME)** (Probabilitas On-Time: {prediction_proba[0]*100:.2f}%)")
            
        with st.expander("Lihat Rincian Probabilitas"):
            st.json({
                "Model Cluster Pemroses": int(assigned_cluster),
                "Probabilitas Tepat Waktu (On-Time)": f"{prediction_proba[0]*100:.2f}%",
                "Probabilitas Terlambat (Delay)": f"{prediction_proba[1]*100:.2f}%"
            })
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi model: {e}")