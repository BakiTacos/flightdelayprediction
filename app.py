import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Delay Prediction App")
st.markdown("Prediksi status keterlambatan penerbangan berdasarkan karakteristik operasional historis.")
st.markdown("---")

# 2. Load Model & Fitur Artefak
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("lgbm_flight_delay_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

try:
    model, expected_features = load_model_artifacts()
    st.sidebar.success("✅ Model & Fitur Berhasil Dimuat")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat komponen model: {e}")
    st.stop()

# ------------------------------------------------------------------
# DATALANDING / KUMPULAN DATA HISTORIS DARI FEATURE ENGINEERING ANDA
# ------------------------------------------------------------------
PEAK_HOURS = [7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19] 
BUSY_MONTHS = [7, 8, 10] 

# Daftar Bandara yang masuk kategori TOP 10% (Hub) berdasarkan data training Anda
HUB_ORIGINS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX'] 
HUB_DESTS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX']

# Mock dictionary untuk congestion index
CONGESTION_LOOKUP = {
    ('JFK', 12): 45, ('LAX', 8): 60, ('ORD', 17): 75, ('ATL', 9): 90
}

# Opsi Pilihan Dropdown di UI (Kota sudah disesuaikan berurutan dengan daftar bandaranya)
CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'HA', 'EV', 'OO']
AIRPORT_OPTIONS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX', 'MIA', 'MCO']
STATE_OPTIONS = ['Georgia', 'Illinois', 'Texas', 'Colorado', 'California', 'New York', 'North Carolina', 'Nevada', 'Arizona', 'Florida']

CITY_OPTIONS = [
    'Atlanta, GA', 
    'Chicago, IL', 
    'Dallas/Fort Worth, TX', 
    'Denver, CO', 
    'Los Angeles, CA', 
    'San Francisco, CA', 
    'New York, NY', 
    'Charlotte, NC', 
    'Las Vegas, NV', 
    'Phoenix, AZ', 
    'Miami, FL', 
    'Orlando, FL'
]

# 3. Form Input Pengguna
st.subheader("📊 Masukkan Informasi Penerbangan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Jadwal & Waktu")
    month = st.slider("Bulan (Month)", min_value=1, max_value=12, value=6)
    day_of_month = st.slider("Tanggal (Day of Month)", min_value=1, max_value=31, value=15)
    day_of_week = st.slider("Hari dalam Seminggu (1=Senin, 7=Minggu)", min_value=1, max_value=7, value=3)
    
    crs_dep_time = st.number_input("Waktu Keberangkatan Asli (Format HHMM, misal: 1530)", min_value=0, max_value=2359, value=1200, step=5)
    
    st.markdown("### 🗺️ Jarak & Durasi")
    distance = st.number_input("Jarak Penerbangan (Distance in Miles)", min_value=10, max_value=10000, value=500)
    crs_elapsed_time = st.number_input("Durasi Terjadwal (CRS Elapsed Time in Minutes)", min_value=10, max_value=1000, value=120)

with col2:
    st.markdown("### ✈️ Maskapai & Lokasi")
    op_unique_carrier = st.selectbox("Maskapai (Carrier)", CARRIER_OPTIONS)
    
    # Bandara dan Kota Asal
    origin = st.selectbox("Bandara Asal (Origin)", AIRPORT_OPTIONS, index=4)       # Default LAX
    origin_city_name = st.selectbox("Kota Asal (Origin City Name)", CITY_OPTIONS, index=4) # Default Los Angeles, CA
    origin_state_nm = st.selectbox("Negara Bagian Asal (Origin State)", STATE_OPTIONS, index=4)
    
    st.markdown("---")
    # Bandara dan Kota Tujuan
    dest = st.selectbox("Bandara Tujuan (Destination)", AIRPORT_OPTIONS, index=6)    # Default JFK
    dest_city_name = st.selectbox("Kota Tujuan (Destination City Name)", CITY_OPTIONS, index=6) # Default New York, NY
    dest_state_nm = st.selectbox("Negara Bagian Tujuan (Destination State)", STATE_OPTIONS, index=5)

st.markdown("---")

# 4. Eksekusi Prediksi & Proses Otomatis Feature Engineering
if st.button("🔮 Hitung Analisis & Prediksi Delay", type="primary", use_container_width=True):
    
    # --- PROSES REKAYASA FITUR ---
    departure_hour = int(crs_dep_time // 100)
    route = f"{origin}-{dest}"
    is_peak = 1 if departure_hour in PEAK_HOURS else 0
    is_weekend = 1 if day_of_week in [6, 7] else 0
    
    airline_origin = f"{op_unique_carrier}-{origin}"
    airline_route = f"{op_unique_carrier}-{route}"
    
    speed = float(distance / crs_elapsed_time) if crs_elapsed_time > 0 else 0.0
    is_busy_month = 1 if month in BUSY_MONTHS else 0
    
    is_hub_origin = 1 if origin in HUB_ORIGINS else 0
    is_hub_dest = 1 if dest in HUB_DESTS else 0
    
    # dep_day_type
    if 0 <= departure_hour < 4:
        dep_day_type = 'Night'
    elif 4 <= departure_hour < 8:
        dep_day_type = 'Early_Morning'
    elif 8 <= departure_hour < 12:
        dep_day_type = 'Morning'
    elif 12 <= departure_hour < 16:
        dep_day_type = 'Midday'
    elif 16 <= departure_hour < 20:
        dep_day_type = 'Afternoon'
    else:
        dep_day_type = 'Evening'
        
    # duration_bin
    if 0 < crs_elapsed_time <= 60:
        duration_bin = 'short'
    elif 60 < crs_elapsed_time <= 120:
        duration_bin = 'medium'
    elif 120 < crs_elapsed_time <= 180:
        duration_bin = 'long'
    else:
        duration_bin = 'very_long'
        
    # day_bin
    if 0 < day_of_month <= 10:
        day_bin = 'early'
    elif 10 < day_of_month <= 20:
        day_bin = 'mid'
    else:
        day_bin = 'late'
        
    # congestion_index
    congestion_index = CONGESTION_LOOKUP.get((origin, departure_hour), 15)

    # --- MEMASUKKAN HASIL KE DALAM DATAFRAME ---
    raw_input_data = {
        'month': int(month),
        'day_of_month': int(day_of_month),
        'day_of_week': int(day_of_week),
        'op_unique_carrier': op_unique_carrier,
        'origin': origin,
        'origin_city_name': origin_city_name,
        'origin_state_nm': origin_state_nm,
        'dest': dest,
        'dest_city_name': dest_city_name,
        'dest_state_nm': dest_state_nm,
        'crs_elapsed_time': float(crs_elapsed_time),
        'distance': float(distance),
        'Departure_Hour': int(departure_hour),
        'Route': route,
        'is_peak': int(is_peak),
        'is_weekend': int(is_weekend),
        'airline_origin': airline_origin,
        'airline_route': airline_route,
        'speed': float(speed),
        'is_busy_month': int(is_busy_month),
        'is_hub_origin': int(is_hub_origin),
        'is_hub_dest': int(is_hub_dest),
        'dep_day_type': dep_day_type,
        'duration_bin': duration_bin,
        'day_bin': day_bin,
        'congestion_index': int(congestion_index)
    }
    
    df_input = pd.DataFrame([raw_input_data])
    
    # Mengunci tipe 'category' agar seragam dengan model LightGBM
    cat_cols = [
        'op_unique_carrier', 'origin', 'origin_city_name', 'origin_state_nm', 
        'dest', 'dest_city_name', 'dest_state_nm', 'Route', 
        'airline_origin', 'airline_route', 'dep_day_type', 'duration_bin', 'day_bin'
    ]
    for col in cat_cols:
        df_input[col] = df_input[col].astype('category')

    # Re-order kolom agar sama persis seperti training data
    df_input = df_input[expected_features]
    
    # 5. Prediksi
    try:
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        # 6. UI Output
        st.subheader("💡 Hasil Analisis Berdasarkan Rekayasa Fitur:")
        
        if prediction == 1:
            st.error(f"⚠️ **Penerbangan Diprediksi DELAY** (Probabilitas: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Penerbangan Diprediksi TEPAT WAKTU (ON TIME)** (Probabilitas: {prediction_proba[0]*100:.2f}%)")
            
        with st.expander("Lihat Parameter Hasil Ekstraksi Otomatis"):
            st.json({
                "Kota Asal": origin_city_name,
                "Kota Tujuan": dest_city_name,
                "Route Gabungan": route,
                "Departure Hour (Jam Ekstraksi)": departure_hour,
                "Apakah Jam Padat (is_peak)": "Ya" if is_peak == 1 else "Tidak",
                "Kecepatan Pesawat (speed)": f"{speed:.4f}",
                "Kategori Waktu Hari (dep_day_type)": dep_day_type,
                "Durasi Kelompok (duration_bin)": duration_bin,
                "Indeks Kepadatan Bandara (congestion_index)": congestion_index
            })
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi model: {e}")