import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman
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
    models = joblib.load("lgbm_cluster_models.pkl")         # Mengurangi beban RAM dengan memuat dict model
    cluster_map = joblib.load("airport_cluster_mapping.pkl") # Pemetaan bandara -> ID Cluster
    features = joblib.load("model_features.pkl")             # Fitur yang diterima LightGBM
    return models, cluster_map, features

try:
    cluster_models, airport_cluster_mapping, expected_features = load_multi_model_artifacts()
    st.sidebar.success("✅ Semua Model Cluster & Pemetaan Berhasil Dimuat")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat komponen model: {e}")
    st.stop()

# ------------------------------------------------------------------
# CONFIG DATA HISTORIS UNTUK FEATURE ENGINEERING
# ------------------------------------------------------------------
PEAK_HOURS = [7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19] 
BUSY_MONTHS = [7, 8, 10] 

HUB_ORIGINS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX'] 
HUB_DESTS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX']

CONGESTION_LOOKUP = {
    ('JFK', 12): 45, ('LAX', 8): 60, ('ORD', 17): 75, ('ATL', 9): 90
}

CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'HA', 'EV', 'OO']
AIRPORT_OPTIONS = list(airport_cluster_mapping.keys()) if airport_cluster_mapping else ['LAX', 'JFK', 'ORD', 'ATL']
STATE_OPTIONS = ['Georgia', 'Illinois', 'Texas', 'Colorado', 'California', 'New York', 'Florida']

CITY_OPTIONS = [
    'Atlanta, GA', 'Chicago, IL', 'Dallas/Fort Worth, TX', 'Denver, CO', 
    'Los Angeles, CA', 'San Francisco, CA', 'New York, NY', 'Miami, FL'
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
    
    origin = st.selectbox("Bandara Asal (Origin)", AIRPORT_OPTIONS, index=0)       
    origin_city_name = st.selectbox("Kota Asal (Origin City Name)", CITY_OPTIONS, index=4) 
    origin_state_nm = st.selectbox("Negara Bagian Asal (Origin State)", STATE_OPTIONS, index=4)
    
    st.markdown("---")
    dest = st.selectbox("Bandara Tujuan (Destination)", AIRPORT_OPTIONS, index=1 if len(AIRPORT_OPTIONS) > 1 else 0)    
    dest_city_name = st.selectbox("Kota Tujuan (Destination City Name)", CITY_OPTIONS, index=6) 
    dest_state_nm = st.selectbox("Negara Bagian Tujuan (Destination State)", STATE_OPTIONS, index=5)

st.markdown("---")

# 4. Eksekusi Prediksi Multi-Model
if st.button("🔮 Hitung Analisis & Prediksi Delay", type="primary", use_container_width=True):
    
    # --- DETEKSI CLUSTER BERDASARKAN BANDARA ASAL ---
    # Jika bandara baru tidak ada di mapping historis, arahkan ke cluster default (misal 0)
    assigned_cluster = airport_cluster_mapping.get(origin, 0)
    
    # Ambil model spesifik untuk cluster ini
    model = cluster_models[assigned_cluster]
    
    # --- PROSES REKAYASA FITUR (Hanya yang dipakai oleh Model setelah drop_cols) ---
    departure_hour = int(crs_dep_time // 100)
    is_peak = 1 if departure_hour in PEAK_HOURS else 0
    is_weekend = 1 if day_of_week in [6, 7] else 0
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
        
    congestion_index = CONGESTION_LOOKUP.get((origin, departure_hour), 15)

    # --- MEMASUKKAN FITUR AKTIF KE DALAM DATAFRAME ---
    raw_input_data = {
        'month': int(month),
        'day_of_month': int(day_of_month),
        'day_of_week': int(day_of_week),
        'op_unique_carrier': op_unique_carrier,
        'crs_elapsed_time': float(crs_elapsed_time),
        'distance': float(distance),
        'Departure_Hour': int(departure_hour),
        'is_peak': int(is_peak),
        'is_weekend': int(is_weekend),
        'speed': float(speed),
        'is_busy_month': int(is_busy_month),
        'is_hub_origin': int(is_hub_origin),
        'is_hub_dest': int(is_hub_dest),
        'dep_day_type': dep_day_type,
        'duration_bin': duration_bin,
        'day_bin': day_bin,
        'congestion_index': int(congestion_index)
        # origin, dest, Route, dkk otomatis tidak dimasukkan karena ada di daftar drop_cols Anda
    }
    
    df_input = pd.DataFrame([raw_input_data])
    
    # Set tipe kategorikal sisa fitur teks yang dikonsumsi model
    cat_cols = ['op_unique_carrier', 'dep_day_type', 'duration_bin', 'day_bin']
    for col in cat_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype('category')

    # Re-order kolom menyesuaikan expected_features untuk cluster model ini
    df_input = df_input[expected_features]
    
    # 5. Eksekusi Prediksi
    try:
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        # 6. Render Hasil Analisis UI
        st.subheader("💡 Hasil Analisis Prediksi:")
        
        # Informasi Info Router Cluster
        st.sidebar.info(f"📍 **Routing Status:** Bandara {origin} secara otomatis diarahkan & diproses menggunakan **Model Cluster {assigned_cluster}**.")

        if prediction == 1:
            st.error(f"⚠️ **Penerbangan Diprediksi DELAY** (Probabilitas Risiko Keterlambatan: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Penerbangan Diprediksi TEPAT WAKTU (ON TIME)** (Probabilitas On-Time: {prediction_proba[0]*100:.2f}%)")
            
        # Diagnosis Faktor Penyebab
        st.markdown("### 🔍 Mengapa Hasilnya Demikian?")
        reasons_delayed = []
        reasons_ontime = []

        if is_peak == 1:
            reasons_delayed.append(f"🔴 **Jam Kerja Padat (`is_peak=1`):** Waktu {departure_hour}:00 rentan antrean lepas landas pada Cluster {assigned_cluster}.")
        else:
            reasons_ontime.append(f"🟢 **Jam Longgar (`is_peak=0`):** Waktu keberangkatan relatif sepi jadwal.")

        if is_busy_month == 1:
            reasons_delayed.append(f"🔴 **High-Season Month:** Bulan {month} memiliki lonjakan volume penumpang historis.")
        
        if is_hub_origin == 1:
            reasons_delayed.append(f"🔴 **Origin Airport Hub:** Bandara asal merupakan salah satu gerbang HUB tersibuk nasional.")

        if prediction == 1:
            st.info("🔺 **Faktor Dominan Pemicu Delay:**")
            for item in reasons_delayed if reasons_delayed else ["• Karakteristik gabungan data makro cluster mengindikasikan delay."]:
                st.write(item)
        else:
            st.info("🔹 **Faktor Dominan Pendukung On-Time:**")
            for item in reasons_ontime if reasons_ontime else ["• Parameter operasional penerbangan berada di batas aman aman historis cluster."]:
                st.write(item)

        with st.expander("Lihat Rincian Probabilitas & Komponen Cluster"):
            col_tab1, col_tab2 = st.columns(2)
            with col_tab1:
                st.json({
                    "ID Cluster Pengolah": int(assigned_cluster),
                    "Probabilitas Tepat Waktu": f"{prediction_proba[0]*100:.2f}%",
                    "Probabilitas Terlambat": f"{prediction_proba[1]*100:.2f}%"
                })
            with col_tab2:
                st.json({
                    "Speed Ratio": f"{speed:.4f}",
                    "Congestion Index": congestion_index,
                    "Duration Bin": duration_bin,
                    "Day Bin": day_bin
                })
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi model: {e}")