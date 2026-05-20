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

# 2. Load Artefak Multi-Model (Model, Mapping Cluster, & Fitur Ekspektasi)
@st.cache_resource
def load_multi_model_artifacts():
    models = joblib.load("lgbm_cluster_models.pkl")         
    cluster_map = joblib.load("airport_cluster_mapping.pkl") 
    features = joblib.load("model_features.pkl")             
    return models, cluster_map, features

try:
    cluster_models, airport_cluster_mapping, expected_features = load_multi_model_artifacts()
    st.sidebar.success("✅ Semua Model Cluster & Pemetaan Berhasil Dimuat")
    
    with st.sidebar.expander("Lihat Fitur Ekspektasi Model"):
        st.write(expected_features)
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat komponen model: {e}")
    st.markdown("### ⚠️ Artefak Model Tidak Ditemukan")
    st.info("Pastikan file `lgbm_cluster_models.pkl`, `airport_cluster_mapping.pkl`, dan `model_features.pkl` berada di folder yang sama dengan file `app.py` ini.")
    st.stop()

# ------------------------------------------------------------------
# CONFIG DATA HISTORIS & OPSI (KATEGORI ASLI SAAT TRAINING)
# ------------------------------------------------------------------
BUSY_MONTHS = [7, 8, 10] 

# List master kategori untuk mengunci tipe data kategorikal (Wajib sinkron dengan training)
CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'HA', 'EV', 'OO']
DEP_DAY_TYPE_OPTIONS = ['Night', 'Early_Morning', 'Morning', 'Midday', 'Afternoon', 'Evening']

CONGESTION_LOOKUP = {
    ('JFK', 12): 45, ('LAX', 8): 60, ('ORD', 17): 75, ('ATL', 9): 90
}

AIRPORT_OPTIONS = list(airport_cluster_mapping.keys()) if airport_cluster_mapping else ['LAX', 'JFK', 'ORD', 'ATL']
STATE_OPTIONS = ['Georgia', 'Illinois', 'Texas', 'Colorado', 'California', 'New York', 'Florida', 'North Carolina', 'Nevada', 'Arizona']

CITY_OPTIONS = [
    'Atlanta, GA', 'Chicago, IL', 'Dallas/Fort Worth, TX', 'Denver, CO', 
    'Los Angeles, CA', 'San Francisco, CA', 'New York, NY', 'Charlotte, NC', 
    'Las Vegas, NV', 'Phoenix, AZ', 'Miami, FL', 'Orlando, FL'
]

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

# 4. Eksekusi Prediksi Multi-Model
if st.button("🔮 Hitung Analisis & Prediksi Delay", type="primary", use_container_width=True):
    
    # --- ROUTING LOGIC: DETEKSI CLUSTER BERDASARKAN BANDARA ASAL ---
    assigned_cluster = airport_cluster_mapping.get(origin, 0)
    model = cluster_models[assigned_cluster]
    
    # --- PROSES REKAYASA FITUR (FEATURE ENGINEERING ASLI) ---
    dep_hour = int(crs_dep_time // 100) 
    is_busy_month = 1 if month in BUSY_MONTHS else 0
    
    if 0 <= dep_hour < 4:
        dep_day_type = 'Night'
    elif 4 <= dep_hour < 8:
        dep_day_type = 'Early_Morning'
    elif 8 <= dep_hour < 12:
        dep_day_type = 'Morning'
    elif 12 <= dep_hour < 16:
        dep_day_type = 'Midday'
    elif 16 <= dep_hour < 20:
        dep_day_type = 'Afternoon'
    else:
        dep_day_type = 'Evening'
        
    congestion_index = CONGESTION_LOOKUP.get((origin, dep_hour), 15)

    # --- MEMASUKKAN VARIABEL AKTIF KE DALAM DATAFRAME ---
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
    
    # --- SOLUSI FIX: MENGUNCI KATEGORI MENGGUNAKAN CategoricalDtype ---
    categories_dict = {
        'op_unique_carrier': CARRIER_OPTIONS,
        'dep_day_type': DEP_DAY_TYPE_OPTIONS
    }
    
    for col, categories in categories_dict.items():
        if col in df_input.columns:
            cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
            df_input[col] = df_input[col].astype(cat_type)

    # --- DETEKSI OTOMATIS ANTISIPASI KEYERROR (SAFETY CHECK) ---
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
            
    # Menyusun ulang urutan kolom agar sinkron 100% dengan model cluster tujuan
    df_input = df_input[expected_features]
    
    # 5. Menghitung Prediksi Melalui Model Cluster Terpilih
    try:
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        # 6. Render Hasil Analisis UI Utama
        st.subheader("💡 Hasil Analisis Prediksi:")
        st.sidebar.info(f"📍 **Routing Status:** Bandara {origin} otomatis diproses menggunakan **Model Cluster {assigned_cluster}**.")

        if prediction == 1:
            st.error(f"⚠️ **Penerbangan Diprediksi DELAY** (Probabilitas Risiko Keterlambatan: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Penerbangan Diprediksi TEPAT WAKTU (ON TIME)** (Probabilitas On-Time: {prediction_proba[0]*100:.2f}%)")
            
        # --- DIAGNOSIS LOGIKA FAKTOR PENYEBAB ---
        st.markdown("### 🔍 Mengapa Hasilnya Demikian?")
        st.write("Berdasarkan kombinasi parameter rekayasa fitur yang masuk ke dalam sistem model cluster, berikut faktor pemicu utamanya:")

        reasons_delayed = []
        reasons_ontime = []

        if is_busy_month == 1:
            reasons_delayed.append(f"🔴 **Bulan Padat Operasional (`is_busy_month=1`):** Bulan {month} merupakan periode sibuk historis untuk cluster ini yang memicu risiko penumpukan jadwal.")
        else:
            reasons_ontime.append(f"🟢 **Bulan Normal (`is_busy_month=0`):** Bulan {month} berada dalam kapasitas operasional yang cenderung longgar.")

        if congestion_index > 40:
            reasons_delayed.append(f"🔴 **Trafik Bandara Sangat Padat (`congestion_index={congestion_index}`):** Bandara asal {origin} mengalami lonjakan keberangkatan pada jam {dep_hour}:00.")
        elif congestion_index <= 15:
            reasons_ontime.append(f"🟢 **Trafik Bandara Aman (`congestion_index={congestion_index}`):** Kepadatan penerbangan terjadwal di bandara asal tergolong rendah.")

        if prediction == 1:
            st.info("🔺 **Faktor yang Mendorong Prediksi DELAY:**")
            for item in reasons_delayed if reasons_delayed else ["• Karakteristik gabungan data makro rute, waktu, dan nomor penerbangan pada Cluster ini cenderung membentuk pola delay."]:
                st.write(item)
        else:
            st.info("🔹 **Faktor yang Mendorong Prediksi TEPAT WAKTU:**")
            for item in reasons_ontime if reasons_ontime else ["• Parameter alokasi waktu terbang, jarak, dan kondisi jam operasional terdeteksi aman oleh sistem."]:
                st.write(item)

        # Tab Rincian Data Teknis
        with st.expander("Lihat Rincian Probabilitas & Komponen Hasil Ekstraksi"):
            col_tab1, col_tab2 = st.columns(2)
            with col_tab1:
                st.json({
                    "Model Cluster Pemroses": int(assigned_cluster),
                    "Probabilitas Tepat Waktu (On-Time)": f"{prediction_proba[0]*100:.2f}%",
                    "Probabilitas Terlambat (Delay)": f"{prediction_proba[1]*100:.2f}%"
                })
            with col_tab2:
                st.json({
                    "Jam Keberangkatan (dep_hour)": dep_hour,
                    "Kategori Waktu (dep_day_type)": dep_day_type,
                    "Indeks Kemacetan Terhitung": congestion_index,
                    "Nomor Penerbangan Terbaca": op_carrier_fl_num
                })
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi model: {e}")