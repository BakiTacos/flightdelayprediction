import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide" # Menggunakan layout wide agar form input terlihat rapi berdampingan
)

st.title("✈️ Flight Delay Prediction App")
st.markdown("Prediksi status keterlambatan penerbangan berdasarkan data operasional maskapai dan bandara.")
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


# 3. Dummy Option untuk Dropdown Kategorikal 
# (Silakan lengkapi atau sesuaikan list di bawah dengan isi unique value pada dataset asli Anda)
CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK']
ORIGIN_OPTIONS = ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'SFO', 'CGK']
DEST_OPTIONS = ['LAX', 'JFK', 'ORD', 'ATL', 'DFW', 'SFO', 'DPS']
STATE_OPTIONS = ['California', 'Texas', 'New York', 'Georgia', 'Illinois']
BIN_OPTIONS = ['Short', 'Medium', 'Long']
DAY_BIN_OPTIONS = ['Morning', 'Afternoon', 'Evening', 'Night']
DAY_TYPE_OPTIONS = ['Weekday', 'Weekend', 'Holiday']


# 4. Form Input Pengguna (Dibagi menjadi 2 Kolom Besar)
st.subheader("📊 Isi Parameter Penerbangan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔢 Fitur Waktu & Numerik")
    
    # Fitur Numerik Utama (Gunakan slider atau number input)
    distance = st.number_input("Distance (Miles)", min_value=0, max_value=10000, value=500, step=10)
    crs_elapsed_time = st.number_input("Scheduled Elapsed Time (Minutes)", min_value=10, max_value=1000, value=120, step=5)
    speed = st.number_input("Estimated Speed", min_value=50.0, max_value=600.0, value=400.0, step=10.0)
    congestion_index = st.slider("Congestion Index (Kepadatan Bandara)", min_value=1, max_value=10, value=3)
    
    # Fitur Berbasis Angka/Waktu Pendek (Gunakan Slider agar presisi)
    month = st.slider("Bulan (Month)", min_value=1, max_value=12, value=6)
    day_of_month = st.slider("Tanggal (Day of Month)", min_value=1, max_value=31, value=15)
    day_of_week = st.slider("Hari dalam Seminggu (1=Senin, 7=Minggu)", min_value=1, max_value=7, value=3)
    departure_hour = st.slider("Jam Keberangkatan (Departure Hour)", min_value=0, max_value=23, value=12)

    # Fitur Binary / Flags (0 atau 1)
    st.markdown("**Flags & Indikator:**")
    is_peak = st.checkbox("Apakah Jam Padat (is_peak)?", value=False)
    is_weekend = st.checkbox("Apakah Akhir Pekan (is_weekend)?", value=False)
    is_busy_month = st.checkbox("Apakah Bulan Padat (is_busy_month)?", value=False)
    is_hub_origin = st.checkbox("Apakah Bandara Asal termasuk HUB?", value=False)
    is_hub_dest = st.checkbox("Apakah Bandara Tujuan termasuk HUB?", value=False)

with col2:
    st.markdown("### 🗂️ Fitur Kategorikal (Dropdown)")
    
    # Mengubah seluruh input kategorikal Anda menjadi Dropdown (st.selectbox)
    op_unique_carrier = st.selectbox("Airline Carrier (op_unique_carrier)", CARRIER_OPTIONS)
    origin = st.selectbox("Airport Origin (origin)", ORIGIN_OPTIONS)
    dest = st.selectbox("Airport Destination (dest)", DEST_OPTIONS)
    
    origin_city_name = st.text_input("Origin City Name", value="Los Angeles, CA")
    dest_city_name = st.text_input("Destination City Name", value="New York, NY")
    
    origin_state_nm = st.selectbox("Origin State (origin_state_nm)", STATE_OPTIONS)
    dest_state_nm = st.selectbox("Destination State (dest_state_nm)", STATE_OPTIONS)
    
    # Fitur Bins / Tipe Kategorikal Baru yang Anda buat
    dep_day_type = st.selectbox("Departure Day Type", DAY_TYPE_OPTIONS)
    duration_bin = st.selectbox("Duration Bin", BIN_OPTIONS)
    day_bin = st.selectbox("Day Bin", DAY_BIN_OPTIONS)
    
    # Fitur Kombinasi Rekayasa (Gunakan text input atau selectbox dummy jika diperlukan)
    st.markdown("*(Fitur kombinasi otomatis di-generate berdasarkan input di atas)*")
    route = f"{origin}-{dest}"
    airline_origin = f"{op_unique_carrier}-{origin}"
    airline_route = f"{op_unique_carrier}-{route}"

st.markdown("---")

# 5. Eksekusi Prediksi
if st.button("🔮 Prediksi Status Penerbangan", type="primary", use_container_width=True):
    
    # Bungkus semua input ke dalam bentuk Dict dengan Key yang sama persis seperti kolom dataframe Anda
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
    
    # Konversi ke DataFrame
    df_input = pd.DataFrame([raw_input_data])
    
    # Pastikan tipe data kolom kategorikal diubah kembali ke tipe 'category' 
    # agar dikenali dengan benar oleh LightGBM model yang Anda buat
    cat_cols = [
        'op_unique_carrier', 'origin', 'origin_city_name', 'origin_state_nm', 
        'dest', 'dest_city_name', 'dest_state_nm', 'Route', 
        'airline_origin', 'airline_route', 'dep_day_type', 'duration_bin', 'day_bin'
    ]
    for col in cat_cols:
        df_input[col] = df_input[col].astype('category')

    # Menjamin urutan kolom persis sama dengan yang diharapkan oleh model saat training
    df_input = df_input[expected_features]
    
    # Melakukan Prediksi
    try:
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        # 6. Menampilkan Hasil Output Ke User
        st.subheader("💡 Hasil Analisis Prediksi:")
        
        if prediction == 1:
            st.error(f"⚠️ **Penerbangan Diprediksi DELAY** (Probabilitas Risiko Keterlambatan: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Penerbangan Diprediksi TEPAT WAKTU (ON TIME)** (Probabilitas On-Time: {prediction_proba[0]*100:.2f}%)")
            
        with st.expander("Lihat Rincian Probabilitas Model"):
            st.json({
                "Probabilitas Tepat Waktu": f"{prediction_proba[0]*100:.2f}%",
                "Probabilitas Terlambat (Delay)": f"{prediction_proba[1]*100:.2f}%"
            })
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi model: {e}")