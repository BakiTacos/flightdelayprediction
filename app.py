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

HUB_ORIGINS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX'] 
HUB_DESTS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX']

CONGESTION_LOOKUP = {
    ('JFK', 12): 45, ('LAX', 8): 60, ('ORD', 17): 75, ('ATL', 9): 90
}

CARRIER_OPTIONS = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'HA', 'EV', 'OO']
AIRPORT_OPTIONS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'JFK', 'CLT', 'LAS', 'PHX', 'MIA', 'MCO']
STATE_OPTIONS = ['Georgia', 'Illinois', 'Texas', 'Colorado', 'California', 'New York', 'North Carolina', 'Nevada', 'Arizona', 'Florida']

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
    day_of_month = st.slider("Tanggal (Day of Month)", min_value=1, max_value=31, value=15)
    day_of_week = st.slider("Hari dalam Seminggu (1=Senin, 7=Minggu)", min_value=1, max_value=7, value=3)
    crs_dep_time = st.number_input("Waktu Keberangkatan Asli (Format HHMM, misal: 1530)", min_value=0, max_value=2359, value=1200, step=5)
    
    st.markdown("### 🗺️ Jarak & Durasi")
    distance = st.number_input("Jarak Penerbangan (Distance in Miles)", min_value=10, max_value=10000, value=500)
    crs_elapsed_time = st.number_input("Durasi Terjadwal (CRS Elapsed Time in Minutes)", min_value=10, max_value=1000, value=120)

with col2:
    st.markdown("### ✈️ Maskapai & Lokasi")
    op_unique_carrier = st.selectbox("Maskapai (Carrier)", CARRIER_OPTIONS)
    
    origin = st.selectbox("Bandara Asal (Origin)", AIRPORT_OPTIONS, index=4)       
    origin_city_name = st.selectbox("Kota Asal (Origin City Name)", CITY_OPTIONS, index=4) 
    origin_state_nm = st.selectbox("Negara Bagian Asal (Origin State)", STATE_OPTIONS, index=4)
    
    st.markdown("---")
    dest = st.selectbox("Bandara Tujuan (Destination)", AIRPORT_OPTIONS, index=6)    
    dest_city_name = st.selectbox("Kota Tujuan (Destination City Name)", CITY_OPTIONS, index=6) 
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
    
    cat_cols = [
        'op_unique_carrier', 'origin', 'origin_city_name', 'origin_state_nm', 
        'dest', 'dest_city_name', 'dest_state_nm', 'Route', 
        'airline_origin', 'airline_route', 'dep_day_type', 'duration_bin', 'day_bin'
    ]
    for col in cat_cols:
        df_input[col] = df_input[col].astype('category')

    df_input = df_input[expected_features]
    
    # 5. Prediksi
    try:
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        # 6. TAMPILAN OUTPUT UTAMA
        st.subheader("💡 Hasil Analisis Prediksi:")
        
        if prediction == 1:
            st.error(f"⚠️ **Penerbangan Diprediksi DELAY** (Probabilitas Risiko Keterlambatan: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Penerbangan Diprediksi TEPAT WAKTU (ON TIME)** (Probabilitas On-Time: {prediction_proba[0]*100:.2f}%)")
            
        # ------------------------------------------------------------------
        # 7. BLOK BARU: DIAGNOSIS / PENYEBAB HASIL PREDIKSI
        # ------------------------------------------------------------------
        st.markdown("### 🔍 Mengapa Hasilnya Demikian?")
        st.write("Berdasarkan kombinasi parameter rekayasa fitur yang Anda masukkan, berikut adalah faktor pemicu utama yang memengaruhi keputusan model:")

        # Mengumpulkan poin-poin alasan
        reasons_delayed = []
        reasons_ontime = []

        # Aturan Logika Waktu (is_peak & dep_day_type)
        if is_peak == 1:
            reasons_delayed.append(f"🔴 **Jam Padat (`is_peak=1`):** Jam keberangkatan ({departure_hour}:00) berada pada waktu puncak lalu lintas penerbangan historis yang rawan memicu antrean.")
        else:
            reasons_ontime.append(f"🟢 **Jam Longgar (`is_peak=0`):** Jam keberangkatan ({departure_hour}:00) memiliki volume lalu lintas yang cenderung lebih sepi secara historis.")

        # Aturan Logika Bulan (is_busy_month)
        if is_busy_month == 1:
            reasons_delayed.append(f"🔴 **Musim Padat (`is_busy_month=1`):** Bulan {month} merupakan periode *high season* (liburan/cuaca tertentu) yang secara historis memiliki tingkat keterlambatan tinggi.")
        else:
            reasons_ontime.append(f"🟢 **Musim Normal (`is_busy_month=0`):** Bulan {month} dikategorikan memiliki frekuensi penerbangan yang stabil/normal.")

        # Aturan Logika Kepadatan Lalu Lintas Nyata (congestion_index)
        if congestion_index > 40:
            reasons_delayed.append(f"🔴 **Trafik Bandara Tinggi (`congestion_index={congestion_index}`):** Bandara asal {origin} diprediksi sangat padat jadwal pada jam {departure_hour}:00.")
        elif congestion_index < 20:
            reasons_ontime.append(f"🟢 **Trafik Bandara Rendah (`congestion_index={congestion_index}`):** Bandara asal {origin} dalam kondisi lengang pada jam tersebut.")

        # Aturan Logika Bandara Besar (HUB)
        if is_hub_origin == 1:
            reasons_delayed.append(f"🔴 **Bandara Asal Terlalu Besar (`is_hub_origin=1`):** Bandara {origin} masuk dalam kategori 10% Bandara Terbesar (HUB). Bandara besar rentan mengalami *efek domino* keterlambatan jadwal.")
        if is_hub_dest == 1:
            reasons_delayed.append(f"🔴 **Bandara Tujuan Terlalu Besar (`is_hub_dest=1`):** Bandara tujuan {dest} termasuk HUB utama yang berisiko menahan antrean mendarat pesawat.")

        # Aturan Logika Kecepatan / Rasio Waktu Penerbangan (speed)
        if speed > 7.0:
            reasons_delayed.append(f"🔴 **Rasio Jadwal Ketat (`speed={speed:.2f}`):** Alokasi waktu terbang (`crs_elapsed_time`) terlalu sempit dibandingkan jarak tempuh, model mendeteksi adanya risiko tinggi kegagalan mengejar ketepatan waktu.")

        # Menampilkan Alasan Berdasarkan Hasil Prediksi Akhir
        if prediction == 1:
            st.info("🔺 **Faktor yang Mendorong Prediksi DELAY:**")
            if reasons_delayed:
                for item in reasons_delayed:
                    st.write(item)
            else:
                st.write("• Karakteristik gabungan koordinat rute, maskapai, dan waktu operasional secara simultan membentuk kecenderungan delay.")
        else:
            st.info("🔹 **Faktor yang Mendorong Prediksi TEPAT WAKTU:**")
            if reasons_ontime:
                for item in reasons_ontime:
                    st.write(item)
            else:
                st.write("• Parameter waktu, pemilihan bandara, dan kelonggaran jadwal penerbangan berada dalam batas aman historis model.")

        # ------------------------------------------------------------------
        # Tab Breakdown Teknis
        # ------------------------------------------------------------------
        with st.expander("Lihat Rincian Probabilitas & Parameter Ekstraksi Otomatis"):
            col_tab1, col_tab2 = st.columns(2)
            with col_tab1:
                st.json({
                    "Probabilitas Tepat Waktu": f"{prediction_proba[0]*100:.2f}%",
                    "Probabilitas Terlambat (Delay)": f"{prediction_proba[1]*100:.2f}%"
                })
            with col_tab2:
                st.json({
                    "Departure Hour": departure_hour,
                    "Route": route,
                    "Speed Ratio": f"{speed:.4f}",
                    "Congestion Index": congestion_index,
                    "Day Bin": day_bin,
                    "Duration Bin": duration_bin
                })
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi model: {e}")