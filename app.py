import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Multi-Cluster Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Multi-Cluster Flight Delay Prediction App")
st.markdown("Aplikasi prediksi keterlambatan penerbangan cerdas dengan **Dual Analysis (Kontekstual & Saintifik)**.")
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

# --- SINKRONISASI MUTLAK KATEGORI (SAMA PERSIS DENGAN NOTEBOOK) ---
MODEL_CARRIERS = ['9e', 'aa', 'as', 'b6', 'dl', 'f9', 'g4', 'ha', 'mq', 'nk', 'oh', 'oo', 'ua', 'wn', 'yx']
MODEL_DAY_TYPES = ['Night', 'Early_Morning', 'Morning', 'Midday', 'Afternoon', 'Evening']
UI_CARRIERS = [c.upper() for c in MODEL_CARRIERS]

BUSY_MONTHS = [7, 8, 10] 
CONGESTION_LOOKUP = {('JFK', 12): 45, ('LAX', 8): 60, ('ORD', 17): 75, ('ATL', 9): 90}
AIRPORT_OPTIONS = list(airport_cluster_mapping.keys()) if airport_cluster_mapping else ['LAX', 'JFK', 'ORD', 'ATL']
STATE_OPTIONS = ['Georgia', 'Illinois', 'Texas', 'Colorado', 'California', 'New York', 'Florida', 'North Carolina', 'Nevada', 'Arizona']
CITY_OPTIONS = ['Atlanta, GA', 'Chicago, IL', 'Dallas/Fort Worth, TX', 'Denver, CO', 'Los Angeles, CA', 'San Francisco, CA', 'New York, NY', 'Miami, FL', 'Orlando, FL']

# 3. Form Input Pengguna
st.subheader("📊 Masukkan Informasi Penerbangan")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Jadwal & Waktu")
    month = st.slider("Bulan (Month)", min_value=1, max_value=12, value=6)
    day_of_week = st.slider("Hari dalam Seminggu (1=Senin, 7=Minggu)", min_value=1, max_value=7, value=3)
    crs_dep_time = st.number_input("Waktu Keberangkatan Terjadwal (HHMM, misal: 1530)", min_value=0, max_value=2359, value=1200, step=5)
    op_carrier_fl_num = st.number_input("Nomor Penerbangan (Flight Number)", min_value=1, max_value=9999, value=1234)
    
    st.markdown("### 🗺️ Jarak & Durasi")
    distance = st.number_input("Jarak Penerbangan (Distance in Miles)", min_value=10, max_value=10000, value=500)
    crs_elapsed_time = st.number_input("Durasi Terjadwal (CRS Elapsed Time in Minutes)", min_value=10, max_value=1000, value=120)

with col2:
    st.markdown("### ✈️ Maskapai & Lokasi")
    op_unique_carrier_ui = st.selectbox("Maskapai (Carrier)", UI_CARRIERS)
    origin = st.selectbox("Bandara Asal (Origin)", AIRPORT_OPTIONS, index=0)       
    origin_city_name = st.selectbox("Kota Asal", CITY_OPTIONS, index=4) 
    origin_state_nm = st.selectbox("Negara Bagian Asal", STATE_OPTIONS, index=4)
    
    st.markdown("---")
    dest = st.selectbox("Bandara Tujuan (Destination)", AIRPORT_OPTIONS, index=1 if len(AIRPORT_OPTIONS) > 1 else 0)    
    dest_city_name = st.selectbox("Kota Tujuan", CITY_OPTIONS, index=6) 
    dest_state_nm = st.selectbox("Negara Bagian Tujuan", STATE_OPTIONS, index=5)

st.markdown("---")

# 4. Eksekusi Prediksi
if st.button("🔮 Hitung Analisis & Prediksi Delay", type="primary", use_container_width=True):
    assigned_cluster = airport_cluster_mapping.get(origin, 0)
    model = cluster_models[assigned_cluster]
    
    # Feature Engineering Otomatis
    dep_hour = int(crs_dep_time // 100) 
    is_busy_month = 1 if month in BUSY_MONTHS else 0
    
    if 0 <= dep_hour < 4: dep_day_type = 'Night'
    elif 4 <= dep_hour < 8: dep_day_type = 'Early_Morning'
    elif 8 <= dep_hour < 12: dep_day_type = 'Morning'
    elif 12 <= dep_hour < 16: dep_day_type = 'Midday'
    elif 16 <= dep_hour < 20: dep_day_type = 'Afternoon'
    else: dep_day_type = 'Evening'
        
    congestion_index = CONGESTION_LOOKUP.get((origin, dep_hour), 15)
    op_unique_carrier_model = op_unique_carrier_ui.lower()

    # Konstruksi data mentah
    raw_input_data = {
        'month': int(month),
        'day_of_week': int(day_of_week),
        'op_unique_carrier': op_unique_carrier_model,
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
    
    # MENGUNCI KATEGORI MENGGUNAKAN CategoricalDtype
    categories_dict = {
        'op_unique_carrier': MODEL_CARRIERS,
        'dep_day_type': MODEL_DAY_TYPES
    }
    for col, categories in categories_dict.items():
        if col in df_input.columns:
            cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
            df_input[col] = df_input[col].astype(cat_type)

    # Safety Check Kolom
    missing_cols = [col for col in expected_features if col not in raw_input_data.keys()]
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
            
    df_input = df_input[expected_features]
    
    # 5. Jalankan Prediksi dengan Analisis Dinamis
    try:
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        st.subheader("💡 Hasil Analisis Prediksi:")
        st.sidebar.info(f"📍 **Routing Status:** Bandara {origin} otomatis diproses menggunakan **Model Cluster {assigned_cluster}**.")

        # --- TAMPILAN STATUS PREDIKSI UTAMA ---
        if prediction == 1:
            st.error(f"⚠️ **Penerbangan Diprediksi DELAY** (Probabilitas Risiko Keterlambatan: {prediction_proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Penerbangan Diprediksi TEPAT WAKTU (ON TIME)** (Probabilitas On-Time: {prediction_proba[0]*100:.2f}%)")
            
        st.markdown("---")
        
        # Membuat Layout 2 Kolom untuk Hasil Analisis (Teks di Kiri, Grafik di Kanan)
        col_ans1, col_ans2 = st.columns([1, 1])
        
        with col_ans1:
            # --- LAPIS 1: DIAGNOSIS KONTEKSTUAL (IF-ELSE CERDAS) ---
            st.markdown("### 🔍 Deskripsi Faktor Penyebab")
            st.write("Berdasarkan silsilah parameter kombinasi input Anda, berikut adalah interpretasi operasionalnya:")
            
            reasons_delayed = []
            reasons_ontime = []

            # Logika Aturan Bulan Sibuk
            if is_busy_month == 1:
                reasons_delayed.append(f"🔴 **High Season Alert:** Bulan {month} secara historis merupakan puncak liburan/cuaca tertentu pada Cluster {assigned_cluster} yang rentan memicu delay massal.")
            else:
                reasons_ontime.append(f"🟢 **Kondisi Musim Stabil:** Bulan {month} berada pada grafik lalu lintas penerbangan tahunan yang normal.")

            # Logika Aturan Kemacetan Real-Time Bandara
            if congestion_index > 40:
                reasons_delayed.append(f"🔴 **Trafik Bandara Sangat Padat:** Slot keberangkatan jam {dep_hour}:00 di bandara {origin} terdeteksi memiliki kepadatan tinggi (Indeks Kepadatan: {congestion_index}).")
            elif congestion_index <= 20:
                reasons_ontime.append(f"🟢 **Trafik Bandara Aman:** Jadwal penerbangan di bandara asal {origin} tergolong lengang pada jam {dep_hour}:00.")
            else:
                reasons_ontime.append(f"🟡 **Trafik Bandara Wajar:** Sifat kepadatan lalu lintas udara berada pada batas kapasitas operasional aman.")

            # Logika Pola Jam Terbang (Malam hari/Pagi buta)
            if dep_day_type in ['Night', 'Evening'] and prediction == 1:
                reasons_delayed.append(f"⚠️ **Efek Domino Waktu Malam:** Keberangkatan pada fase `{dep_day_type}` sangat rentan terkena akumulasi keterlambatan pesawat dari jadwal penerbangan subuh/siang hari sebelumnya.")

            # Menampilkan interpretasi teks
            if prediction == 1:
                st.warning("🔺 **Indikator Utama Pemicu Risiko Delay:**")
                for item in reasons_delayed if reasons_delayed else ["• Pola pergerakan rute maskapai gabungan pada jam ini secara historis membentuk kecenderungan delay."]:
                    st.write(item)
            else:
                st.info("🔹 **Indikator Utama Pendukung Ketepatan Waktu:**")
                for item in reasons_ontime if reasons_ontime else ["• Parameter alokasi waktu terbang dan kesiapan armada terpantau berada di zona aman model."]:
                    st.write(item)

        with col_ans2:
            # --- LAPIS 2: DYNAMIC FEATURE IMPORTANCE (SAINTIFIK) ---
            st.markdown("### 📊 Bobot Pengaruh Fitur Internal")
            
            # Mengambil skor kepentingan fitur asli langsung dari model .pkl aktif
            importances = model.feature_importances_
            df_importance = pd.DataFrame({
                'Indikator': expected_features,
                'Skor (Gain)': importances
            }).sort_values(by='Skor (Gain)', ascending=False)
            
            # Normalisasi ke 100%
            total_gain = df_importance['Skor (Gain)'].sum()
            df_importance['Bobot Pengaruh (%)'] = (df_importance['Skor (Gain)'] / total_gain * 100) if total_gain > 0 else 0.0

            # Render grafik batang interaktif Streamlit
            st.bar_chart(
                df_importance.head(6), 
                x='Indikator', 
                y='Bobot Pengaruh (%)', 
                horizontal=True,
                color='#2ca02c' if prediction == 0 else '#d62728'
            )
            
            top_3 = df_importance['Indikator'].head(3).tolist()
            st.caption(f"💡 *Tiga fitur yang paling mendikte struktur keputusan matematika LightGBM pada pengujian ini secara berurutan adalah: **{', '.join(top_3)}**.*")

        # Tab Rincian Data Teknis untuk Transparansi Nilai
        with st.expander("⚙️ Rincian Nilai Probabilitas Matematika"):
            col_tab1, col_tab2 = st.columns(2)
            with col_tab1:
                st.json({
                    "ID Model Cluster Terpilih": int(assigned_cluster),
                    "Probabilitas Tepat Waktu (On-Time)": f"{prediction_proba[0]*100:.2f}%",
                    "Probabilitas Terlambat (Delay)": f"{prediction_proba[1]*100:.2f}%"
                })
            with col_tab2:
                st.json({
                    "Jam Keberangkatan": dep_hour,
                    "Kategori Waktu Hari": dep_day_type,
                    "Indeks Kemacetan Bandara": congestion_index,
                    "Maskapai Terbaca Model": op_unique_carrier_model
                })

    except Exception as e:
        st.error("⚠️ **Terjadi Kesalahan Klasifikasi Model!**")
        st.code(str(e), language="text")
        
        st.markdown("### 🔍 LOG DEBUGGER: Analisis Mismatch Tipe Data")
        debug_data = []
        for col in expected_features:
            is_missing = "❌ Terlewat (Diisi 0)" if col in missing_cols else "✅ Tersedia"
            col_dtype = str(df_input[col].dtype)
            issue = "Kemungkinan ini tipe Category saat training!" if col in missing_cols and ("cat" in col_dtype or "int" in col_dtype) else ""
            debug_data.append({"Nama Fitur": col, "Status Input": is_missing, "Tipe Data Streamlit": col_dtype, "Analisis": issue})
            
        st.dataframe(pd.DataFrame(debug_data), use_container_width=True)