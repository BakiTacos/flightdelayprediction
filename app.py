import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="Flight Delay Prediction (Clustering + XGBoost)",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Delay Prediction (Clustering + XGBoost)")
st.markdown("Predict Flight Delay with **XGBoost**.")
st.markdown("---")

# 2. Load Multi-Model XGBoost Artifacts
@st.cache_resource
def load_multi_model_artifacts():
    # Points to your latest exported XGBoost pkl files
    models = joblib.load("xgb_cluster_models.pkl")         
    cluster_map = joblib.load("airport_cluster_mapping.pkl") 
    features = joblib.load("model_features.pkl")             
    return models, cluster_map, features

try:
    cluster_models, airport_cluster_mapping, expected_features = load_multi_model_artifacts()
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Failed to Load Model: {e}")
    st.stop()

# --- ABSOLUTE CATEGORY SYNCHRONIZATION (MATCHES NOTEBOOK EXACTLY) ---
MODEL_CARRIERS = ['9e', 'aa', 'as', 'b6', 'dl', 'f9', 'g4', 'ha', 'mq', 'nk', 'oh', 'oo', 'ua', 'wn', 'yx']
MODEL_DAY_TYPES = ['Night', 'Early_Morning', 'Morning', 'Midday', 'Afternoon', 'Evening']
UI_CARRIERS = [c.upper() for c in MODEL_CARRIERS]

BUSY_MONTHS = [7, 8, 10] 
CONGESTION_LOOKUP = {('JFK', 12): 45, ('LAX', 8): 60, ('ORD', 17): 75, ('ATL', 9): 90}
AIRPORT_OPTIONS = list(airport_cluster_mapping.keys()) if airport_cluster_mapping else ['LAX', 'JFK', 'ORD', 'ATL']
STATE_OPTIONS = ['Georgia', 'Illinois', 'Texas', 'Colorado', 'California', 'New York', 'Florida', 'North Carolina', 'Nevada', 'Arizona']
CITY_OPTIONS = ['Atlanta, GA', 'Chicago, IL', 'Dallas/Fort Worth, TX', 'Denver, CO', 'Los Angeles, CA', 'San Francisco, CA', 'New York, NY', 'Miami, FL', 'Orlando, FL']

# 3. User Input Form
st.subheader("📊 Insert Flight Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Date and Time")
    month = st.slider("Month", min_value=1, max_value=12, value=6)
    day_of_week = st.slider("Day of Week (1=Monday, 7=Sunday)", min_value=1, max_value=7, value=3)
    crs_dep_time = st.number_input("Departure Time (HHMM, Ex: 1530)", min_value=0, max_value=2359, value=1200, step=5)
    op_carrier_fl_num = st.number_input("Flight Number", min_value=1, max_value=9999, value=1234)
    
    st.markdown("### 🗺️ Distance & Duration")
    distance = st.number_input("Distance in Miles", min_value=10, max_value=10000, value=500)
    crs_elapsed_time = st.number_input("Scheduled Elapsed Time (Minutes)", min_value=10, max_value=1000, value=120)

with col2:
    st.markdown("### ✈️ Airline and Locations")
    op_unique_carrier_ui = st.selectbox("Airlines", UI_CARRIERS)
    origin = st.selectbox("Origin Airport", AIRPORT_OPTIONS, index=0)       
    origin_city_name = st.selectbox("Origin City", CITY_OPTIONS, index=4) 
    origin_state_nm = st.selectbox("Origin State", STATE_OPTIONS, index=4)
    
    st.markdown("---")
    dest = st.selectbox("Destination Airport", AIRPORT_OPTIONS, index=1 if len(AIRPORT_OPTIONS) > 1 else 0)    
    dest_city_name = st.selectbox("Destination City", CITY_OPTIONS, index=6) 
    dest_state_nm = st.selectbox("Destination State", STATE_OPTIONS, index=5)

st.markdown("---")

# 4. Prediction Execution
if st.button("🔮 Predict", type="primary", use_container_width=True):
    assigned_cluster = airport_cluster_mapping.get(origin, 0)
    model = cluster_models[assigned_cluster]
    
    # Automated Feature Engineering
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

    # Construct raw input data dictionary
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
    
    # LOCK CATEGORIES USING CategoricalDtype (Crucial to support enable_categorical=True in XGBoost)
    categories_dict = {
        'op_unique_carrier': MODEL_CARRIERS,
        'dep_day_type': MODEL_DAY_TYPES
    }
    for col, categories in categories_dict.items():
        if col in df_input.columns:
            cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
            df_input[col] = df_input[col].astype(cat_type)

    # Column Safety Check & Reordering to Match XGBoost's Training Features
    missing_cols = [col for col in expected_features if col not in raw_input_data.keys()]
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
            
    df_input = df_input[expected_features]
    
    # 5. Run Prediction with Custom Decision Threshold (0.40)
    try:
        # Get raw probabilities to enable custom threshold moving
        prediction_proba = model.predict_proba(df_input)[0]
        prob_delay = prediction_proba[1]
        prob_ontime = prediction_proba[0]
        
        # Class judgment based on your optimized custom research threshold (0.40)
        XGB_CUSTOM_THRESHOLD = 0.40
        prediction = 1 if prob_delay > XGB_CUSTOM_THRESHOLD else 0
        
        st.subheader("💡 Analysis of Predictions:")
        st.sidebar.info(f"📍 **Routing Status:** Airport {origin} automatically processed using **Model Cluster {assigned_cluster}**.")

        # --- MAIN PREDICTION STATUS DISPLAY ---
        if prediction == 1:
            st.error(f"⚠️ **Flight is predicted to be DELAYED** (Delay Probability: {prob_delay*100:.2f}%)")
            st.caption(f"ℹ️ *Based on Decision Threshold: {XGB_CUSTOM_THRESHOLD}*")
        else:
            st.success(f"✅ **Flight is predicted to be ON TIME** (On-Time Probability: {prob_ontime*100:.2f}%)")
            
        st.markdown("---")
        
        # 2-Column Layout for Analysis Results (Text on Left, Chart on Right)
        col_ans1, col_ans2 = st.columns([1, 1])
        
        with col_ans1:
            # --- LAYER 1: CONTEXTUAL DIAGNOSIS (SMART IF-ELSE) ---
            st.markdown("### 🔍 Description and Cause:")
            st.write("Based on the pattern of your combined input parameters, here is the operational breakdown:")
            
            reasons_delayed = []
            reasons_ontime = []

            # Busy Month Logic
            if is_busy_month == 1:
                reasons_delayed.append(f"🔴 **High Season Alert:** Month {month} historically represents a holiday peak or severe weather window for Cluster {assigned_cluster}, which is prone to triggering mass delays.")
            else:
                reasons_ontime.append(f"🟢 **Stable Season Conditions:** Month {month} sits within a normal, manageable annual flight traffic curve.")

            # Real-Time Airport Congestion Logic
            if congestion_index > 40:
                reasons_delayed.append(f"🔴 **Heavy Airport Traffic:** The {dep_hour}:00 departure slot at airport {origin} shows a high density level (Congestion Index: {congestion_index}).")
            elif congestion_index <= 20:
                reasons_ontime.append(f"🟢 **Clear Airport Traffic:** Flight schedules at origin airport {origin} are relatively light around {dep_hour}:00.")
            else:
                reasons_ontime.append(f"🟡 **Moderate Airport Traffic:** Air traffic density remains within standard safe operational capacities.")

            # Flight Time Windows Logic (Evening/Night Accumulation)
            if dep_day_type in ['Night', 'Evening'] and prediction == 1:
                reasons_delayed.append(f"⚠️ **Evening Knock-On Effect:** Departures scheduled during the `{dep_day_type}` phase are highly vulnerable to accumulated backlog delays from earlier morning/midday flights.")

            # Displaying text interpretations
            if prediction == 1:
                st.warning("🔺 **Primary Risk Indicators Triggering Delay:**")
                for item in reasons_delayed if reasons_delayed else ["• Combined airline route movement patterns at this specific hour historically yield a strong mathematical bias toward delays."]:
                    st.write(item)
            else:
                st.info("🔹 **Primary Indicators Supporting On-Time Performance:**")
                for item in reasons_ontime if reasons_ontime else ["• Flight time allocation and fleet readiness parameters stay safely within the model's low-risk zones."]:
                    st.write(item)

        with col_ans2:
            # --- LAYER 2: DYNAMIC FEATURE IMPORTANCE (SCIENTIFIC) ---
            st.markdown("### 📊 Internal Feature Weights")
            
            # Fetching raw gain importance scores directly from the active XGBoost .pkl model
            importances = model.feature_importances_
            df_importance = pd.DataFrame({
                'Indicator': expected_features,
                'Score (Gain)': importances
            }).sort_values(by='Score (Gain)', ascending=False)
            
            # Normalize to 100%
            total_gain = df_importance['Score (Gain)'].sum()
            df_importance['Influence Weight (%)'] = (df_importance['Score (Gain)'] / total_gain * 100) if total_gain > 0 else 0.0

            # Render interactive Streamlit bar chart
            st.bar_chart(
                df_importance.head(6), 
                x='Indicator', 
                y='Influence Weight (%)', 
                horizontal=True,
                color='#2ca02c' if prediction == 0 else '#d62728'
            )
            
            top_3 = df_importance['Indicator'].head(3).tolist()
            st.caption(f"💡 *The three features most heavily dictating the XGBoost decision structure for this particular runtime test are: **{', '.join(top_3)}**.*")

        # Technical Details Expander for Mathematical Transparency
        with st.expander("⚙️ Mathematical Probability Details"):
            col_tab1, col_tab2 = st.columns(2)
            with col_tab1:
                st.json({
                    "Selected Cluster Model ID": int(assigned_cluster),
                    "On-Time Probability": f"{prob_ontime*100:.2f}%",
                    "Delay Probability": f"{prob_delay*100:.2f}%",
                    "Active Decision Threshold": XGB_CUSTOM_THRESHOLD
                })
            with col_tab2:
                st.json({
                    "Departure Hour": dep_hour,
                    "Day Type Category": dep_day_type,
                    "Airport Congestion Index": congestion_index,
                    "Model-Parsed Airline String": op_unique_carrier_model
                })

    except Exception as e:
        st.error("⚠️ **An Error Occurred During XGBoost Classification!**")
        st.code(str(e), language="text")
        st.code(traceback.format_exc(), language="text") # Full traceback included to make debugging Dtype mismatch on cloud runtimes easier
        
        st.markdown("### 🔍 DEBUG LOG: Data Type Mismatch Analysis")
        debug_data = []
        for col in expected_features:
            is_missing = "❌ Omitted (Defaulted to 0)" if col in missing_cols else "✅ Available"
            col_dtype = str(df_input[col].dtype)
            issue = "XGBoost is highly sensitive to missing/empty categorical types!" if col in missing_cols and ("cat" in col_dtype or "int" in col_dtype) else ""
            debug_data.append({"Feature Name": col, "Input Status": is_missing, "Streamlit Dtype": col_dtype, "Analysis": issue})
            
        st.dataframe(pd.DataFrame(debug_data), use_container_width=True)