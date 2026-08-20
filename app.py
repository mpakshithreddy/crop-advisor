"""
Crop Advisor - Web App (Streamlit)
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
 
# ---------- Page setup ----------
st.set_page_config(page_title="Crop Advisor", page_icon="🌱", layout="centered")
 
st.title("🌱 Crop Advisor")
st.caption("V1 Prototype — recommends a crop based on soil and climate conditions")
 
# ---------- Load model + data (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    return joblib.load('crop_model.pkl')
 
@st.cache_data
def load_data():
    return pd.read_csv('Crop_recommendation.csv')
 
model = load_model()
df = load_data()
 
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
 
# ---------- Input form ----------
st.subheader("Enter your field's conditions")
 
col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
 
confidence_threshold = st.slider("Confidence threshold for a 'solid' recommendation", 0, 100, 50) / 100
 
# ---------- Explanation logic (same as before) ----------
def explain_crop(field_input, crop):
    crop_avg = df[df['label'] == crop][FEATURES].mean()
    rows = []
    for col in FEATURES:
        your_val = field_input[col].values[0]
        typical = crop_avg[col]
        tolerance = 0.15 * typical if typical != 0 else 1
        if abs(your_val - typical) <= tolerance:
            status = "✅ Matches"
        elif your_val > typical:
            status = f"⚠️ Higher than typical ({typical:.1f})"
        else:
            status = f"⚠️ Lower than typical ({typical:.1f})"
        rows.append({"Factor": col, "Your value": round(your_val, 1), "Status": status})
    return pd.DataFrame(rows)
 
# ---------- Predict button ----------
if st.button("Get Recommendation", type="primary"):
    field_input = pd.DataFrame([{
        'N': N, 'P': P, 'K': K, 'temperature': temperature,
        'humidity': humidity, 'ph': ph, 'rainfall': rainfall
    }])
 
    probs = model.predict_proba(field_input)[0]
    crop_names = model.classes_
    top3_idx = probs.argsort()[-3:][::-1]
    top_crop = crop_names[top3_idx[0]]
    top_confidence = probs[top3_idx[0]]
 
    st.divider()
 
    if top_confidence >= confidence_threshold:
        st.success(f"### Recommended: **{top_crop.title()}** ({top_confidence*100:.0f}% confidence)")
        st.write("This is a solid match for your field conditions.")
    else:
        st.warning(f"### No confident match (best guess only {top_confidence*100:.0f}% confident)")
        st.write("Your field conditions don't strongly match any single crop. Closest options below, for you to judge:")
 
    st.subheader("Top 3 candidates")
    for i in top3_idx:
        crop = crop_names[i]
        conf = probs[i]
        with st.expander(f"{crop.title()} — {conf*100:.0f}% confidence", expanded=(i == top3_idx[0])):
            st.dataframe(explain_crop(field_input, crop), hide_index=True, use_container_width=True)
 
st.divider()
st.caption("⚠️ Trained on a single reference dataset (India, ~2,200 samples). Not yet validated on real field data from your specific region.")