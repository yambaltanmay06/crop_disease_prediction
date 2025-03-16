import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load Processed Data & Label Encoders
with open("processed_data.pkl", "rb") as file:
    processed_data = pickle.load(file)

with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

with open("categorical_features.pkl", "rb") as file:
    categorical_features = pickle.load(file)

# Load Best Trained Model
try:
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠ Model file not found! Train and save the model first.")
    st.stop()

# Identify relevant categorical input features (excluding disease_type) 
selected_features = processed_data.drop(columns=["disease_type"]).columns.tolist()
categorical_input_features = [col for col in categorical_features if col in selected_features]

# Human-readable value mappings
readable_value_mappings = {
    "season": {'october': 'October', 'august': 'August', 'july': 'July', 'september': 'September',
                'may': 'May', 'april': 'April', 'june': 'June', '?': 'Unknown'},
    "plant_count": {' normal': 'Normal', ' lt-normal': 'Less than Normal', ' ?': 'Unknown'},
    "rainfall": {' gt-norm': 'Greater than Normal', ' lt-norm': 'Less than Normal', ' norm': 'Normal', ' ?': 'Unknown'},
    "temperature": {' norm': 'Normal', ' gt-norm': 'Greater than Normal', ' lt-norm': 'Less than Normal', ' ?': 'Unknown'},
    "hail_damage": {' yes': 'Yes', ' no': 'No', ' ?': 'Unknown'},
    "past_crops": {' same-lst-yr': 'Same as Last Year', ' same-lst-two-yrs': 'Same as Last Two Years',
                    ' same-lst-sev-yrs': 'Same as Last Seven Years', ' diff-lst-year': 'Different from Last Year', ' ?': 'Unknown'},
    "damaged_area": {' low-areas': 'Low Areas', ' scattered': 'Scattered', ' whole-field': 'Whole Field', ' upper-areas': 'Upper Areas', ' ?': 'Unknown'},
    "damage_level": {' pot-severe': 'Potentially Severe', ' severe': 'Severe', ' minor': 'Minor', ' ?': 'Unknown'},
    "seed_treatment": {' none': 'None', ' fungicide': 'Fungicide', ' other': 'Other', ' ?': 'Unknown'},
    "seed_germination": {' 90-100': '90-100%', ' 80-89': '80-89%', ' lt-80': 'Less than 80%', ' ?': 'Unknown'},
    "growth_stage": {' abnorm': 'Abnormal', ' norm': 'Normal', ' ?': 'Unknown'},
    "halo_spots": {' absent': 'Absent', ' no-yellow-halos': 'No Yellow Halos', ' yellow-halos': 'Yellow Halos', ' ?': 'Unknown'},
    "pod_health": {' norm': 'Normal', ' dna': 'DNA', ' diseased': 'Diseased', ' few-present': 'Few Present'},
    "pod_spots": {' dna': 'DNA', ' absent': 'Absent', ' colored': 'Colored', ' brown-w/blk-specks': 'Brown with Black Specks'},
    "stem_health": {' abnorm': 'Abnormal', ' norm': 'Normal', ' ?': 'Unknown'},
    "stem_sores": {' above-sec-nde': 'Above Second Node', ' absent': 'Absent', ' below-soil': 'Below Soil', ' above-soil': 'Above Soil'},
    "lesion_size": {' brown': 'Brown', ' dna': 'DNA', ' tan': 'Tan', ' dk-brown-blk': 'Dark Brown to Black'},
}

# Streamlit UI
st.set_page_config(page_title="Soybean Disease Prediction", layout="wide")
st.title("🌾 Soybean Disease Prediction")
st.write("Use the sidebar to input feature values and predict the disease.")

# Sidebar for User Inputs
st.sidebar.header("🔍 Input Features")
user_input = {}

# Categorical Inputs (Dropdown in Sidebar)
for feature in categorical_input_features:
    if feature in label_encoders and feature in readable_value_mappings:  # Ensure encoder and mapping exist
        options = label_encoders[feature].classes_
        options = [opt for opt in options if opt.strip() != "?"]  # Remove "?" from options
        readable_options = [readable_value_mappings[feature].get(opt, opt) for opt in options]
        selected_value = st.sidebar.selectbox(f"{feature.replace('_', ' ').capitalize()}", readable_options)
        # Reverse map human-readable selection back to encoded value
        encoded_value = next(k for k, v in readable_value_mappings[feature].items() if v == selected_value)
        encoded_value = label_encoders[feature].transform([encoded_value])[0]
        user_input[feature] = encoded_value

# Convert input to DataFrame
user_df = pd.DataFrame([user_input])

# Ensure user_df matches model's training features
missing_features = set(selected_features) - set(user_df.columns)  # Features in training but not in user_df

# Fix missing features
for feature in missing_features:
    user_df[feature] = 0  # Fill missing features with neutral values

user_df = user_df[selected_features]  # Ensure correct column order

# Output Section in Center
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.sidebar.button("🚀 Predict Disease"):
        prediction_encoded = model.predict(user_df)[0]
        predicted_disease = label_encoders["disease_type"].inverse_transform([prediction_encoded])[0]
        st.subheader("🦠 Prediction Result:")
        st.success(f"🛑 Predicted Disease: **{predicted_disease}**")
