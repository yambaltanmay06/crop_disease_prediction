import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import os

# Load Processed Data & Label Encoders
with open("pkl\\processed_data.pkl", "rb") as file:
    processed_data = pickle.load(file)

with open("pkl\\label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

with open("pkl\\categorical_features.pkl", "rb") as file:
    categorical_features = pickle.load(file)

# Load Best Trained Model
with open("pkl\\best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Identify relevant categorical input features
selected_features = processed_data.drop(columns=["disease_type"]).columns.tolist()
categorical_input_features = [col for col in categorical_features if col in selected_features]

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

disease_image_mapping = {
    " diaporthe-stem-canker": r"images/Diaporthe Stem Canker.jpg",
    " charcoal-rot": r"images/charcol rot.jpg",
    " rhizoctonia-root-rot": r"images/rhizoctonia root rot.jpg",
    " phytophthora-rot": r"images/phytophthora root.jpg",
    " brown-stem-rot": r"images/Brown Stem Rot.jpg",
    " powdery-mildew": r"images/Powdery Mildew.jpg",
    " downy-mildew": r"images/Downy Mildew.jpg",
    " brown-spot": r"images/Brown Spot.jpg",
    " bacterial-blight": r"images/Bacterial Blight.jpg",
    " bacterial-pustule": r"images/Bacterial Pustule.jpg",
    " purple-seed-stain": r"images/Purple Seed Stain.jpg",
    " anthracnose": r"images/Anthracnose.jpg",
    " phyllosticta-leaf-spot": r"images/Phyllosticta Leaf Spot.jpg",
    " alternarialeaf-spot": r"images/Alternaria Leaf Spot.jpg",
    " frog-eye-leaf-spot": r"images/Frog Eye Leaf Spot.jpg",
    " diaporthe-pod-&-stem-blight": r"images/Diaporthe Pod & Stem Blight.jpg",
    " cyst-nematode": r"images/Cyst Nematode.jpg",
    " 2-4-d-injury": r"images/2,4-D Injury.jpg",
    " herbicide-injury": r"images/Herbicide Injury.jpg"
}

# Streamlit UI
st.set_page_config(page_title="Soybean Disease Prediction", layout="wide")
st.title("🌾 Soybean Disease Prediction")
st.write("Enter the input features to predict the disease.")

# Layout for Input & Output
col1, col2 = st.columns([2, 1])

# Left Side - Input Fields
with col1:
    st.subheader("🔍 Input Features")
    user_input = {}
    colA, colB = st.columns(2)
    
    for idx, feature in enumerate(categorical_input_features):
        if feature in label_encoders and feature in readable_value_mappings:
            options = label_encoders[feature].classes_
            options = [opt for opt in options if opt.strip() != "?"]
            readable_options = [readable_value_mappings[feature].get(opt, opt) for opt in options]
            
            # Distribute input fields in two columns
            selected_value = (colA if idx % 2 == 0 else colB).selectbox(
                f"{feature.replace('_', ' ').capitalize()}", readable_options
            )
            
            encoded_value = next(k for k, v in readable_value_mappings[feature].items() if v == selected_value)
            encoded_value = label_encoders[feature].transform([encoded_value])[0]
            user_input[feature] = encoded_value

# Convert input to DataFrame
user_df = pd.DataFrame([user_input]).reindex(columns=selected_features, fill_value=0)

# Right Side - Prediction Output
with col2:
    st.subheader("🦠 Prediction Result")
    if st.button("🚀 Predict Disease"):
        prediction_encoded = model.predict(user_df.values)[0]
        predicted_disease = label_encoders["disease_type"].inverse_transform([prediction_encoded])[0]
        
        st.success(f"🛑 Predicted Disease: **{predicted_disease}**")
        
        # Display Image
        image_path = disease_image_mapping.get(predicted_disease, None)
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=predicted_disease, width=600)  # Removed unsupported argument
        else:
            st.warning("⚠️ Image not found for this disease.")
