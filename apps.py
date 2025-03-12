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

# Streamlit UI
st.set_page_config(page_title="Soybean Disease Prediction", layout="wide")
st.title("🌾 Soybean Disease Prediction")
st.write("Use the sidebar to input feature values and predict the disease.")

# Sidebar for User Inputs
st.sidebar.header("🔍 Input Features")
user_input = {}

# Categorical Inputs (Dropdown in Sidebar)
for feature in categorical_input_features:
    if feature in label_encoders:  # Ensure encoder exists
        options = label_encoders[feature].classes_
        options = [opt for opt in options if opt.strip() != "?"]  # Remove "?" from options
        if options:  # Ensure there are valid options
            selected_value = st.sidebar.selectbox(f"{feature.replace('_', ' ').capitalize()}", options)
            encoded_value = label_encoders[feature].transform([selected_value])[0]  # Encode selected value
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
