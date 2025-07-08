import streamlit as st
import pandas as pd
import joblib

# Load data and models
meals = pd.read_csv("data/meals.csv")
ctr_model = joblib.load("models/ctr_model.pkl")
cvr_model = joblib.load("models/cvr_model.pkl")

# Category encoding should match training
category_mapping = {cat: idx for idx, cat in enumerate(meals["category"].unique())}
meals["category_encoded"] = meals["category"].map(category_mapping)

# Streamlit UI
st.title("📈 Meal Ad CTR & CVR Predictor")

selected_title = st.selectbox("Choose a meal", meals["title"])
meal = meals[meals["title"] == selected_title].iloc[0]

# Prepare features
features = pd.DataFrame({
    "price": [meal["price"]],
    "rating": [meal["rating"]],
    "category_encoded": [meal["category_encoded"]]
})

# Make predictions
ctr_prob = ctr_model.predict_proba(features)[0][1]
cvr_prob = cvr_model.predict_proba(features)[0][1]
expected_value = ctr_prob * cvr_prob * meal["price"]

# Display results
st.markdown(f"**CTR:** {ctr_prob:.2%}")
st.markdown(f"**CVR:** {cvr_prob:.2%}")
st.markdown(f"**Expected Value:** ${expected_value:.2f}")
