import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("model.joblib")

st.set_page_config(page_title="ML Classifier", page_icon="🤖")
st.title("🤖 Machine Learning Classifier App")
st.write("Upload your dataset and enter feature values to get predictions!")

# Upload dataset manually
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # Detect features automatically (exclude 'target')
    if "target" in df.columns:
        feature_names = [col for col in df.columns if col != "target"]
    else:
        feature_names = df.columns.tolist()

    st.write("### ✍️ Enter your feature values:")
    inputs = {}

    for feature in feature_names:
        inputs[feature] = st.text_input(f"{feature}:")

    # Predict button
    if st.button("🔍 Predict"):
        try:
            # Convert inputs into dataframe
            input_df = pd.DataFrame([inputs])

            # Convert numeric values where possible
            for col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except:
                    pass

            # Make prediction
            prediction = model.predict(input_df)[0]
            st.success(f"🎯 Predicted class: **{prediction}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("👆 Please upload a CSV file to continue.")
