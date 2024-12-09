import streamlit as st
import pandas as pd
import pickle

# Load the uploaded model
def load_uploaded_model(uploaded_file):
    try:
        model = pickle.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main prediction logic
def main():
    st.title("Lung Cancer Prediction Tool")
    st.write("This tool uses a pre-trained model to predict the likelihood of lung cancer based on user inputs.")

    # File uploader for model
    st.subheader("Upload a Pre-Trained Model")
    uploaded_file = st.file_uploader("Choose a model file (must be .pkl)", type=["pkl"])

    if uploaded_file is not None:
        model = load_uploaded_model(uploaded_file)
        if model is not None:
            st.success("Model uploaded and loaded successfully!")

            # Input fields for user data
            st.subheader("Input Features")

            # Personal Information
            st.markdown("### Personal Information")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, step=1)

            # Leave col2 blank (no content added here)

            with col3:
                gender = st.selectbox("Gender", ["Male", "Female"])


            # Lifestyle Factors
            st.markdown("### Lifestyle Factors")
            col3, col4 = st.columns(2)
            with col3:
                smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True)
                alcohol_consuming = st.radio("Do you consume alcohol?", ["Yes", "No"], horizontal=True)
            with col4:
                peer_pressure = st.radio("Are you under peer pressure?", ["Yes", "No"], horizontal=True)
                allergy = st.radio("Do you have allergies?", ["Yes", "No"], horizontal=True)

            # Symptoms
            st.markdown("### Symptoms")
            col5, col6 = st.columns(2)
            with col5:
                anxiety = st.radio("Do you have anxiety?", ["Yes", "No"], horizontal=True)
                yellow_fingers = st.radio("Do you have yellow fingers?", ["Yes", "No"], horizontal=True)
                fatigue = st.radio("Do you experience fatigue?", ["Yes", "No"], horizontal=True)
                coughing = st.radio("Do you experience coughing?", ["Yes", "No"], horizontal=True)
                shortness_of_breath = st.radio("Do you experience shortness of breath?", ["Yes", "No"], horizontal=True)
            with col6:
                chronic_disease = st.radio("Do you have a chronic disease?", ["Yes", "No"], horizontal=True)
                wheezing = st.radio("Do you experience wheezing?", ["Yes", "No"], horizontal=True)
                swallowing_difficulty = st.radio("Do you have difficulty swallowing?", ["Yes", "No"], horizontal=True)
                chest_pain = st.radio("Do you experience chest pain?", ["Yes", "No"], horizontal=True)

            # Map inputs to model-compatible format
            input_data = pd.DataFrame({
                "AGE": [age],
                "GENDER": [1 if gender == "Male" else 0],
                "SMOKING": [1 if smoking == "Yes" else 0],
                "YELLOW_FINGERS": [1 if yellow_fingers == "Yes" else 0],
                "ANXIETY": [1 if anxiety == "Yes" else 0],
                "PEER_PRESSURE": [1 if peer_pressure == "Yes" else 0],
                "CHRONIC_DISEASE": [1 if chronic_disease == "Yes" else 0],
                "FATIGUE": [1 if fatigue == "Yes" else 0],
                "ALLERGY": [1 if allergy == "Yes" else 0],
                "WHEEZING": [1 if wheezing == "Yes" else 0],
                "ALCOHOL_CONSUMING": [1 if alcohol_consuming == "Yes" else 0],
                "COUGHING": [1 if coughing == "Yes" else 0],
                "SHORTNESS_OF_BREATH": [1 if shortness_of_breath == "Yes" else 0],
                "SWALLOWING_DIFFICULTY": [1 if swallowing_difficulty == "Yes" else 0],
                "CHEST_PAIN": [1 if chest_pain == "Yes" else 0],
            })

            # Prediction
            if st.button("Predict"):
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

                # Display results
                if prediction[0] == 1:
                    st.error("Prediction: High likelihood of lung cancer.")
                else:
                    st.success("Prediction: Low likelihood of lung cancer.")

                if probability is not None:
                    st.write(f"Prediction Confidence: {probability * 100:.2f}%")
        else:
            st.error("Failed to load the model. Please upload a valid .pkl file.")

if __name__ == "__main__":
    main()
