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
            age = st.number_input("Age", min_value=1, max_value=120, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
            yellow_fingers = st.selectbox("Do you have yellow fingers?", ["Yes", "No"])
            anxiety = st.selectbox("Do you have anxiety?", ["Yes", "No"])
            peer_pressure = st.selectbox("Are you under peer pressure?", ["Yes", "No"])
            chronic_disease = st.selectbox("Do you have a chronic disease?", ["Yes", "No"])
            fatigue = st.selectbox("Do you experience fatigue?", ["Yes", "No"])
            allergy = st.selectbox("Do you have allergies?", ["Yes", "No"])
            wheezing = st.selectbox("Do you experience wheezing?", ["Yes", "No"])
            alcohol_consuming = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
            coughing = st.selectbox("Do you experience coughing?", ["Yes", "No"])
            shortness_of_breath = st.selectbox("Do you experience shortness of breath?", ["Yes", "No"])
            swallowing_difficulty = st.selectbox("Do you have difficulty swallowing?", ["Yes", "No"])
            chest_pain = st.selectbox("Do you experience chest pain?", ["Yes", "No"])

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
