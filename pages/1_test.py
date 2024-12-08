import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

# Initialize session state variables
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'log_loss_value' not in st.session_state:
    st.session_state.log_loss_value = None
if 'report_df' not in st.session_state:
    st.session_state.report_df = None

# Save Model Section
save_model_path = "C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\Models"

# Function to load data (cached)
@st.cache_data
def load_data():
    return pd.read_csv('C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\csv\\survey lung cancer.csv')

# Function to forcefully reload data (no cache)
def force_reload_data():
    return pd.read_csv('C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\csv\\survey lung cancer.csv')

# Function to clean data (removing rows with blank values)
def clean_data(data):
    # Drop rows with blank or missing values across all columns
    data = data.dropna(subset=data.columns)
    return data

# Preprocessing function
def preprocess_data(data):
    # Map GENDER to 1 (M) and 0 (F)
    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    # Map 2 -> 1 (Yes) and 1 -> 0 (No) for relevant columns
    for col in data.columns:
        if col not in ['AGE', 'GENDER', 'LUNG_CANCER']:
            data[col] = data[col].map({2: 1, 1: 0})
    return data

# Load the data initially using cache
data = load_data()

# Display application title
st.title("Lung Cancer Prediction Tool")
st.write('Data Loaded Successfully!')

# Display file path
file_path = 'C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\csv\\survey lung cancer.csv'
st.write('Data Name:', file_path.split('\\')[-1])

# Sidebar for model selection
model_type = st.sidebar.selectbox(
    'Select Classifier',
    ('Decision Tree', 'Gaussian Naive Bayes', 'AdaBoost Classifier', 'K-Nearest Neighbors Classifier',
     'MLP Classifier', 'Perceptron Classifier', 'Random Forest', 'Support Vector Machine (SVM)')
)

# Sidebar for cross-validation selection
cv_type = st.sidebar.selectbox(
    'Select Cross-Validation Method',
    ('Train/Test Split', 'K-Fold Cross-Validation', 'Leave-One-Out Cross-Validation (LOOCV)')
)

# Number of folds for K-Fold (only applicable if K-Fold is selected)
if cv_type == 'K-Fold Cross-Validation':
    k_folds = st.sidebar.slider('Select number of folds for K-Fold', min_value=2, max_value=10, value=5)

# Use session state to manage datasets
if 'original_data' not in st.session_state:
    st.session_state.original_data = load_data()  # Load original dataset

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.original_data.copy()

# Sidebar layout for Refresh CSV File and Clean Data buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button('Refresh CSV File'):
        st.session_state.original_data = force_reload_data()  # Reload original data
        st.session_state.cleaned_data = st.session_state.original_data.copy()
        st.session_state.cleaned_data_display = False  # Reset cleaned data display state
        st.sidebar.success('CSV file refreshed successfully!')

with col2:
    if st.button('Clean CSV File'):
        st.session_state.cleaned_data = clean_data(st.session_state.original_data.copy())
        st.session_state.cleaned_data_display = True  # Enable cleaned data display
        st.sidebar.success('Data cleaned successfully!')

# Display original or refreshed data preview
st.subheader('Original or Refreshed Data Preview')
st.write(st.session_state.original_data.head())

# Display cleaned data preview only if cleaned data was processed
if 'cleaned_data_display' in st.session_state and st.session_state.cleaned_data_display:
    st.subheader('Cleaned Data Preview')
    st.write(st.session_state.cleaned_data.head())

# Preprocessed data
st.subheader('Preprocessed Data Preview')
preprocessed_data = preprocess_data(st.session_state.cleaned_data.copy())
st.write(preprocessed_data.head())

# Sidebar layout for Train Model and Save Model buttons
col3, col4 = st.sidebar.columns(2)

with col3:
    if st.button('Train Model'):
        preprocessed_data = preprocessed_data.dropna()  # Ensure no NaN values
        X = preprocessed_data.drop('LUNG_CANCER', axis=1)
        y = preprocessed_data['LUNG_CANCER']

        # Initialize the chosen model
        if model_type == 'Decision Tree':
            model = DecisionTreeClassifier()
        elif model_type == 'Gaussian Naive Bayes':
            model = GaussianNB()
        elif model_type == 'AdaBoost Classifier':
            model = AdaBoostClassifier()
        elif model_type == 'K-Nearest Neighbors Classifier':
            model = KNeighborsClassifier()
        elif model_type == 'MLP Classifier':
            model = MLPClassifier()
        elif model_type == 'Perceptron Classifier':
            model = Perceptron()
        elif model_type == 'Random Forest':
            model = RandomForestClassifier()
        elif model_type == 'Support Vector Machine (SVM)':
            model = SVC(probability=True)

        # Train/Test Split
        if cv_type == 'Train/Test Split':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            log_loss_value = log_loss(y_test, model.predict_proba(X_test)) if hasattr(model, "predict_proba") else "Not Applicable"
            report_df = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose()

        # K-Fold Cross-Validation
        elif cv_type == 'K-Fold Cross-Validation':
            kf = KFold(n_splits=k_folds)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            accuracy = cv_scores.mean()
            log_loss_value = "Not Applicable"
            report_df = None

        # LOOCV
        elif cv_type == 'Leave-One-Out Cross-Validation (LOOCV)':
            loo = LeaveOneOut()
            cv_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
            accuracy = cv_scores.mean()
            log_loss_value = "Not Applicable"
            report_df = None

        # Store the model and training results in session state
        st.session_state.trained_model = model
        st.session_state.model_accuracy = accuracy
        st.session_state.model_type = model_type
        st.session_state.log_loss_value = log_loss_value
        st.session_state.report_df = report_df

        st.sidebar.success(f"Model trained successfully using {cv_type}!")

# Display training results if available in session state
if st.session_state.trained_model:
    st.subheader('Model Training Results')
    st.write('Model:', st.session_state.model_type)
    st.write('Classification Accuracy:', st.session_state.model_accuracy * 100, '%')
    st.write('Log Loss:', st.session_state.log_loss_value)
    if st.session_state.report_df is not None:
        st.subheader('Classification Report')
        st.table(st.session_state.report_df)

with col4:
    if st.session_state.trained_model and st.button('Save Model'):
        os.makedirs(save_model_path, exist_ok=True)
        model_file = f"{st.session_state.model_type.replace(' ', '_')}_{round(st.session_state.model_accuracy * 100, 2)}.pkl"
        model_path = os.path.join(save_model_path, model_file)
        with open(model_path, 'wb') as f:
            pickle.dump(st.session_state.trained_model, f)
        st.sidebar.success(f"Model saved as `{model_file}`!")
