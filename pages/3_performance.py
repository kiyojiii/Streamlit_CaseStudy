import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    log_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

# Initialize session state variables
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'log_loss_value' not in st.session_state:
    st.session_state.log_loss_value = None
if 'conf_matrix' not in st.session_state:
    st.session_state.conf_matrix = None
if 'roc_auc_value' not in st.session_state:
    st.session_state.roc_auc_value = None
if 'report_df' not in st.session_state:
    st.session_state.report_df = None

# Load CSV File
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocessing function
def preprocess_data(data):
    data = data.copy()
    if 'GENDER' in data.columns:
        data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    if 'LUNG_CANCER' in data.columns:
        data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64'] and data[col].nunique() <= 2:
            data[col] = data[col].map({2: 1, 1: 0})
    return data

# Sidebar: File Upload
st.sidebar.title("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.title("Lung Cancer Prediction Tool")
    st.write("### Original Data")
    st.write(data.head())

    # Preprocess Data
    preprocessed_data = preprocess_data(data)

    st.write("### Preprocessed Data")
    st.write(preprocessed_data.head())

    # Sidebar: Classifier Selection
    model_type = st.sidebar.selectbox(
        'Select Classifier',
        ('Decision Tree', 'Gaussian Naive Bayes', 'AdaBoost Classifier', 
         'K-Nearest Neighbors Classifier', 'MLP Classifier', 
         'Perceptron Classifier', 'Random Forest', 'Support Vector Machine (SVM)')
    )

    # Sidebar: Cross-Validation Selection
    cv_type = st.sidebar.selectbox(
        'Select Cross-Validation Method',
        ('Train/Test Split', 'K-Fold Cross-Validation', 'Leave-One-Out Cross-Validation (LOOCV)')
    )

    # Sidebar: Performance Metric Selection
    metric_to_display = st.sidebar.selectbox(
        'Select Performance Metric to Display',
        ('Classification Accuracy', 'Log Loss', 'Confusion Matrix', 'Classification Report', 'ROC AUC')
    )

    # Sidebar: K-Fold Setting
    if cv_type == 'K-Fold Cross-Validation':
        k_folds = st.sidebar.slider('Select number of folds for K-Fold', min_value=2, max_value=10, value=5)

    # Train the model button
    if st.sidebar.button('Train Model'):
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
            proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            st.session_state.model_accuracy = accuracy_score(y_test, predictions)
            st.session_state.log_loss_value = log_loss(y_test, proba) if proba is not None else "Not Applicable"
            st.session_state.conf_matrix = confusion_matrix(y_test, predictions)
            st.session_state.report_df = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose()
            st.session_state.roc_auc_value = roc_auc_score(y_test, proba[:, 1]) if proba is not None else "Not Applicable"

        # K-Fold Cross-Validation
        elif cv_type == 'K-Fold Cross-Validation':
            kf = KFold(n_splits=k_folds)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            st.session_state.model_accuracy = cv_scores.mean()
            st.session_state.log_loss_value = "Not Applicable"
            st.session_state.conf_matrix = None
            st.session_state.report_df = None
            st.session_state.roc_auc_value = "Not Applicable"

        # LOOCV
        elif cv_type == 'Leave-One-Out Cross-Validation (LOOCV)':
            loo = LeaveOneOut()
            cv_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
            st.session_state.model_accuracy = cv_scores.mean()
            st.session_state.log_loss_value = "Not Applicable"
            st.session_state.conf_matrix = None
            st.session_state.report_df = None
            st.session_state.roc_auc_value = "Not Applicable"

        # Display Results
        st.write(f"### Selected Model: {model_type}")
        st.write(f"### Cross-Validation Method: {cv_type}")
        st.write(f"### Selected Metric: {metric_to_display}")

        if metric_to_display == 'Classification Accuracy':
            st.write(f"Classification Accuracy: {st.session_state.model_accuracy:.2f}")

        elif metric_to_display == 'Log Loss':
            st.write(f"Log Loss: {st.session_state.log_loss_value}")

        elif metric_to_display == 'Confusion Matrix':
            if st.session_state.conf_matrix is not None:
                st.write("Confusion Matrix:")
                fig, ax = plt.subplots()
                sns.heatmap(st.session_state.conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
            else:
                st.write("Confusion Matrix is not applicable for the selected cross-validation method.")

        elif metric_to_display == 'Classification Report':
            if st.session_state.report_df is not None:
                st.write("Classification Report:")
                st.table(st.session_state.report_df)
            else:
                st.write("Classification Report is not applicable for the selected cross-validation method.")

        elif metric_to_display == 'ROC AUC':
            if st.session_state.roc_auc_value != "Not Applicable":
                st.write(f"ROC AUC Score: {st.session_state.roc_auc_value:.2f}")
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label="ROC Curve")
                ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig)
            else:
                st.write("ROC AUC is not applicable for the selected cross-validation method.")
