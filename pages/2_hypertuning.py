import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

# Load dataset
@st.cache_data
def load_data():
    file_path = 'C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\csv\\survey lung cancer.csv'
    return pd.read_csv(file_path)

# Clean data
def clean_data(data):
    return data.dropna()

# Preprocess data
def preprocess_data(data):
    mappings = {'M': 1, 'F': 0, 'YES': 1, 'NO': 0}
    data['GENDER'] = data['GENDER'].map(mappings)
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map(mappings)
    for col in data.columns:
        if col not in ['AGE', 'GENDER', 'LUNG_CANCER']:
            data[col] = data[col].map({2: 1, 1: 0})
    return data

# Get model and hyperparameter grid
def get_model_and_params(name):
    models_and_params = {
        "Decision Tree": (
            DecisionTreeClassifier(),
            {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10, None]}
        ),
        "Random Forest": (
            RandomForestClassifier(),
            {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10, None], 'criterion': ['gini', 'entropy']}
        ),
        "AdaBoost Classifier": (
            AdaBoostClassifier(),
            {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 1.0]}
        ),
        "K-Nearest Neighbors Classifier": (
            KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']}
        ),
        "Support Vector Machine (SVM)": (
            SVC(probability=True),
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        ),
        "MLP Classifier": (
            MLPClassifier(),
            {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'max_iter': [200, 500]}
        ),
        "Gaussian Naive Bayes": (
            GaussianNB(),
            {'var_smoothing': np.logspace(-9, -3, 7)}
        ),
        "Perceptron Classifier": (
            CalibratedClassifierCV(Perceptron(), method='sigmoid', cv=3),
            {'base_estimator__penalty': ['l2', 'l1', 'elasticnet'], 'base_estimator__alpha': [0.0001, 0.001, 0.01]}
        )
    }
    return models_and_params.get(name)

# Perform hyperparameter tuning
def perform_hyper_tuning(model, param_grid, X_train, y_train, method="grid", num_folds=3):
    if param_grid is None:
        raise ValueError("This algorithm does not support hyperparameter tuning.")

    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=num_folds, n_jobs=-1) if method == "grid" else \
             RandomizedSearchCV(model, param_grid, scoring='accuracy', cv=num_folds, n_jobs=-1, n_iter=20, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in (' ', '_') else "_" for c in name)

# Suggest the best model
def suggest_best_model(data, sampling_method, tuning_method, test_size=None, num_folds=3):
    models_and_params = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
        "Support Vector Machine (SVM)": SVC(probability=True),
        "MLP Classifier": MLPClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Perceptron Classifier": CalibratedClassifierCV(Perceptron(), method='sigmoid', cv=3)
    }

    # Preprocess the data
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    best_model_name = None
    best_accuracy = 0
    best_model = None

    for model_name, model in models_and_params.items():
        param_grid = get_model_and_params(model_name)[1]
        if sampling_method == "Train/Test Split":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            tuned_model, _, accuracy = perform_hyper_tuning(
                model, param_grid, X_train, y_train,
                method="grid" if tuning_method == "Grid Search" else "random"
            )
        elif sampling_method == "K-Fold Cross Validation":
            tuned_model, _, accuracy = perform_hyper_tuning(
                model, param_grid, X, y,
                method="grid" if tuning_method == "Grid Search" else "random",
                num_folds=num_folds
            )

        if accuracy > best_accuracy:
            best_model_name = model_name
            best_accuracy = accuracy
            best_model = tuned_model

    return best_model_name, best_model, best_accuracy

# Main hypertuning and download logic
def main():
    # Initialize session state variables
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'cv_type' not in st.session_state:
        st.session_state.cv_type = None
    if 'tuning_method' not in st.session_state:
        st.session_state.tuning_method = None
    if 'model_saved' not in st.session_state:
        st.session_state.model_saved = False

    st.title("Model Hyperparameter Tuning Tool")

    # Load and preprocess the data
    data = load_data()
    data = clean_data(data)
    data = preprocess_data(data)

    st.subheader("Data Preview")
    st.write(data.head())

    # Sidebar configuration
    st.sidebar.subheader("Hyperparameter Tuning Configuration")
    
    sampling_method = st.sidebar.selectbox(
        "Select Sampling Technique",
        ["Select", "Train/Test Split", "K-Fold Cross Validation"]
    )
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        [
            "Select", "Decision Tree", "Gaussian Naive Bayes", "AdaBoost Classifier",
            "K-Nearest Neighbors Classifier", "MLP Classifier", "Perceptron Classifier",
            "Random Forest", "Support Vector Machine (SVM)"
        ]
    )
    tuning_method = st.sidebar.selectbox(
        "Select Tuning Method",
        ["Grid Search", "Random Search"]
    )
    test_size = num_folds = None

    # Train/Test Split
    if sampling_method == "Train/Test Split":
        test_size = st.sidebar.slider("Test Size", 0.1, 0.9, 0.2, 0.01)

    # K-Fold Cross Validation
    elif sampling_method == "K-Fold Cross Validation":
        num_folds = st.sidebar.slider("Number of Folds", 2, 20, 5)

    # Perform hyperparameter tuning
    if st.sidebar.button("Suggest Best Model"):
        if sampling_method == "Select" or tuning_method == "Select":
            st.sidebar.error("Please select both a sampling technique and a tuning method.")
        else:
            with st.spinner("Finding the best model..."):
                best_model_name, best_model, best_accuracy = suggest_best_model(
                    data, sampling_method, tuning_method,
                    test_size=test_size, num_folds=num_folds
                )

            if best_model_name:
                st.session_state.trained_model = best_model
                st.session_state.model_accuracy = best_accuracy
                st.session_state.model_type = best_model_name
                st.session_state.cv_type = sampling_method
                st.session_state.tuning_method = tuning_method
                st.session_state.model_saved = False

                # Display the suggested best model
                st.subheader("Suggested Best Model")
                st.write(f"Best Model: {best_model_name}")
                st.write(f"Best Accuracy: {best_accuracy * 100:.2f}%")
                st.write(f"Cross-Validation Type: {sampling_method}")
                st.write(f"Tuning Method: {tuning_method}")
                
                
    # Perform hyperparameter tuning
    if st.sidebar.button("HyperTune"):
        if sampling_method == "Select" or algorithm == "Select":
            st.sidebar.error("Please select both a sampling technique and an algorithm.")
        else:
            model, param_grid = get_model_and_params(algorithm)
            if model is not None:  # Ensure model exists
                X = data.drop('LUNG_CANCER', axis=1)
                y = data['LUNG_CANCER']

                with st.spinner("Performing hyperparameter tuning..."):
                    if sampling_method == "Train/Test Split":
                        # Use Train/Test Split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        best_model, best_params, best_score = perform_hyper_tuning(
                            model, param_grid, X_train, y_train, method="grid" if tuning_method == "Grid Search" else "random"
                        )
                    elif sampling_method == "K-Fold Cross Validation":
                        # Use num_folds for K-Fold Cross Validation
                        best_model, best_params, best_score = perform_hyper_tuning(
                            model, param_grid, X, y, method="grid" if tuning_method == "Grid Search" else "random",
                            num_folds=num_folds
                        )

                # Store results in session state
                st.session_state.trained_model = best_model
                st.session_state.model_accuracy = best_score
                st.session_state.model_type = algorithm
                st.session_state.cv_type = sampling_method
                st.session_state.tuning_method = tuning_method
                st.session_state.model_saved = False

                # Display results
                st.subheader("Hyperparameter Tuning Results")
                st.write(f"Best Model: {algorithm}")
                st.write("Best Parameters:", best_params)
                st.write(f"Best Accuracy: {best_score * 100:.2f}%")
                st.write(f"Tuning Method: {tuning_method}")
                if sampling_method == "K-Fold Cross Validation":
                    st.write(f"Number of Folds: {num_folds}")

    # Display previous results if model has been trained
    if st.session_state.trained_model:
        st.subheader("Previous Model Training Results")
        st.write(f"Best Model: {st.session_state.model_type}")
        st.write(f"Best Accuracy: {st.session_state.model_accuracy * 100:.2f}%")
        st.write(f"Cross-Validation Type: {st.session_state.cv_type}")
        st.write(f"Tuning Method: {st.session_state.tuning_method}")

    # Save model section
    save_model_path = "C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\Models"

    # Sidebar save model section
    with st.sidebar:
        if st.session_state.trained_model:
            if st.button("Save Model"):
                model_file = (
                    f"{st.session_state.model_type.replace(' ', '_')}_"
                    f"{sanitize_filename(st.session_state.cv_type)}_"
                    f"{round(st.session_state.model_accuracy * 100, 2)}.pkl"
                )
                model_path = os.path.join(save_model_path, model_file)

                os.makedirs(save_model_path, exist_ok=True)
                # Overwrite the existing model file if it exists
                with open(model_path, "wb") as f:
                    pickle.dump(st.session_state.trained_model, f)

                st.session_state.model_saved = True
                st.sidebar.success(f"Model saved as `{model_file}`!")


if __name__ == "__main__":
    main()
