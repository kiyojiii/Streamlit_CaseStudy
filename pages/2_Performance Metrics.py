import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    log_loss, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# Function to load the dataset
@st.cache_data
def load_data():
    file_path = 'C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\csv\\survey lung cancer.csv'
    dataframe = pd.read_csv(file_path)
    return dataframe

# Function to clean the dataset
def clean_data(data):
    # Drop rows with blank or missing values across all columns
    data = data.dropna(subset=data.columns)
    return data

# Preprocessing function with mapping logic
def preprocess_data(data):
    # Map GENDER to 1 (M) and 0 (F)
    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    # Map binary columns (2 -> 1, 1 -> 0) for relevant columns
    for col in data.columns:
        if col not in ['AGE', 'GENDER', 'LUNG_CANCER']:
            data[col] = data[col].map({2: 1, 1: 0})
    return data

# Function to get the selected model
def get_model(algorithm):
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
        "MLP Classifier": MLPClassifier(),
        "Perceptron Classifier": CalibratedClassifierCV(Perceptron(), method='sigmoid', cv=3),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine (SVM)": SVC(probability=True)
    }
    return models[algorithm]

# Function to calculate log loss
def calculate_log_loss(model, X, Y, cv_method):
    log_loss_values = []
    for train_index, test_index in cv_method.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        Y_prob = model.predict_proba(X_test)[:, 1]

        if len(np.unique(Y_test)) > 1:
            fold_log_loss = log_loss(Y_test, Y_prob)
            log_loss_values.append(fold_log_loss)
        else:
            log_loss_values.append(np.nan)

    log_loss_values = [loss for loss in log_loss_values if not np.isnan(loss)]
    return log_loss_values

# Function to evaluate the model
def evaluate_model(dataframe, method, algorithm, test_size):
    model = get_model(algorithm)
    X = dataframe.iloc[:, :-1].values
    Y = dataframe.iloc[:, -1].values
    X_test = Y_test = num_folds = None
    results = []
    log_loss_values = []
    matrix = None
    report = {}
    true_labels = []
    predicted_probs = []

    if method == "Train/Test Split":
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        model.fit(X_train, Y_train)
        results = [model.score(X_test, Y_test)]
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            if len(np.unique(Y_test)) > 1:
                log_loss_values.append(log_loss(Y_test, y_prob))
            predicted_probs.extend(y_prob)
            true_labels.extend(Y_test)
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(Y_test, y_pred)
        report = classification_report(Y_test, y_pred, output_dict=True)

    elif method == "K-Fold Cross Validation":
        num_folds = st.sidebar.slider("Number of Folds (K-Fold):", 2, 20, 10)
        cv_method = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        for train_index, test_index in cv_method.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            model.fit(X_train, Y_train)
            results.append(model.score(X_test, Y_test))
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                if len(np.unique(Y_test)) > 1:  # Check if Y_test has more than one unique label
                    log_loss_values.append(log_loss(Y_test, y_prob))
                predicted_probs.extend(y_prob)
                true_labels.extend(Y_test)
        y_pred = model.fit(X, Y).predict(X)
        matrix = confusion_matrix(Y, y_pred)
        report = classification_report(Y, y_pred, output_dict=True)

    # elif method == "Leave One Out Cross Validation":
    #     cv_method = LeaveOneOut()
    #     for train_index, test_index in cv_method.split(X):
    #         X_train, X_test = X[train_index], X[test_index]
    #         Y_train, Y_test = Y[train_index], Y[test_index]
    #         model.fit(X_train, Y_train)
    #         if hasattr(model, "predict_proba"):
    #             y_prob = model.predict_proba(X_test)[:, 1]
    #             if len(np.unique(Y_test)) > 1:  # Check if Y_test has more than one unique label
    #                 log_loss_values.append(log_loss(Y_test, y_prob))
    #             predicted_probs.extend(y_prob)
    #             true_labels.extend(Y_test)
    #         results.append(model.score(X_test, Y_test))
    #     y_pred = model.fit(X, Y).predict(X)
    #     matrix = confusion_matrix(Y, y_pred)
    #     report = classification_report(Y, y_pred, output_dict=True)

    return model, results, log_loss_values, matrix, report, X_test, Y_test, num_folds, true_labels, predicted_probs

# Display metrics based on user selection
def display_metrics(model, results, log_loss_values, matrix, report, method, X_test=None, Y_test=None, num_folds=None, true_labels=None, predicted_probs=None):
    st.sidebar.subheader("Display Options")
    display_accuracy = st.sidebar.checkbox("Show Classification Accuracy", value=True)
    display_log_loss = st.sidebar.checkbox("Show Log Loss", value=True)
    display_conf_matrix = st.sidebar.checkbox("Show Confusion Matrix", value=True)
    display_class_report = st.sidebar.checkbox("Show Classification Report", value=True)
    display_roc = st.sidebar.checkbox("Show ROC Curve", value=True)

    if display_accuracy:
        st.subheader("Classification Accuracy")
        st.write(f"Accuracy: {np.mean(results) * 100:.2f}% ± {np.std(results) * 100:.2f}%")
        # Plotting a boxplot for accuracy results
        fig, ax = plt.subplots()
        ax.boxplot(results, patch_artist=True)
        ax.set_title('Classification Accuracy')
        ax.set_ylabel('Accuracy')
        st.pyplot(fig)
    else:
        st.subheader("Classification Accuracy")
        st.error("Accuracy cannot be determined. No results are available.")

    if display_log_loss and log_loss_values:
        st.subheader("Log Loss")
        average_log_loss = np.mean(log_loss_values)
        stdev_log_loss = np.std(log_loss_values)
        st.write(f"Mean Log Loss: {average_log_loss:.3f} ± {stdev_log_loss:.3f}")
        # Plotting a line graph for log loss values
        fig, ax = plt.subplots()
        ax.plot(log_loss_values, label='Log Loss per Fold', marker='o', linestyle='-')
        ax.set_title('Log Loss for Each Fold')
        ax.set_xlabel('Fold Number')
        ax.set_ylabel('Log Loss')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    else:
        st.subheader("Log Loss")
        st.error("Log Loss cannot be determined. No results are available.")

    if display_conf_matrix:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_).plot(ax=ax)
        st.pyplot(fig)
    else:
        st.subheader("Confusion Matrix")
        st.error("Confusion Matrix cannot be determined. No results are available.")

    # Classification Report
    if display_class_report:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.markdown("""<style>
            .report-table { font-size: 18px; width: 100%; text-align: center; }
            .report-table th { font-size: 20px; padding: 10px; background-color: #f0f0f0; text-align: center; }
            .report-table td { font-size: 18px; padding: 10px; text-align: center; }
        </style>""", unsafe_allow_html=True)
        st.write(report_df.to_html(classes="report-table", index=True), unsafe_allow_html=True)
    else:
        st.subheader("Classification Report")
        st.error("Classification Report cannot be determined. No results are available.")

    if display_roc and true_labels and predicted_probs:
        st.subheader("ROC AUC and Curve")
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = roc_auc_score(true_labels, predicted_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC AUC and Curve")
        plt.legend()
        st.pyplot(plt)
    else:
        st.subheader("ROC AUC and Curve")
        st.error("ROC AUC and Curve cannot be determined. No results are available.")

# Main app
def main():
    st.sidebar.subheader("Classification Sampling Techniques")
    method = st.sidebar.selectbox("Choose Sampling Technique:", ["Train/Test Split", "K-Fold Cross Validation"])
    algorithm = st.sidebar.selectbox("Choose Machine Learning Algorithm:", [
        "Decision Tree", "Gaussian Naive Bayes", "AdaBoost Classifier",
        "K-Nearest Neighbors Classifier", "MLP Classifier", "Perceptron Classifier",
        "Random Forest", "Support Vector Machine (SVM)"
    ])
    if method == "Train/Test Split":
        test_size = st.sidebar.slider("Test Size (Proportion)", 0.1, 0.9, 0.2, 0.05)
    else:
        test_size = 0.2  # Default value or calculate as needed for other methods
    
    st.title("Classification Performance Metrics")

    dataframe = load_data()
    st.subheader("Dataset Preview (Before Cleaning)")
    st.write(dataframe.head())

    dataframe = clean_data(dataframe)
    st.subheader("Dataset Preview (After Cleaning)")
    st.write(dataframe.head())

    dataframe = preprocess_data(dataframe)
    st.subheader("Dataset Preview (After Preprocessing)")
    st.write(dataframe.head())

    # Evaluate the model with dynamic test size
    model, results, log_loss_values, matrix, report, X_test, Y_test, num_folds, true_labels, predicted_probs = evaluate_model(dataframe, method, algorithm, test_size)
    display_metrics(model, results, log_loss_values, matrix, report, method, X_test, Y_test, num_folds, true_labels, predicted_probs)

if __name__ == "__main__":
    main()

