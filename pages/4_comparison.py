import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.multiclass import unique_labels
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import LeaveOneOut

@st.cache(allow_output_mutation=True)
def load_data():
    file_path = 'C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\csv\\survey lung cancer.csv'
    return pd.read_csv(file_path)

def clean_data(data):
    return data.dropna()

def preprocess_data(data):
    mappings = {'M': 1, 'F': 0, 'YES': 1, 'NO': 0}
    data['GENDER'] = data['GENDER'].map(mappings)
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map(mappings)
    for col in data.columns:
        if col not in ['AGE', 'GENDER', 'LUNG_CANCER']:
            data[col] = data[col].map({2: 1, 1: 0})
    return data

def get_model(name):
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "MLP": MLPClassifier(),
        "Perceptron": CalibratedClassifierCV(Perceptron(), method='sigmoid', cv=3),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }
    return models.get(name)

def evaluate_single_split(X, y, model, train_index, test_index, all_labels):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = model.score(X_test, y_test)
    log_loss_value = np.nan
    try:
        log_loss_value = log_loss(y_test, y_proba, labels=all_labels)
    except Exception:
        pass

    roc_auc_value = np.nan
    if len(np.unique(y_test)) > 1:
        roc_auc_value = roc_auc_score(y_test, y_proba[:, 1])

    return accuracy, roc_auc_value, log_loss_value, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)

def evaluate_models(dataframe, models, sampling_method, test_size=None, num_folds=None):
    X = dataframe.drop('LUNG_CANCER', axis=1)
    y = dataframe['LUNG_CANCER']
    all_labels = unique_labels(y)  # Get all unique labels in the target variable
    results = []
    
    for model_name in models:
        model = get_model(model_name)
        if sampling_method == "Train/Test Split":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = model.score(X_test, y_test) * 100  # Multiply by 100 to convert to percentage
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) * 100  # Same for ROC AUC
            
            # Calculate log_loss only if there are at least two unique labels in y_test
            log_loss_value = np.nan
            if len(np.unique(y_test)) > 1:
                log_loss_value = log_loss(y_test, model.predict_proba(X_test), labels=all_labels)
            
            report = classification_report(y_test, y_pred, output_dict=True)
            confusion_mat = confusion_matrix(y_test, y_pred)
            
            # Format using provided functions
            formatted_report = format_classification_reports([report])
            formatted_confusion_matrix = format_confusion_matrices([confusion_mat])
            
            results.append({
                'Model': model_name,
                'Accuracy': f"{accuracy:.2f}%",
                'ROC AUC': f"{roc_auc:.2f}%",
                'Log Loss': log_loss_value,
                'Confusion Matrix': formatted_confusion_matrix,
                'Classification Report': formatted_report,
            })
        elif sampling_method == "K-Fold Cross Validation":
            cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            # Create lists to hold metrics
            accuracies, roc_aucs, log_losses, conf_matrices, class_reports = [], [], [], [], []

            for train_index, test_index in cv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                accuracies.append(model.score(X_test, y_test))
                
                # Calculate log_loss only if there are at least two unique labels in y_test
                if len(np.unique(y_test)) > 1:
                    log_losses.append(log_loss(y_test, y_proba, labels=all_labels))
                else:
                    log_losses.append(np.nan)  # Append NaN for invalid log_loss calculation
                
                conf_matrices.append(confusion_matrix(y_test, y_pred))
                class_reports.append(classification_report(y_test, y_pred, output_dict=True))
                
                if len(np.unique(y_test)) > 1:
                    roc_aucs.append(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    roc_aucs.append(np.nan)  # Append NaN for invalid ROC AUC computation

            # Compute averages and format
            average_accuracy = np.mean(accuracies) * 100
            average_roc_auc = np.nanmean(roc_aucs) * 100
            average_log_loss = np.nanmean(log_losses)
            formatted_conf_matrices = format_confusion_matrices(conf_matrices)
            formatted_class_reports = format_classification_reports(class_reports)

            results.append({
                'Model': model_name,
                'Average Accuracy': f"{average_accuracy:.2f}%",
                'Average ROC AUC': f"{average_roc_auc:.2f}%",
                'Average Log Loss': f"{average_log_loss:.2f}",
                'Confusion Matrix': formatted_conf_matrices,
                'Classification Report': formatted_class_reports
            })

    return results

def format_confusion_matrices(conf_matrices):
    formatted = ["\n".join([" ".join(map(str, row)) for row in matrix]) for matrix in conf_matrices]
    return " | ".join(formatted)

def format_classification_reports(reports):
    # Assuming you want to average precision, recall, and f1-score
    precision = np.mean([report['weighted avg']['precision'] for report in reports if 'weighted avg' in report])
    recall = np.mean([report['weighted avg']['recall'] for report in reports if 'weighted avg' in report])
    f1_score = np.mean([report['weighted avg']['f1-score'] for report in reports if 'weighted avg' in report])
    return f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}"

def main():
    st.title("Model Performance Comparison")

    # Load and preprocess the data
    data = load_data()
    data = clean_data(data)
    data = preprocess_data(data)

    st.subheader("Data Preview")
    st.write(data.head())

    # Class distribution for validation
    class_counts = data['LUNG_CANCER'].value_counts()
    min_class_count = class_counts.min()

    # Sidebar configuration for Model 1
    st.sidebar.subheader("Configuration for Model 1")
    sampling_method1 = st.sidebar.selectbox(
        "Select Sampling Technique for Model 1",
        ["Select", "Train/Test Split", "K-Fold Cross Validation"],
        key="sampling_method1"
    )
    models1 = st.sidebar.multiselect(
        "Select Algorithms for Model 1",
        ["Decision Tree", "Gaussian Naive Bayes", "AdaBoost", "K-Nearest Neighbors", "MLP", "Perceptron", "Random Forest", "SVM"],
        key="models1"
    )
    test_size1 = num_folds1 = None

    # Train/Test Split for Model 1
    if sampling_method1 == "Train/Test Split":
        test_size1 = st.sidebar.slider("Test Size for Model 1", 0.1, 0.9, 0.2, 0.01, key="test_size1")
        if test_size1 > 0.85:
            st.sidebar.warning("Test size is too high! Consider reducing it below 85%.")

    # K-Fold Cross Validation for Model 1
    elif sampling_method1 == "K-Fold Cross Validation":
        num_folds1 = st.sidebar.slider("Number of Folds for Model 1", 2, 20, 5, key="num_folds1")
        if num_folds1 > min_class_count:
            st.sidebar.warning(f"Number of folds exceeds the smallest class size ({min_class_count}). Reduce the folds.")

    # Sidebar configuration for Model 2
    st.sidebar.subheader("Configuration for Model 2")
    available_sampling_methods2 = [method for method in ["Select", "Train/Test Split", "K-Fold Cross Validation"] if method != sampling_method1]
    sampling_method2 = st.sidebar.selectbox(
        "Select Sampling Technique for Model 2",
        available_sampling_methods2,
        key="sampling_method2"
    )
    models2 = st.sidebar.multiselect(
        "Select Algorithms for Model 2",
        ["Decision Tree", "Gaussian Naive Bayes", "AdaBoost", "K-Nearest Neighbors", "MLP", "Perceptron", "Random Forest", "SVM"],
        key="models2"
    )
    test_size2 = num_folds2 = None

    # Train/Test Split for Model 2
    if sampling_method2 == "Train/Test Split":
        test_size2 = st.sidebar.slider("Test Size for Model 2", 0.1, 0.9, 0.2, 0.01, key="test_size2")
        if test_size2 > 0.85:
            st.sidebar.warning("Test size is too high! Consider reducing it below 85%.")

    # K-Fold Cross Validation for Model 2
    elif sampling_method2 == "K-Fold Cross Validation":
        num_folds2 = st.sidebar.slider("Number of Folds for Model 2", 2, 20, 5, key="num_folds2")
        if num_folds2 > min_class_count:
            st.sidebar.warning(f"Number of folds exceeds the smallest class size ({min_class_count}). Reduce the folds.")

    # Comparison button
    if st.sidebar.button("Compare Models"):
        # Input validation
        if sampling_method1 == "Select" or not models1:
            st.sidebar.error("Please select a sampling technique and at least one algorithm for Model 1.")
        elif sampling_method2 == "Select" or not models2:
            st.sidebar.error("Please select a sampling technique and at least one algorithm for Model 2.")
        else:
            with st.spinner("Comparing models, please wait..."):
                # Evaluate Model 1
                results1 = evaluate_models(data, models1, sampling_method1, test_size=test_size1, num_folds=num_folds1)
                sorted_results1 = sorted(results1, key=lambda x: float(x[next(iter(x.keys() & {'Average Accuracy', 'Accuracy'}))][:-1]), reverse=True)

                # Evaluate Model 2
                results2 = evaluate_models(data, models2, sampling_method2, test_size=test_size2, num_folds=num_folds2)
                sorted_results2 = sorted(results2, key=lambda x: float(x[next(iter(x.keys() & {'Average Accuracy', 'Accuracy'}))][:-1]), reverse=True)

                # Display results side by side
                st.subheader("Comparison of Two Model Groups")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Results for Model 1")
                    st.write(pd.DataFrame(sorted_results1))

                with col2:
                    st.subheader("Results for Model 2")
                    st.write(pd.DataFrame(sorted_results2))

if __name__ == "__main__":
    main()


