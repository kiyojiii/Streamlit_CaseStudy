import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import base64

#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Lung Cancer Dashboard", page_icon=":bar_chart:", layout="wide")

# Path to your logo file
logo_path = "C:/Users/user/Desktop/jeah/ITD105/CaseStudy/logo/logo.png"

# Function to load and encode the image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Encode the logo image
encoded_logo = get_base64_image(logo_path)

# Display the logo with a larger size and round border in the sidebar
with st.sidebar:
    st.markdown(
        f"""
        <style>
        .logo {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 250px; /* Adjusted size for a larger logo */
            height: 250px; /* Ensure the height matches the width for circular shape */
            border-radius: 50%; /* Makes the logo round */
            border: 2px solid #ccc; /* Optional border around the logo */
        }}
        </style>
        <img src="data:image/png;base64,{encoded_logo}" class="logo" />
        """,
        unsafe_allow_html=True,
    )

# Main content area (replace with your existing content)
st.title("Welcome to the Data Dashboard")
st.write("Explore the csv file lung cancer-related data and insights.")

#######################################
# DATA LOADING
#######################################

@st.cache_data
def load_data(path: str):
    """Loads the dataset from the specified path."""
    return pd.read_csv(path)

# Load dataset
file_path = "C:\\Users\\user\\Desktop\\jeah\\ITD105\\CaseStudy\\csv\\survey lung cancer.csv"
df = load_data(file_path)

# Check for required columns
if 'LUNG_CANCER' not in df.columns:
    st.error("The dataset is not valid for lung cancer analysis. Please provide a proper dataset.")
    st.stop()

# Preview the dataset
with st.expander("Data Preview"):
    st.dataframe(df)

#######################################
# VISUALIZATION FUNCTIONS
#######################################

def plot_gauge(label, value, max_bound, color):
    """Displays a gauge chart for a given metric."""
    st.markdown(f"**{label}**")
    fig = go.Figure(
        go.Indicator(
            value=value,
            mode="gauge+number",
            gauge={
                "axis": {"range": [0, max_bound]},
                "bar": {"color": color},
            },
        )
    )
    fig.update_layout(
        height=150,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_gender_vs_lung_cancer():
    """Plots a bar chart showing the gender distribution against lung cancer occurrence."""
    data = df.groupby(['GENDER', 'LUNG_CANCER']).size().reset_index(name='Count')
    fig = px.bar(
        data,
        x='GENDER',
        y='Count',
        color='LUNG_CANCER',
        title="Gender Distribution vs Lung Cancer (Yes/No)",
        labels={'GENDER': 'Gender', 'Count': 'Count', 'LUNG_CANCER': 'Lung Cancer'},
        barmode='group',
        color_discrete_map={"YES": "#ff0e0e", "NO": "#4BB543"}
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_lung_cancer_rate_by_age():
    """Plots a line chart showing lung cancer rates across age groups."""
    df['Age Group'] = pd.cut(df['AGE'], bins=range(0, 101, 10), labels=[f"{i}-{i+9}" for i in range(0, 100, 10)])
    data = df.groupby('Age Group')['LUNG_CANCER'].apply(lambda x: (x == "YES").mean() * 100).reset_index()  # Multiply by 100
    data.columns = ['Age Group', 'Lung Cancer Rate']

    fig = px.line(
        data,
        x='Age Group',
        y='Lung Cancer Rate',
        title="Lung Cancer Rate by Age Group",
        markers=True,
        labels={'Lung Cancer Rate': 'Rate (%)'},
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_gender_age_distribution():
    """Plots a stacked histogram showing age and gender distribution."""
    fig = px.histogram(
        df,
        x='AGE',
        color='GENDER',
        title="Gender and Age Distribution for Lung Cancer",
        barmode='stack',
        color_discrete_map={"M": "#4682B4", "F": "#FF69B4"}
    )
    st.plotly_chart(fig, use_container_width=True)

#######################################
# STREAMLIT LAYOUT
#######################################

# Calculate class balance ratio
class_counts = df['LUNG_CANCER'].value_counts()
if len(class_counts) > 1:
    class_balance_ratio = class_counts.min() / class_counts.max() * 100
else:
    class_balance_ratio = 0

# Top Section
st.subheader("Dataset Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rows = len(df)
    plot_gauge("Data Count (Gauge)", total_rows, max_bound=total_rows * 1.2, color="#4CAF50")

with col2:
    total_columns = len(df.columns)
    plot_gauge("Column Count (Gauge)", total_columns, max_bound=total_columns * 1.5, color="#2196F3")

with col3:
    missing_count = df.isnull().sum().sum()
    plot_gauge("Missing Data (Gauge)", missing_count, max_bound=(total_rows * total_columns) * 0.1, color="#FF9800")

with col4:
    plot_gauge("Class Balance Ratio (Gauge)", class_balance_ratio, max_bound=100, color="#E91E63")

# Visualizations
st.subheader("Visualizations")

# Row 1: Gender vs Lung Cancer and Age Group Analysis
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    plot_gender_vs_lung_cancer()

with row1_col2:
    plot_lung_cancer_rate_by_age()

# Row 2: Gender and Age Distribution
st.subheader("Age and Gender Distribution")
plot_gender_age_distribution()
