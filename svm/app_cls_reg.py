import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Folder Setup
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config("End-to-End SVM", layout="wide")
st.title("End-to-End SVM Platform (Classification & Regression)")

# =====================================================
# Sidebar – SVM Settings
# =====================================================
st.sidebar.header("SVM Settings")

problem_type = st.sidebar.radio(
    "Problem Type",
    ["Classification", "Regression"]
)

kernel = st.sidebar.selectbox(
    "Kernel",
    ["linear", "rbf", "poly", "sigmoid"]
)

C = st.sidebar.slider(
    "C (Regularization)",
    0.01, 10.0, 1.0
)

gamma = st.sidebar.selectbox(
    "Gamma",
    ["scale", "auto"]
)

# =====================================================
# Step 1 : Data Ingestion (California Housing)
# =====================================================
st.header("Step 1 : Data Ingestion")

@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame

df = load_data()

raw_path = os.path.join(RAW_DIR, "california_housing_raw.csv")
df.to_csv(raw_path, index=False)

st.success("California Housing Dataset Loaded")
st.info(f"Raw dataset saved at: {raw_path}")
st.dataframe(df.head())

# =====================================================
# Step 2 : Exploratory Data Analysis (Simple)
# =====================================================
st.header("Step 2 : Exploratory Data Analysis")

st.write("Dataset Shape:", df.shape)
st.write("Missing Values:")
st.write(df.isnull().sum())

# =====================================================
# Step 3 : Data Cleaning
# =====================================================
st.header("Step 3 : Data Cleaning")

strategy = st.selectbox(
    "Missing Value Strategy",
    ["Mean", "Median", "Drop Rows"]
)

df_clean = df.copy()

if strategy == "Drop Rows":
    df_clean.dropna(inplace=True)
else:
    for col in df_clean.columns:
        if strategy == "Mean":
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

st.success("Data Cleaning Completed")

# =====================================================
# Step 4 : Save Cleaned Dataset
# =====================================================
if st.button("Save Cleaned Dataset"):
    clean_path = os.path.join(
        CLEAN_DIR,
        "cleaned_california_housing.csv"
    )
    df_clean.to_csv(clean_path, index=False)
    st.success("Cleaned Dataset Saved")
    st.info(f"Saved at: {clean_path}")

# =====================================================
# Step 5 : Load Cleaned Dataset
# =====================================================
st.header("Step 5 : Load Cleaned Dataset")

clean_files = [
    f for f in os.listdir(CLEAN_DIR)
    if "california" in f.lower()
]

if not clean_files:
    st.error("No cleaned California Housing dataset found. Please save one in Step 4.")
    st.stop()

selected = st.selectbox("Select Cleaned Dataset", clean_files)
df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))

st.success(f"Loaded dataset: {selected}")
st.dataframe(df_model.head())

# =====================================================
# Step 6 : Train SVM
# =====================================================
st.header("Step 6 : Train SVM")

target = "MedHouseVal"

X = df_model.drop(columns=[target])
y = df_model[target]

# -------- Classification Logic --------
if problem_type == "Classification":
    threshold = y.median()
    st.info(f"Binary Classification → High Value ≥ {threshold:.2f}")
    y = (y >= threshold).astype(int)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# =====================================================
# Classification
# =====================================================
if problem_type == "Classification":
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# =====================================================
# Regression
# =====================================================
else:
    model = SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.success(f"R² Score: {r2:.2f}")
    st.success(f"MSE: {mse:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual House Value")
    ax.set_ylabel("Predicted House Value")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
