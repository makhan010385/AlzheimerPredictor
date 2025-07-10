import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Page config
st.set_page_config(page_title="Alzheimer's Predictor", layout="centered")

st.title("🧠 Early-Stage Alzheimer's Disease Prediction using ML")
st.markdown("Built with data from the [OASIS dataset](https://www.kaggle.com/jboysen/mri-and-alzheimers)")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("oasis_longitudinal.csv")
    df = df.drop(columns=["MRI ID", "Visit", "Subject ID"])
    df['SES'].fillna(df['SES'].median(), inplace=True)
    df['MMSE'].fillna(df['MMSE'].median(), inplace=True)
    df = df.dropna()
    df['M/F'] = LabelEncoder().fit_transform(df['M/F'])  # Male=1, Female=0
    df['Group'] = df['Group'].replace({'Nondemented': 0, 'Demented': 1, 'Converted': 1})
    return df

df = load_data()

# Sidebar user input
st.sidebar.header("📝 Patient Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 60, 96, 75)
educ = st.sidebar.slider("Years of Education", 0, 20, 12)
ses = st.sidebar.slider("Socio Economic Status", 1, 5, 2)
mmse = st.sidebar.slider("MMSE Score", 16, 30, 27)
etiv = st.sidebar.slider("Estimated Total Intracranial Volume (eTIV)", 1100, 2000, 1500)
nwbv = st.sidebar.slider("Normalized Whole Brain Volume (nWBV)", 0.5, 0.9, 0.7)
asf = st.sidebar.slider("Atlas Scaling Factor (ASF)", 0.8, 1.5, 1.2)

# Model selection
model_name = st.sidebar.selectbox("Choose ML Model", ["Random Forest", "SVM", "XGBoost", "Voting Classifier"])

# Input sample
input_data = pd.DataFrame([[
    1 if gender == "Male" else 0,
    age,
    educ,
    ses,
    mmse,
    etiv,
    nwbv,
    asf
]], columns=["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"])

# Training
X = df[["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]]
y = df["Group"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
dt = DecisionTreeClassifier(random_state=0)
rf = RandomForestClassifier(n_estimators=100, random_state=0)
svm = SVC(kernel='linear', probability=True, random_state=0)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

# Train models
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Voting
voting = VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)], voting='soft')
voting.fit(X_train, y_train)

# Predict
if model_name == "Random Forest":
    model = rf
elif model_name == "SVM":
    model = svm
elif model_name == "XGBoost":
    model = xgb
else:
    model = voting

prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0][prediction]

# Output
st.subheader("🧾 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ Likely Demented (Probability: {prediction_proba:.2f})")
else:
    st.success(f"✅ Likely Non-Demented (Probability: {prediction_proba:.2f})")

# Optional metrics
st.subheader("📊 Model Performance on Test Data")
y_pred = model.predict(X_test)
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
with col2:
    st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
