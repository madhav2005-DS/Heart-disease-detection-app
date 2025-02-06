import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming `df` is your original dataframe

# Step 1: Prepare features and scale them


warnings.filterwarnings('ignore')

# Sample Heart Disease Dataset with glucose, BMI, age, and family history
data = {
    'age': [63, 67, 67, 37, 41, 56, 57, 45, 46, 56],
    'sex': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'glucose': [150, 200, 190, 110, 120, 160, 170, 180, 130, 140],
    'bmi': [25.5, 28.2, 26.4, 24.0, 23.5, 27.6, 29.0, 24.8, 26.2, 24.3],
    'family_history': [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    'cp': [1, 4, 4, 3, 2, 1, 1, 1, 1, 1],
    'trestbps': [145, 160, 120, 130, 130, 140, 130, 130, 130, 130],
    'chol': [233, 286, 229, 250, 204, 236, 235, 240, 226, 236],
    'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'restecg': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'thalach': [150, 108, 129, 187, 172, 178, 160, 168, 162, 148],
    'exang': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'oldpeak': [2.3, 3.5, 1.4, 0.6, 0.6, 0.7, 0.6, 0.4, 1.5, 0.2],
    'slope': [3, 3, 2, 1, 1, 1, 1, 1, 1, 1],
    'ca': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'thal': [2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
    'target': [0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

# Prepare feature and target variables
X = df.drop(columns=['target'])
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = df.drop(columns=['target'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=2)  # Choose the number of components you want to reduce to
X_pca = pca.fit_transform(X_scaled)
# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model and scaler for predictions in Streamlit app
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("Heart Disease Prediction App")

# Input fields for user to enter their details
st.header("Enter your details to predict heart disease risk")

age = st.selectbox("Age", list(range(18, 101)))
sex = st.selectbox("Sex", ["Male", "Female"])
glucose = st.selectbox("Glucose Levels (mg/dL)", list(range(0, 301,)))
bmi = st.selectbox("Body Mass Index (BMI)", [round(i*0.5, 1) for i in range(0, 51)])
family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
trestbps = st.selectbox("Resting Blood Pressure (trestbps)", list(range(90, 201)))
chol = st.selectbox("Serum Cholesterol (chol)", list(range(200, 400, 10)))
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["No", "Yes"])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
thalach = st.selectbox("Max Heart Rate Achieved (thalach)", list(range(150, 220)))
exang = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
oldpeak = st.selectbox("Depression Induced by Exercise (oldpeak)", [round(i*0.1, 1) for i in range(0, 60)])
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [1, 2, 3])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [3, 6, 7])

# Map 'Yes'/'No' to 1/0 for binary variables
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0
sex = 1 if sex == "Male" else 0

# Prepare input for prediction
input_data = np.array([[age, sex, glucose, bmi, family_history, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input data using the trained scaler
input_data_scaled = scaler.transform(input_data)

# Predict the risk of heart disease
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction == 1:
        st.subheader("Prediction: High Risk of Heart Disease")
        st.write("You may be at risk for heart disease. Please consult with a healthcare provider for further tests.")
    else:
        st.subheader("Prediction: Low Risk of Heart Disease")
        st.write("You are at a lower risk for heart disease, but maintaining a healthy lifestyle is always recommended.")

# Analysis and Visualization section
st.header("Data Analysis & Visualizations")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)

# Feature Importances
st.subheader("Feature Importance in Prediction")
feature_importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance")
st.pyplot(plt)
