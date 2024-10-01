import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('MODEL/churn_model.h5')

@st.cache_resource
def load_gender_encoder():
    with open('DATA/gender_encoder.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_geo_encoder():
    with open('DATA/geo_encoder.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_scaler():
    with open('DATA/churn_scaler.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()
gender_encoder = load_gender_encoder()
geo_encoder = load_geo_encoder()
scaler = load_scaler()

st.title('Customer Churn Predicter')

geography = st.selectbox('Geography', geo_encoder.classes_)
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography' : [geo_encoder.transform([geography])[0]],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

if st.button("SUBMIT"):
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.success(f'Probability : {prediction_proba:.2f}, The customer is likely to churn')
    else:
        st.success(f'Probability : {prediction_proba:.2f}, The customer is not likely to churn')
