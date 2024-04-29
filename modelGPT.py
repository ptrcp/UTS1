import streamlit as st
import numpy as np
import pandas as pd
import joblib

def main():
    # Load the model
    model = joblib.load('modelUTS.pkl')

    # Collect input features
    features = collect_input_features()

    # Make prediction
    result = make_prediction(model, features)
    st.write('Prediction:', result)

def collect_input_features():
    # Collect input features from user
    age = st.number_input('Age')
    credit_score = st.number_input('Credit Score')
    tenure = st.number_input('Tenure')
    balance = st.number_input('Balance')
    num_of_products = st.number_input('Number of Products')
    has_credit_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
    is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
    estimated_salary = st.number_input('Estimated Salary')

    # Process categorical features
    has_credit_card = 1 if has_credit_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    # Create a dictionary of features
    features = {
        'Age': age,
        'CreditScore': credit_score,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }

    return pd.DataFrame([features])

def make_prediction(model, input_data):
    # Preprocess input data
    input_data = input_data.fillna(0)  # Fill missing values with 0 or apply any other strategy

    # Make prediction
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    main()
