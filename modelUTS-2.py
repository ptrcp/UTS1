import streamlit as st
import joblib
import numpy as np

model = pickle.load('OOPmodel.pkl')

def main():
    st.title('Prediction Model of Churn and Not Churn')

    credit_score = st.number_input(label, min_value=0, max_value=1000, value="min",
                                   label_visibility="visible")
    st.write('Your credit score number is ', credit_score)
    
    geography = st.selectbox("Location", ("0:France", "1:Spain", "2:Germany"),
                             index=None, placeholder="Select geography area...",)
    st.write('Your area:', geography)
    
    gender = st.radio("Your gender", ["***Female***", "***Male***"])
    if gender == '***Female***':
        st.write('You are a female')
    else:
        st.write('You are a male')
    
    age = st.number_input("Your age", value=None, placeholder="Your current age...")
    st.write('You are', age, 'years old')

    tenure = st.slider('Tenure', min_value=0, max_value=10, value=1)
    
    balance = st.slider('Balance', min_value=0.0, max_value=10.0, value=0.1)
    
    product = st.slider('Number of Products', min_value=0, max_value=5, value=1)
    
    crcard = st.checkbox('Credit Card')
    if crcard:
        st.write('You have a credit card')
    else:
        st.write('You do not have a credit card')
        
    active = st.checkbox('Active member')
    if active:
        st.write('You have an active membership')
    else:
        st.write('You do not have an active membership')
        
    salary = st.slider('Your estimated salary', min_value=0.00, max_value=None, value=0.01)

    if st.button('Make Prediction'):
        features = [credit_score,geography,gender,age,tenure,
                   balance,product,crcard,active,salary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')
        if result == 1:
            st.write('Churn customer')
        else:
            st.write('Loyal customer')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
