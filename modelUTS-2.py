import streamlit as st
import joblib
import numpy as np

model = joblib.load('modelUTS.pkl')

def main():
    st.title(':red[Customer Churn Prediction]')

    # INPUT 1
    st.text("")
    st.subheader('Customer Credit Score')
    credit_score = st.number_input('Input the value below', min_value=300.0,
                                   max_value=850.0, value=300.0)
    st.write('*Answer:* ', credit_score)
    
    #credit_score = st.number_input('CreditScore', min_value=0, max_value=1000, value=0)
    #st.write('Your credit score number is ', credit_score)

    # INPUT 2
    st.text("")
    st.subheader('Customer Location')
    st.write('0: France | 1: Germany | 2: Spain')
    geography = st.selectbox('Select the option below', [0, 1, 2])
    if geography == 0:
        st.write('*Answer: France*')
    elif geography == 1:
        st.write('*Answer: Germany*')
    else:
        st.write('*Answer: Spain*')
        
    #geography = st.selectbox('Geography', ["France", "Spain", "Germany"])
    #st.write('Your area:', geography)

    # INPUT 3
    st.text("")
    st.subheader('Customer Gender')
    st.write('0: Female | 1: Male')
    gender = st.radio('Choose the option below', [0, 1])
    if gender == 0:
        st.write('*Answer: Female*')
    else:
        st.write('*Answer: Male*')
        
    #gender = st.radio('Gender', ["Female", "Male"])
    #if gender == 'Female':
        #st.write('You are a female')
    #else:
        #st.write('You are a male')

    # INPUT 4
    st.text("")
    st.subheader('Customer Age')
    st.write('0: Female | 1: Male')
    age = st.number_input('Input the number below', min_value=17, max_value=100, value=17)
    st.write('*Answer:*', age, '*years old*')
    
    #age = st.number_input('Age')
    #st.write('You are', age, 'years old')

    # INPUT 5
    st.text("")
    st.subheader('Customer Tenure')
    tenure = st.slider('Choose the number below', min_value=0, max_value=20, value=0)
    st.write('*Answer:*', tenure)
    
    #tenure = st.slider('Tenure', min_value=0, max_value=10, value=1)

    # INPUT 6
    st.text("")
    st.subheader('Customer Balance Value')
    balance = st.number_input('Input the value below', min_value=0)
    st.write('*Answer:* ', balance)
    
    #balance = st.slider('Balance', min_value=0.0, max_value=10.0, value=0.1)

    # INPUT 7
    st.text("")
    st.subheader('Number of Products Owned')
    product = st.slider('Choose the number below', min_value=0, max_value=5, value=0)
    st.write('*Answer:*', product)
    
    #product = st.slider('NumOfProducts', min_value=0, max_value=5, value=1)

    # INPUT 8
    st.text("")
    st.subheader('Do They Have a Credit Card?')
    st.write('0: No | 1: Yes')
    crcard = st.radio('Choose the option below', [0, 1], key='cc')
    if crcard == 0:
        st.write('*Answer: No*')
    else:
        st.write('*Answer: Yes*')
        
    #crcard = st.checkbox('HasCrCard')
    #if crcard:
        #st.write('You have a credit card')
    #else:
        #st.write('You do not have a credit card')

    # INPUT 9
    st.text("")
    st.subheader('Do They Have an Active Membership?')
    st.write('0: No | 1: Yes')
    active = st.radio('Choose the option below', [0, 1], key='member')
    if active == 0:
        st.write('*Answer: No*')
    else:
        st.write('*Answer: Yes*')
        
    #active = st.checkbox('IsActiveMember')
    #if active:
        #st.write('You have an active membership')
    #else:
        #st.write('You do not have an active membership')

    # INPUT 10
    st.text("")
    st.subheader('Customer Salary')
    salary = st.number_input('Input the estimation below', min_value=0.00,
                             max_value=None, value=0.00)
    st.write('*Answer:*', salary)
    
    st.text("")
    st.text("")
    
    #salary = st.slider('EstimatedSalary', min_value=0.00, max_value=None, value=0.01)

    if st.button('Create Prediction'):
        features = [credit_score,geography,gender,age,tenure,
                   balance,product,crcard,active,salary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')
        if result == 1:
            st.subheader('Churn customer')
        else:
            st.subheader('Loyal customer')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    input_array = np.nan_to_num(input_array)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
