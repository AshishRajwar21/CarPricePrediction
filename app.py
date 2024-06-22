
#import required libraries
import numpy as np
import streamlit as st
import pickle

#import model
model_data = pickle.load(open('model/model.pkl','rb'))
car_df = model_data['df']

#headings
st.title('Car Price Prediction')
st.sidebar.header('Car Details')
st.subheader('Training Data Statistics')
st.write(car_df.describe())


#function to take input from.sidebar and convert it into dataframe
def carReport():
    option1 = st.sidebar.selectbox(
    "Which company car would you like to know?",
    tuple(sorted(car_df['company'].unique())),
    placeholder="Select a company ...",
    )
    st.sidebar.write("You selected:", option1)
    
    option2 = st.sidebar.selectbox(
    "Which car would you like to know?",
    tuple(sorted(car_df['name'][car_df['company']==option1].unique())),
    placeholder="Select a car_df ...",
    )
    st.sidebar.write("You selected:", option2)


    option3 = st.sidebar.selectbox(
    "Which year would you like to know?",
    tuple(sorted(car_df['year'].unique())),
    placeholder="Select a year ...",
    )
    st.sidebar.write("You selected:", option3)

    option4 = st.sidebar.number_input(
    "How many distance it driven", 
    placeholder="Kilometer driven ...")
    #st.sidebar.write("The current number is ", number)
    st.sidebar.write("You selected:", option4)
        
    option5 = st.sidebar.selectbox(
    "What type of fuel it should?",
    tuple(sorted(car_df['fuel_type'].unique())),
    placeholder="Select a fuel type ...",
    )
    st.sidebar.write("You selected:", option5)

    car_report_data = {
        'name': option2,
        'company':option1,
        'year': option3,
        'kms_driven':option4,
        'fuel_type':option5,
    }
    report_data = pd.DataFrame(car_report_data, index=[0])
    return report_data


#displaying the patient data
user_data = carReport()
#patientReport()
st.subheader('Car Report Data')
st.write(user_data)


pipe = model_data['model']
resulted_price = pipe.predict(user_data)


#Final prediction
st.subheader('Final Result: ')
st.write('The estimated price of the car is Rs.',str(resulted_price[0]))

#R2 score
st.write('The R2score of the model : ',model_data['r2_score_val'])





