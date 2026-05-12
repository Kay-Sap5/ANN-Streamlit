import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle

import streamlit as st


model = load_model("model.h5")

with open("label_encoder_gender.pkl",'rb') as file:
    label_encoder_gender = pickle.load(file)

with open("onehotencoder_geo.pkl" , 'rb') as file:
    ohe_geo = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)


credit_score = st.slider(label="CreditScore" ,min_value=100 , max_value=999)
geography = st.selectbox(label="Gegraphy" , options=ohe_geo.categories_[0])
gender = st.selectbox("Gender" , label_encoder_gender.classes_)
age = st.slider("Age",17,92)
tenure = st.slider('Tenure',0,10)
balance = st.number_input("Balance")
num_of_product = st.slider("Number of Product" , 1,4)
has_cr_card = st.selectbox("Has Credit Cat" , [0,1])
is_active_member = st.selectbox("Is Active Member" , [0,1])
estimated_salary = st.number_input("Salary")


input_data = {'CreditScore': credit_score,
 'Geography': geography,
 'Gender': gender,
 'Age': age,
 'Tenure': tenure,
 'Balance': balance,
 'NumOfProducts': num_of_product,
 'HasCrCard': has_cr_card,
 'IsActiveMember':is_active_member,
 'EstimatedSalary':estimated_salary,}

df = pd.DataFrame([input_data])

df['Gender']  = label_encoder_gender.transform([df['Gender']])

geo_df = pd.DataFrame(data=ohe_geo.transform([df['Geography']]).toarray() , columns=ohe_geo.get_feature_names_out())
final_df = pd.concat([df.drop(columns=['Geography']) , geo_df ],axis=1)
final_df = scaler.transform(final_df)
pred = model.predict(final_df)

print(final_df)
st.write(f"Prediction Propability {pred[0][0]}")
print(pred[0][0])
if pred[0][0] <0.5 :
    st.write("The Customer will not Leave")
else:
    st.write("The Customer will Leave")
