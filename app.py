import streamlit as st
import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Starbucks Offer Completion Predictor')

st.header('Customer Demographics')
age = st.slider('Age', 18, 100, 40)
income = st.number_input('Annual Income', min_value=30000, max_value=120000, value=60000, step=1000)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])

st.header('Offer Details')
offer_type = st.selectbox('Offer Type', ['BOGO', 'Discount', 'Informational'])
difficulty = st.slider('Difficulty', 0, 20, 5)
duration = st.slider('Duration (days)', 3, 10, 7)
reward = st.slider('Reward', 0, 10, 5)
member_tenure_days = st.slider('Member Tenure (days)', 0, 2000, 500)

st.header('Channels')
channel_email = st.checkbox('Email')
channel_web = st.checkbox('Web')
channel_mobile = st.checkbox('Mobile')
channel_social = st.checkbox('Social')

gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
offer_map = {'BOGO': 0, 'Discount': 1, 'Informational': 2}

features = np.array([[
    gender_map[gender], age, income, reward, difficulty, duration,
    offer_map[offer_type], member_tenure_days,
    int(channel_email), int(channel_web), int(channel_mobile), int(channel_social)
]])

if st.button('Predict'):
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    if pred == 1:
        st.success(f'Likely to complete the offer ({prob:.0%} probability)')
    else:
        st.error(f'Unlikely to complete the offer ({prob:.0%} probability)')
