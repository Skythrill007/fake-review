import streamlit as st
import joblib
import preprocess as pp

#loading model
lgr_model = joblib.load('logistic_regression')

st.set_page_config(page_title="Fake Review Detection", page_icon="🕵️‍♂️")
st.header("Fake Review Detection")
review = st.text_input(label="Enter a review")
run = st.button("Predict")

if run:
    if review == "":
        st.subheader("Please enter a review !!")

    else:
       input = pp.preprocess_input(pp.preprocess_text(str(review)))
       output = lgr_model.predict(input)
       if(output==['OR']):
           st.subheader("Real Review")
       else:
           st.subheader("Fake Review")

