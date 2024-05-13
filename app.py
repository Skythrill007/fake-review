import streamlit as st
import joblib
import preprocess as pp
import requests
from streamlit_lottie import st_lottie




#loading model
lgr_model = joblib.load('logistic_regression')

st.set_page_config(page_title="Fake Review Detection", page_icon="🕵️‍♂️")
co1,co2,co3 = st.columns([1,2,1])

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/ca37c189-48d7-49f2-bec8-ad2e366b46d7/q0Wgrror7m.json"
lottie_hello = load_lottieurl(lottie_url_hello)

with co2:
    st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="medium", # medium ; high
        height=None,
        width=None,
        key=None,
    )
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

