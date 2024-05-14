import streamlit as st
import pandas as pd
from predict import predict_one_data
import torch


st.header("Quora Irrelevent question predictor")
question = st.text_input("please enter your qustion")
question = str(question)
predicted = 2
df = pd.DataFrame({'question_text': [question]})
print(df)
predicted = predict_one_data(df)
if predicted == 0:
    st.write("It is a valid Question")
else:
    st.write("Sorry !! Its a invalid Question")