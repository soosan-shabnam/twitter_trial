import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Twitter Sentimenttal Analysis')

text = st.text_area('Enter tweet',None)

if st.button('Calculate sentiment'):
    res = predict([text])
    st.text(res[0])






