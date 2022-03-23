
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import functions
import time
import numpy as np


if "button_clicked" not in st.session_state:    
    st.session_state.button_clicked = False

def callback():
    st.session_state.button_clicked = True

st.title('Just Put Any HASHTAG YOU Want To Analys it  !!!')
text_Input = st.text_input('put Hashtag',placeholder="#twitter")
def getCountry() -> np.ndarray:
    array = np.array(['ar','en'])
    return array

col1, col2, col3 , col4= st.columns(4)
with col1:
    selectBox = st.selectbox('select Language',getCountry())
    print(selectBox)
with col2:
    dateInputIn = st.date_input('Date from:')
    print(dateInputIn)
with col3:
    dataInputOu = st.date_input('Date to:')
    
with col4:
    numberOfLikes = st.text_input('number of likes')

space1, spac2 , space3, space4 , space5 = st.columns(5)
with space1:
    pass
with spac2:
    pass
with space4:
    pass
with space5:
    pass
with space3:
    analysButton = st.button('lets Analyize',on_click= callback)
    
if analysButton or st.session_state.button_clicked:
    functions.featchData(text_Input,str(dateInputIn),numberOfLikes)
    df = functions.getData(text_Input,selectBox)

    st.markdown("### Data preview")
    st.dataframe(df.head(10),1000,410)

    with st.form(key='myForm'):
        st.markdown('### Chose tyep of analyze data by sentiment ')
        selectSentiment = st.selectbox('sentiment type',['All','positive','natural','negative']
        ,index=0)

        df['tweet'] = df['tweet'].map(lambda x : functions.cleaner(x,selectBox))
        df = functions.anlalyseTheTweets(df)
        df['Label'] = df['sentiment'].apply(functions.sentiment_category)

        if selectSentiment != 'All':
            df = df[df['Label'] == selectSentiment]

        wordCloud = st.selectbox('chose wordCloud sentiment',['All Words','Positive Words','natural Words','negative Words'])
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
            st.dataframe(df.head(10),1000,500)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot = st.pyplot(functions.getWordCloud(df,selectBox))
            print(df.shape)
            os.remove(f'{text_Input}.csv')
        
    