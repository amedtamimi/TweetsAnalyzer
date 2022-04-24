from pyexpat import ExpatError
import streamlit as st
import os
import functions
import numpy as np
from PIL import Image

image = Image.open('raw/twitter-logo.png')
side1,side2 = st.columns(2)
with side1:
    st.image(image,width=100,output_format='outo',channels='RGB')
with side2:
    pass
st.title('Just Put Any HASHTAG YOU Want To Analys it  !!!')
text_Input = st.text_input('put Hashtag',placeholder="#twitter")
def getCountry() -> np.ndarray:
    array = np.array(['ar','en'])
    return array

col1, col2, col3 , col4, col5,col6= st.columns(6)
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
with col5:
    selectSentiment = st.selectbox('sentiment type',['All','Positive','Natural','Negative'],index=0)
with col6:
    wordCloud = st.selectbox('wordCloud sen',['All','Positive ','Natural','Negative'])
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
    if "button_clicked" not in st.session_state:    
            st.session_state.button_clicked = False
    def callback():
        st.session_state.button_clicked = True
    analysButton = st.button('lets Analyize')

if analysButton:
    df = functions.featchData(text_Input,str(dateInputIn),numberOfLikes,selectBox)
    st.markdown("### Data preview")
    st.dataframe(df.head(20),1000,410)
    df['tweet'] = df['tweet'].map(lambda x : functions.cleaner(x,selectBox))
    df = functions.anlalyseTheTweets(df,selectBox)

    if selectBox == 'en':
        df['Label'] = df['sentiment'].apply(functions.sentiment_category)
        if selectSentiment != 'All':
            df = df[df['Label'] == selectSentiment]
    else:
        if(selectSentiment != 'All'):
            df = df[df['sentiment'] == selectSentiment]
    st.markdown("### Cleaned Data and Sentiment ")
    st.dataframe(df.head(20),1000,500)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plot = st.pyplot(functions.getWordCloud(df,selectBox))

    barPlot  = st.pyplot(functions.barblot(df,selectBox))
        
    