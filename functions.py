import twint
import streamlit as sst
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd 
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyarabic.araby import tokenize,is_arabicrange,strip_tashkeel
from ar_wordcloud import ArabicWordCloud
def featchData(hashtag_name,fromDate,numberOfLikes):
    tweets = twint.Config
    tweets.Search = [hashtag_name]
    tweets.Limit = 100
    tweets.Min_likes =numberOfLikes
    tweets.Since = fromDate
    tweets.Store_csv = True
    tweets.Output = f'{hashtag_name}.csv'
    twint.run.Search(tweets)
def getData(input,languges):
    df_tweets = pd.read_csv(f'{input}.csv')
    df_twitter_new =['user_id','username','name','tweet','language','likes_count']
    df_tweets = df_tweets[df_twitter_new]
    df_tweets = df_tweets[df_tweets['language'] == languges]
    return df_tweets

stop = ['إذ', 'إذا', 'إذما', 'إذن', 'أف', 'أقل', 'أكثر', 'ألا', 'إلا', 'التي', 'الذي', 'الذين', 'اللاتي', 'اللائي', 'اللتان', 'اللتيا', 'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'إلى', 'إليك', 'إليكم', 'إليكما', 'إليكن', 'أم', 'أما', 'أما', 'إما', 'أن', 'إن', 'إنا', 'أنا', 'أنت', 'أنتم', 'أنتما', 'أنتن', 'إنما', 'إنه', 'أنى', 'أنى', 'آه', 'آها', 'أو', 'أولاء', 'أولئك', 'أوه', 'آي', 'أي', 'أيها', 'إي', 'أين', 'أين', 'أينما', 'إيه', 'بخ', 'بس', 'بعد', 'بعض', 'بك', 'بكم', 'بكم', 'بكما', 'بكن', 'بل', 'بلى', 'بما', 'بماذا', 'بمن', 'بنا', 'به', 'بها', 'بهم', 'بهما', 'بهن', 'بي','بسبب', 'بين', 'بيد',
            'تلك', 'تلكم', 'تلكما', 'ته', 'تي', 'تين', 'تينك',
            'ثم', 'ثمة', 'حاشا', 'حبذا', 'حتى', 'حيث', 'حيثما',
            'حين', 'خلا', 'دون', 'ذا', 'ذات', 'ذاك', 'ذان', 'ذانك', 'ذلك',
            'ذلكم', 'ذلكما', 'ذلكن', 'ذه', 'ذو', 'ذوا', 'ذواتا',
            'ذواتي', 'ذي', 'ذين', 'ذينك', 'ريث', 'سوف', 'سوى', 'شتان', 'عدا',
            'عسى', 'عل', 'على', 'عليك', 'عليه', 'عما', 'عن', 'عند', 'غير',
            'فإذا', 'فإن', 'فلا', 'فمن', 'في', 'فيم', 'فيما', 'فيه', 'فيها',
            'قد', 'كأن', 'كأنما', 'كأي', 'كأين', 'كذا', 'كذلك', 'كل', 'كلا',
            'كنا','كلاهما', 'كلتا', 'كلما', 'كليكما', 'كليهما', 'كم', 'كم', 'كما',
            'كي', 'كيت', 'كيف', 'كيفما', 'لا', 'لاسيما', 'لدى', 'لست', 'لستم',
            'لستما', 'لستن', 'لسن', 'لسنا', 'لعل', 'لك', 'لكم', 'لكما',
            'لكن', 'لكنما', 'لكي', 'لكيلا', 'لم', 'لما', 'لن', 'لنا',
            'له', 'لها', 'لهم', 'لهما', 'لهن', 'لو', 'لولا', 'لوما',
            'لي', 'لئن', 'ليت', 'ليس', 'ليسا', 'ليست', 'ليستا', 'ليسوا', 'ما',
            'ماذا', 'متى', 'مذ', 'مع', 'مما', 'ممن', 'من', 'منه', 'منها', 'منذ',
            'مه', 'مهما', 'نحن', 'نحو','مش', 'نعم', 'ها', 'هاتان','رح', 'هاته', 'هاتي',
            'هاتين', 'هاك', 'هاهنا', 'هذا', 'هذان', 'هذه', 'هذي', 'هذين', 'هكذا',
            'هل', 'هلا', 'هم', 'هما', 'هن', 'هنا', 'هناك', 'هنالك', 'هو', 'هؤلاء',
            'هي', 'هيا', 'هيت', 'هيهات', 'والذي', 'والذين', 'وإذ', 'وإذا', 'وإن',
            'ولا', 'ولكن', 'ولو', 'وما', 'ومن', 'وهو', 'يا' , 'من' , 'على', 'الى','هما', 'مع', 'هذه', 'التي', 'كما ', 'ذلك ', 'لذا', 'عن', 'في','ان','كان','كانت','الى','قبل','أنه','تم'
            ,'وقال','قال','فى','وقد','قد','ولم','وذلك','ذلك','يكون','او','وهذه','وهي ','وبعد','وهذا','عندها','جدا','بأن','انه','الي']
@sst.cache
def cleaner(tweet,lang):
    if lang == 'ar':
        tweet = " ".join([word for word in tokenize(tweet,conditions=is_arabicrange) if not word in stop] )
        
    else:
        #nltk.download('words')
        words = set(nltk.corpus.words.words())
        tweet = re.sub("@[A-Za-z0-9]+","",tweet)
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) 
        tweet = " ".join(tweet.split())
        tweet = tweet.replace("#", "").replace("_", " ") 
        tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    if w.lower() in words or not w.isalpha())
    return tweet
@sst.cache
def anlalyseTheTweets(df):
    #nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    nltk.download('words')
    words = set(nltk.corpus.words.words())
    list  = []
    for tweet in df['tweet']:
        list.append((sid.polarity_scores(str(tweet)))['compound'])
    df['sentiment'] = pd.Series(list)
    return df

def sentiment_category(sentiment):
    label = ''
    if(sentiment>0):
        label = 'positive'
    elif(sentiment == 0):
        label = 'neutral'
    else:
        label = 'negative'
    return(label)

def getWordCloud(df,lang):
    allTweets = ' '.join([twts for twts in df['tweet']])
    if lang == 'ar':
        awc = ArabicWordCloud(background_color="white")
        t = allTweets
        wc = awc.from_text(t)
        fig = plt.figure(figsize=(20,10))
        plt.imshow(wc, interpolation='bilinear')  
        plt.axis('off')
        plt.tight_layout(pad =0)
    else:
        wordCloud =  WordCloud(width=1200, height=600, colormap='Spectral').generate(allTweets)
        fig = plt.figure(figsize=(20,10))
        plt.imshow(wordCloud, interpolation='bilinear')  
        plt.axis('off')
        plt.tight_layout(pad =0)

    return fig

