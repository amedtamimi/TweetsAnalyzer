import streamlit as sst
import nltk
import twint
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyarabic.araby import tokenize,is_arabicrange,strip_tashkeel
from ar_wordcloud import ArabicWordCloud
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import asyncio
import pymysql


data = pd.read_csv('AJGT.csv')


def model(predect):
    data = pd.read_csv('AJGT.csv')
    feature = data.Feed
    target = data.Sentiment
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size =.2, random_state=100)
    pipe = make_pipeline(TfidfVectorizer(),MultinomialNB())
    pipe.fit(X_train,Y_train)
    pipe.predict(X_test)
    X_new = [predect]
    neww_prediction = pipe.predict(X_new)
    return neww_prediction[0]


def featchData(hashtag_name,fromDate,numberOfLikes,languges):
    tweets = twint.Config
    tweets.Search = hashtag_name
    tweets.Limit = 100
    tweets.Min_likes =numberOfLikes
    tweets.Since = fromDate
    tweets.Database = True
    twint.run.Search(tweets)
    twint.run.Search(tweets)

    con= pymysql.connect(host="containers-us-west-169.railway.app",user=sst.secrets["db_username"],password=sst.secrets["db_password"],database="twitter",charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor,port=6589)
    try:
        with con.cursor() as cur:
                cur.execute('SELECT user_id,screen_name,tweet,lang,likes_count FROM tweets')
                rows = cur.fetchall()
                df_tweets = pd.DataFrame(rows)
                df_tweets = df_tweets[df_tweets['lang'] == languges]

    finally:
        con.cursor().execute('DELETE FROM tweets')
        con.commit()
        con.close()   

    return df_tweets  




stop = ['انت','إذ', 'إذا', 'إذما', 'إذن', 'أف', 'أقل', 'أكثر', 'ألا', 'إلا', 'التي', 'الذي', 'الذين', 'اللاتي', 'اللائي', 'اللتان', 'اللتيا', 'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'إلى', 'إليك', 'إليكم', 'إليكما', 'إليكن', 'أم', 'أما', 'أما', 'إما', 'أن', 'إن', 'إنا', 'أنا', 'أنت', 'أنتم', 'أنتما', 'أنتن', 'إنما', 'إنه', 'أنى', 'أنى', 'آه', 'آها', 'أو', 'أولاء', 'أولئك', 'أوه', 'آي', 'أي', 'أيها', 'إي', 'أين', 'أين', 'أينما', 'إيه', 'بخ', 'بس', 'بعد', 'بعض', 'بك', 'بكم', 'بكم', 'بكما', 'بكن', 'بل', 'بلى', 'بما', 'بماذا', 'بمن','انو', 'بنا', 'به', 'بها', 'بهم', 'بهما', 'بهن', 'بي','بسبب', 'بين', 'بيد',
            'تلك','انك', 'تلكم', 'تلكما','تكون','اشي', 'ته', 'تي', 'تين', 'تينك',
            'ثم', 'ثمة', 'حاشا', 'حبذا', 'حتى', 'حيث', 'حيثما',
            'حين', 'خلا', 'دون', 'ذا', 'ذات', 'ذاك', 'ذان', 'ذانك', 'ذلك',
            'ذلكم', 'ذلكما','حد', 'ذلكن', 'ذه', 'ذو', 'ذوا', 'ذواتا',
            'ذواتي', 'ذي', 'ذين', 'ذينك', 'ريث', 'سوف', 'سوى', 'شتان', 'عدا',
            'عسى', 'عل', 'على', 'عليك', 'عليه', 'عما', 'عن', 'عند', 'غير',
            'فإذا','شو', 'فإن', 'فلا', 'فمن', 'في', 'فيم', 'فيما', 'فيه', 'فيها',
            'قد', 'كأن', 'كأنما', 'كأي', 'كأين', 'كذا', 'كذلك', 'كل', 'كلا',
            'كنا','كلاهما', 'كلتا', 'كلما', 'كليكما', 'كليهما', 'كم', 'كم', 'كما',
            'كي', 'كيت', 'كيف', 'كيفما', 'لا', 'لاسيما', 'لدى', 'لست', 'لستم',
            'لستما', 'لستن', 'لسن', 'لسنا','فعلا', 'لعل', 'لك', 'لكم', 'لكما',
            'لكن', 'لكنما', 'لكي', 'لكيلا', 'لم', 'لما', 'لن', 'لنا',
            'له', 'لها', 'لهم', 'لهما', 'لهن', 'لو', 'لولا', 'لوما',
            'لي', 'لئن', 'ليت', 'ليس', 'ليسا', 'ليست', 'ليستا', 'ليسوا', 'ما',
            'ماذا', 'متى', 'مذ', 'مع', 'مما', 'ممن', 'من', 'منه', 'منها', 'منذ',
            'مه', 'مهما', 'نحن', 'نحو','مش', 'نعم', 'ها', 'هاتان','رح', 'هاته', 'هاتي',
            'هاتين', 'هاك', 'هاهنا', 'هذا', 'هذان', 'هذه', 'هذي', 'هذين', 'هكذا',
            'هل', 'هلا', 'هم', 'هما', 'هن', 'هنا', 'هناك', 'هنالك', 'هو', 'هؤلاء',
            'هي', 'هيا', 'هيت', 'هيهات','اجعل', 'والذي', 'والذين', 'وإذ', 'وإذا', 'وإن',
            'ولا', 'ولكن', 'ولو', 'وما', 'ومن', 'وهو', 'يا' , 'من' , 'على', 'الى','هما', 'مع', 'هذه', 'التي', 'كما ', 'كنت','ذلك ', 'لذا', 'عن', 'في','ان','كان','كانت','وفي','شيء','الى','قبل','أنه','تم'
            ,'وقال','قال','فى','وقد','قد','ولم','وذلك','ذلك','يكون','او','وهذه','وهي ','وين','وبعد','لان','وهذا','عندها','جدا','بأن','انه','الي']



def cleaner(tweet,lang):
    if lang == 'ar':
        tweet = " ".join([word for word in tokenize(tweet,conditions=is_arabicrange,   morphs=strip_tashkeel) if not word in stop] )
        tweet = re.sub("[إأآا]", "ا", tweet)
        tweet = re.sub("ى", "ي", tweet)
        tweet = re.sub("ؤ", "ء", tweet)
        tweet = re.sub("ئ", "ء", tweet)
        tweet = re.sub("ة", "ه", tweet)
        tweet = re.sub("گ", "ك", tweet)
        
    else:
        nltk.download('words')
        words = set(nltk.corpus.words.words())
        tweet = re.sub("@[A-Za-z0-9]+","",tweet)
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) 
        tweet = " ".join(tweet.split())
        tweet = tweet.replace("#", "").replace("_", " ") 
        tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    if w.lower() in words or not w.isalpha())

    return tweet



def anlalyseTheTweets(df ,lang):
    list  = []
    if lang == 'ar':
        list = []
        for tweet in df['tweet']:
            list.append(model(tweet))

    else:
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        nltk.download('words')
        words = set(nltk.corpus.words.words())

        for tweet in df['tweet']:
            list.append((sid.polarity_scores(str(tweet)))['compound'])

    df['sentiment'] = pd.Series(list)

    return df


def sentiment_category(sentiment):
        label = ''

        if(sentiment>0):
            label = 'Positive'

        elif(sentiment == 0):
            label = 'Neutral'

        else:
            label = 'Negative'

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
        wordCloud =  WordCloud(width=1200, height=600,background_color="white").generate(allTweets)
        fig = plt.figure(figsize=(20,10))
        plt.imshow(wordCloud, interpolation='bilinear')  
        plt.axis('off')
        plt.tight_layout(pad =0)

    return fig


def barblot(df,lang):
    fig = plt.figure(figsize=(20,10))
    if lang  == 'ar':
        sns.countplot(x=df['sentiment'])

    else:
        sns.countplot(x=df['Label'])

    plt,plt.plot

    return fig