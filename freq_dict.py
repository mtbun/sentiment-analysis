import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
import pickle

#1-Calling the data from path
data_hpy ="raw_datas/happy.csv"
df_hp = pd.read_csv(data_hpy)

data_sad ="raw_datas/sad.csv"
df_sad = pd.read_csv(data_sad)

data_ex ="raw_datas/exciting.csv"
df_ex = pd.read_csv(data_ex)

data_an ="raw_datas/sad.csv"
df_an = pd.read_csv(data_an)


#2-Preparing the data
def clean_data(df):
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    tweet_stm = []
    for tweet in df["tweet"]:
        # 2.1 remove old style tweets
        tweet = re.sub(r'^RT[/s]+', '', tweet)
        # 2.2 remove links
        tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, flags=re.MULTILINE)
        # 2.3 remove hashtages (only the # mark)
        tweet = re.sub(r'#', '', tweet)
        # 2.4 instantaite tokenizer class
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        # 2.4.1 tokenize tweets
        tweet_tok = tokenizer.tokenize(tweet)
        # 2.5 remove stopwrods and punctuation
        # to see the stop words
        tweet_clean = []
        for word in tweet_tok:
            if (word not in stopwords_en and word not in string.punctuation):
                tweet_clean.append(word)
        # stem tweets
        for word in tweet_clean:
            stem_word = stemmer.stem(word)
            tweet_stm.append(stem_word)
    return tweet_stm


hpy = clean_data(df_hp)
sad = clean_data(df_sad)
ex = clean_data(df_ex)
an = clean_data(df_an)

all_tw = hpy+sad+ex+an

def freq(tweets,label):
    ylis = np.squeeze(label).tolist()
    freqs = {}
    for y,tweet in zip(ylis,tweets):
        pair = (tweet,y)
        if pair in freqs:
            freqs[pair]+=1
        else:
            freqs[pair]  =1
    return freqs

label_1=np.append(np.zeros((len(hpy))),np.ones((len(sad))))
label_2 = np.append(np.ones((len(ex)))*2,np.ones((len(an)))*3)
labels = np.append(label_1,label_2)
frq = freq(all_tw,labels)


a_file = open("data.pkl", "wb")
pickle.dump(frq, a_file)
a_file.close()