import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
import pickle

#Calling the data
data_hpy ="happy.csv"
df_hp = pd.read_csv(data_hpy)

data_sad ="sad.csv"
df_sad = pd.read_csv(data_sad)


#Preparing the data
def clean_data(df):
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    tweet_stm = []
    for tweet in df["tweet"]:
        tweet = re.sub(r'^RT[/s]+', '', tweet)
        tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'#', '', tweet)
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tok = tokenizer.tokenize(tweet)

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

all_tw = hpy+sad

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

labels=np.append(np.zeros((len(hpy))),np.ones((len(sad))))
frq = freq(all_tw,labels)


a_file = open("data_freqs.pkl", "wb")
pickle.dump(frq, a_file)
a_file.close()
