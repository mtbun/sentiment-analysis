import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string

from sklearn.model_selection import train_test_split

data_hpy ="happy.csv"
df_hp = pd.read_csv(data_hpy)

data_sad ="sad.csv"
df_sad = pd.read_csv(data_sad)



def clean_data(df):
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    tweet_stm = []
    for tweet in df["tweet"]:
        tweet = re.sub(r'^RT[/s]+', '', tweet)
        tweet = re.sub(r"’", '', tweet)
        tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'#', '', tweet)
        tweet = re.sub(r"’", '', tweet)
        tweet = re.sub(r'…', '', tweet)
        tweet = re.sub(r':', '', tweet)
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

tweet =[]
label =[]

noise=['c','cri','la','away','15','loser','clip','ea','una','oh','tri','huidig','took','maravilla','<3','genshin','even',
	   'knock',"i'm",'🎈','ahh','1','queen','dan','esto','guy','emocionars','im','man','colleg','ezay','🏼','never','old','котором',
	   'こぼれた粉ものの片づけ','pep','preciou','issu','birthday','gif','olanları',"can't",'lol','sa','ship','2','но','pretti',
	   'sophia','hood','thought','omg','like','jensen','temp','попались','wish','advanc','want',"i'v",'that','bea','late','th','“',
	   "we'r",'bh','main','game','ali','bless','🥺','say','stood','song','final','средний','lo','🥵','end','peopl','n','축하','😉',
	   '✊','feel','sound','bojom','u','😳','suv','fuck','‘','etc','justifi','good','part','lot','mainli','...','year','car',
	   'querida','prayer','sunshin','miss','mine','big','six','letter','elderli','one','tuesday','challeng',"they'r",'gosh',
	   'declar','imm','dediğimiz','к','episod','decid','install','leav','svih','exposur','😭','🎉','memori','samo','wors','sooo',
	   'richard','ti','pleas','care','ㅤin','women','nake','','today','feder','estar','still','al','zach','make','need','čuvamo','😍',
	   'eye','doesnt','🥳','get','babe','earn','6','i','não','-','3','lov','offic','😔','ﾟ','cm','sanjivinsmok','ovo','would','age',
	   'trobojka','conflict','use','ye','photo','yoonkook','fic','imagin','anniversari','😩','target','🙁','ian','finish','south',
	   'finish','está','gonna','chiquita','maximum','pra','💛','know','k','meet','die','system','atackam','ispr','fill','let',
	   'celebr','republican','pagi','ovih']

for i in range(4200):
	if hpy[i] not in noise :
		if hpy[i] != 'sad':
			tweet.append(hpy[i])
			label.append(1)
sad_noise = ['great','happi','happiest']
for i in range(4200):
	if sad[i] not in noise :
		if sad[i] not in sad_noise:
			tweet.append(sad[i])
			label.append(0)




tweet = np.array(tweet)
label = np.array(label)




X , x_test, Y, y_test = train_test_split(tweet, label, test_size=0.001,random_state=32)

X = np.array(X).reshape(-1,1)
Y = np.array(Y).reshape(-1,1)

x_test = np.array(x_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)



def test_logistic_regression(test_x, test_y, freqs, theta):


    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5000012500000302:
            y_hat.append(1)
        else:
            y_hat.append(0)


    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)

    return accuracy


feqs = pd.read_pickle("data_freqs.pkl")


def clean_data2(tweet):
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    tweet_stm = []
    for tweet in tweet:
        tweet = re.sub(r'^RT[/s]+', '', tweet)
        tweet = re.sub(r"’", '', tweet)
        tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'#', '', tweet)
        tweet = re.sub(r"’", '', tweet)
        tweet = re.sub(r'…', '', tweet)
        tweet = re.sub(r':', '', tweet)
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


def extract_features(tweet, freqs):

	word_l = clean_data2(tweet)

	x = np.zeros((1, 3))

	x[0, 0] = 1

	for word in word_l:
		x[0, 1] += freqs.get((word, 1.0), 0)

		x[0, 2] += freqs.get((word, 0.0), 0)

	assert (x.shape == (1, 3))
	return x




def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def predict_tweet(tweet, freqs, theta):

	x = extract_features(tweet, freqs)
	y_pred = sigmoid(np.dot(x, theta))
	return y_pred[0][-1]


theta3= np.load("weights.npy")


for tweet in ["great day ",
				"happy birthday dady",
			  "my friend died",
				"very bad day",
				"i love this place"
			  ]:

	sonuc = predict_tweet(tweet, feqs, theta3)
	if sonuc >= 0.44056:
		print(tweet, " -> happy")
	else:
		print(tweet, " -> sad")

tmp_accuracy = test_logistic_regression(x_test, y_test, feqs, theta3)
print(f"modelin doğruluğu = {tmp_accuracy * 100}%")
