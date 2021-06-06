import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1-Calling the data from path
data_hpy ="happy.csv"
df_hp = pd.read_csv(data_hpy)

data_sad ="sad.csv"
df_sad = pd.read_csv(data_sad)



#2-Preparing the data
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

        for word in tweet_clean:
            stem_word = stemmer.stem(word)
            tweet_stm.append(stem_word)
    return tweet_stm


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




X , x_test, Y, y_test = train_test_split(tweet, label, test_size=0.01,random_state=45)

X = np.array(X).reshape(-1,1)
Y = np.array(Y).reshape(-1,1)

x_test = np.array(x_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)





def predict_tweet(tweet, freqs, theta):

	x = extract_features(tweet, freqs)

	y_pred = sigmoid(np.dot(x, theta))
	return y_pred[0][-1]



def test_logistic_regression(test_x, test_y, freqs, theta,should):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """


    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > should:
            y_hat.append(1)
        else:
            y_hat.append(0)


    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)


    return accuracy



def sigmoid(z):
	return 1 / (1 + np.exp(-z))



def gradientDescent(x, y, theta, alpha, num_iters):

	m = x.shape[0]
	per =[]
	should=0.49
	for i in range(0, num_iters):
		print("training iter {} of {}".format(i+1,num_iters))
		z = np.dot(x, theta)
		h = sigmoid(z)

		J = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
		theta = theta- (alpha / m) * np.dot(x.transpose(), (h - y))

		tmp_accuracy = test_logistic_regression(X, Y, feqs, theta,should)
		print(f"Logistic regression model's accuracy on train = {tmp_accuracy * 100}%")
		per.append(tmp_accuracy * 100)
		tmp_accuracy2 = test_logistic_regression(x_test, y_test, feqs, theta,should)
		print(f"Logistic regression model's accuracy on test = {tmp_accuracy2 * 100}%")
		print("*****************************")
		should += 0.0000028575

	print("en iyi sınır",should)
	J =  float(J)
	return J, theta, per


feqs = pd.read_pickle("data_freqs.pkl")


def extract_features(tweet, freqs):

	word_l = clean_data2(tweet)
	x = np.zeros((1, 3))

	x[0, 0] = 1

	for word in word_l:
		x[0, 1] += freqs.get((word, 1.0), 0)
		x[0, 2] += freqs.get((word, 0.0), 0)

	assert (x.shape == (1, 3))
	return x


train_x = np.zeros((len(X), 3))

for i in range(len(X)):
    train_x[i, :]= extract_features(X[i], feqs)

train_y = Y

#gradient descent

J, theta,per = gradientDescent(train_x, train_y, np.zeros((3, 1)), 0.000001, 3487)
np.save("weights_yeni.npy",theta)
theta3= np.load("weights_yeni.npy")


max_value = max(per)
max_index = per.index(max_value)
print(max_value,max_index)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[t for t in np.squeeze(theta)]}")

plt.xlabel("tekrarlama numarası")
plt.ylabel("doğruluk")
plt.title("A test graph")
plt.plot(per)
plt.legend()
plt.show()
