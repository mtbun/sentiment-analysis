import pandas as pd
import numpy as np


names=["happy :0","sad:1","exciting:2","angry:3"]
happy_path ="raw_datas/happy.csv"
hp =pd.read_csv(happy_path)


sd_path ="raw_datas/sad.csv"
sd =pd.read_csv(sd_path)


an_path ="raw_datas/angry.csv"
an =pd.read_csv(an_path)


ex_path ="raw_datas/exciting.csv"
ex =pd.read_csv(ex_path)

sen_list =[]
sen_list_label =[]
for i in hp["tweet"]:
    sen_list.append(i)
    sen_list_label.append(0)

for i in sd["tweet"]:
    sen_list.append(i)
    sen_list_label.append(1)

for i in ex["tweet"]:
    sen_list.append(i)
    sen_list_label.append(2)

for i in an["tweet"]:
    sen_list.append(i)
    sen_list_label.append(3)

print(len(hp)+len(ex)+len(an)+len(sd))
print(len(sen_list),len(sen_list_label))
array =np.column_stack((sen_list,sen_list_label))

data_frame = pd.DataFrame(array)
data_frame =data_frame.rename(columns={0:"Tweet",1:"Sentiment"})
names=("Tweet","Sentiment")

data_frame.to_csv('Data_Set.csv',index=False)