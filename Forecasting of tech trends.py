
import numpy as np
import pandas as pd
import gensim #the library for Topic modelling
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models
import warnings
warnings.simplefilter('ignore')

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

path = r"/content/drive/My Drive/LDA"

import os

dpath=os.listdir(path)

dirname=dpath[0]

dirname =[]
for i in range(0,len(dpath)):
  dirname.append(path + '/' + dpath[i])

dirname

data=dirname[1]

df = pd.read_csv(data,encoding='latin1')
df

df.info()

df['date']=pd.to_datetime(df['date'],errors='coerce')

df.info()

df.head()

idx = df[df['date'].isnull()].index
idx

df.dropna(inplace=True)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
my_stop_words = text.ENGLISH_STOP_WORDS.union(["million","raises","startup","funds","ceo","billion","startups","funding","york","billions"])
tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=my_stop_words)
doc_term_matrix = tfidf_vect.fit_transform(df['title'].values.astype('U'))

from sklearn.decomposition import NMF

nmf = NMF(n_components=5, random_state=42)
nmf.fit(doc_term_matrix )

import random

for i in range(10):
    random_id = random.randint(0,len(tfidf_vect.get_feature_names()))
    print(tfidf_vect.get_feature_names()[random_id])

first_topic = nmf.components_[0]
top_topic_words = first_topic.argsort()[-10:]

for i in top_topic_words:
    print(tfidf_vect.get_feature_names()[i])

sec_topic = nmf.components_[1]
top_topic_words1 = sec_topic.argsort()[-40:]

for i in top_topic_words:
    print(tfidf_vect.get_feature_names()[i])

for i,topic in enumerate(nmf.components_):
    print(f'Top 10 words for topic #{i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

topic_values = nmf.transform(doc_term_matrix)
df['Topic_nmf'] = topic_values.argmax(axis=1)
df.head()

df['Topic_nmf'].value_counts()

df['MONTHLY'] = df['date'].map(lambda x: '{year}-{month}'.format(year=x.year,
                                                              month=x.month,
                                                              day=x.day))

df.head()

from google.colab import files

data=df.to_csv("plotdata.csv")
files.download("plotdata.csv")

a = df.groupby(['Topic_nmf','MONTHLY']).size()

a

a.dropna()

import numpy as np
import matplotlib.pyplot as plt
import pandas

a.head()

x=a.to_csv('abc.csv')

from google.colab import files
files.download('abc.csv')

"""TRY"""

zero=pd.read_csv(dirname[0])

zero['DATE']=pd.to_datetime(zero['DATE'])

zero.info()

zero.head()

series = pd.read_csv(dirname[0], \
                     header=0, \
                     parse_dates=[0], \
                     index_col=0, \
                     squeeze=True, \
)

series

import matplotlib.pyplot as plt

series.plot()
plt.show()

from statsmodels.tsa.arima_model import ARIMA

X = series.values
train, test = X[0:9], X[9:]
history = [x for x in train]
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%d, expected=%d' % (yhat, obs))

predyear = model_fit.forecast(steps=12)[0]

print(predyear.astype(int))

mape = np.mean(np.abs(yhat - obs)/np.abs(yhat))

mape

pred1 = model_fit.forecast(steps=1)[0]

print(pred1.astype(int))

pred3 = model_fit.forecast(steps=3)[0]

pred3.astype(int)

"""PLOT

Graph
"""

plotdata=pd.read_csv(dirname[2])

plotdata.head()

import seaborn as sns

sns.set_theme(style="darkgrid")

fig=plt.figure(figsize=(12,8))
ax = sns.countplot(x="MONTHLY",hue="Topic_nmf" ,data=plotdata)
plt.legend(title='Technologies', loc='upper left', labels=['App', 'Data','Mobile','Video','Self Driving Cars'])
plt.show()

fig=plt.figure(figsize=(15,10))
ax1 = sns.countplot(x="INTERVAL_15",hue="Topic_nmf" ,data=plotdata)
ax1.tick_params(direction='out',labelrotation=90)
plt.legend(title='Technologies', loc='upper left', labels=['App', 'Data','Mobile','Video','Self Driving Cars'])
plt.show()

fig=plt.figure(figsize=(15,10))
ax2 = sns.countplot(x="quarter",hue="Topic_nmf" ,data=plotdata)
plt.legend(title='Technologies', loc='upper left', labels=['App', 'Data','Mobile','Video','Self Driving Cars'])
plt.show()