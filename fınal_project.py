# -*- coding: utf-8 -*-

# -- Sheet --

# 1. Coffeeshop Review_Text'lerine Göre Tavsiye Geliştirme
##########################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

from string import digits

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("/data/notebook_files/raw_yelp_review_data.csv", encoding= "utf-8",  delimiter=',', skipinitialspace=True )
df.head()
d.shape

# çift olanları çıkartmak :

df.loc[df["coffee_shop_name"] == "Caffé Medici ", ["coffee_shop_name"]] = "Caffe Medici "
df.loc[df["coffee_shop_name"] == "Lola Savannah Coffee Downtown ", ["coffee_shop_name"]] = "Lola Savannah Coffee Lounge "
df.loc[df["coffee_shop_name"] == "Summer Moon Coffee Bar ", ["coffee_shop_name"]] = "Summermoon Coffee Bar "

df.head()

#tarih dahil tüm sayıları silme
for i in df.index :
    remove_digits = str.maketrans('', '', digits)
    df["full_review_text"][i] = df["full_review_text"][i].translate(remove_digits)
    df["full_review_text"][i] = df["full_review_text"][i].lstrip(df["full_review_text"][i][0:3])
df.head()
df["star_rating"][i] = df["star_rating"][i].rstrip(df["star_rating"][i][3:])

#rating leri float a çevirmek:
df["star_rating"] = df["star_rating"].astype(float)
df["star_rating"].info()

#spaceleri silme
df['coffee_shop_name'] = df['coffee_shop_name'].str.strip()

#tüm yorumları küçük harf yapma
df['full_review_text'] = df['full_review_text'].str.lower()
df.head()

#bu df i group by a sokup indirgemeden önce yapılmalıydı, görselleştirme için gerkli
df_wordcloud = df.copy()

#stop words kaldıma
stop_words = stopwords.words("english") + ["check-in", "check-ins"]
df["full_review_text"] = df["coffee_shop_description"] = df["full_review_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#noktalamaları çıkart:
df["coffee_shop_description"] = df["coffee_shop_description"].str.replace(r'[^\w\s]+', '')
df1 = df.copy()  #görselleştirme için group by'a almadan kopyalandı

#coffe_shop_description değişkeni oluşturma, burada değişken fazlalığı olmaması için yine df olarak kaydediyorum yeni dataframei
#eski df kayboluyor ama ona gerek yok zaten
df = df.groupby(["coffee_shop_name"]).agg({"coffee_shop_description": "sum"}).reset_index()
df.head()
df.shape  #(76, 2)
df.head(76)

# 2. TF-IDF Matrisinin Oluşturulması
#################################################

df.head()
df.shape


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['full_review_text'])

tfidf_matrix.shape
# (79, 17566)
# Satırlardaki bizim yorumlarımız
# Sütunlarda ise eşsiz kelimeler vardır


# Sütunlarımızdaki tüm featurelerimiz geldi
tfidf.get_feature_names()

# Bu sütunlardaki kelimeleri /featureleri scorlaştıralım.
# Dokümanlar ile terimlerin kesişimlerindeki scorlar
tfidf_matrix.toarray()



# 3. Cosine Similarity Matrisinin Oluşturulması
#################################################

# Bu matrix ile her bir kafenin diğer kafelerle benzerliği var.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
cosine_sim[1]
cosine_sim[5]

# 4. Benzerliklere Göre Café Önerilerin Yapılması
#################################################

indices = pd.Series(df.index, index=df['coffee_shop_name'])
indices.index.value_counts()

# Café Shop İsimlerindeki Çoklamaları Silelim, son güncel yorumlara göre
indices = indices[~indices.index.duplicated(keep='last')]


#indices["The Steeping Room "]

cafe_index = indices["Caffe Medici "]
cosine_sim[cafe_index]


similarity_scores = pd.DataFrame(cosine_sim[cafe_index], columns=["score"])

cafe_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['coffee_shop_name'].iloc[cafe_indices]
"""

## 4. Verin Analizi ve Görselleştirilmesi
#################################################

### numerik verilerin analizi (tek numerik veri:star_rating_num):
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel("count")
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df1, "star_rating_num", plot=True)

### heat map :
corr = pd.DataFrame(cosine_sim)
sns.heatmap(corr.corr(), xticklabels=True, yticklabels=True, cmap="viridis", vmin=-1, vmax=1, center= 0, square=True)

plt.show()

# Kişilerin verdiği puan dağılımı - pie chart 

df1.groupby(['star_rating']).sum().plot(
	kind='pie', y='star_rating_num', autopct='%1.0f%%')
plt.show()




##Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# Apply lambda function to get compound scores.
function = lambda title: vader.polarity_scores(title)['compound']
df1['compound'] = df1['coffee_shop_description'].apply(function)
df1.head(5)

#Word cloud visualization.
from wordcloud import WordCloud

allWords = ' '.join([twts for twts in df_wordcloud['full_review_text']])

wordCloud = WordCloud(colormap= "flare", background_color="white", contour_color ="yellow",
                      width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


def getAnalysis(score):
 if score < 0:
    return 'Negative'
 elif score == 0:
    return 'Neutral'
 else:
    return 'Positive'

df1['coffee_shop_description'] = df1['compound'].apply(getAnalysis)

df1.head(5)

#Visualize the counts for each sentiment type.
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df1['coffee_shop_description'].value_counts().plot(kind = 'bar')
plt.show()

#pie chart:
df1.coffee_shop_description.value_counts().plot(kind='pie', autopct='%1.0f%%',  fontsize=8, figsize=(9,6), colors=["purple", "yellow", "pink"])
plt.ylabel(" ", size=12)
plt.xlabel("Coffeeshop Reviews Sentiment ", size=12)
plt.show()

## 5. Çalışma Scriptinin Hazırlanması
#################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import digits
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from wordcloud import WordCloud

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r'C:\Users\burcu\OneDrive\Masaüstü\DS Miiul\Project\raw_yelp_review_data.csv', encoding='utf-8', delimiter=',', skipinitialspace=True)

df.loc[df["coffee_shop_name"] == "Caffé Medici ", ["coffee_shop_name"]] = "Caffe Medici "
df.loc[df["coffee_shop_name"] == "Lola Savannah Coffee Downtown ", ["coffee_shop_name"]] = "Lola Savannah Coffee Lounge "
df.loc[df["coffee_shop_name"] == "Summer Moon Coffee Bar ", ["coffee_shop_name"]] = "Summermoon Coffee Bar "

for i in df.index :
    remove_digits = str.maketrans('', '', digits)
    df["full_review_text"][i] = df["full_review_text"][i].translate(remove_digits)
    df["full_review_text"][i] = df["full_review_text"][i].lstrip(df["full_review_text"][i][0:3])
    #stringlerde sondan başlayıp eleman silme
    df["star_rating"][i] = df["star_rating"][i].rstrip(df["star_rating"][i][3:])

df["star_rating_num"] = df["star_rating"].astype(float)

df['coffee_shop_name'] = df['coffee_shop_name'].str.strip()

df['full_review_text'] = df['full_review_text'].str.lower()

df_wordcloud = df.copy()

stop_words = ["check-in", "check-ins"] + stopwords.words("english")
df["full_review_text"] = df["coffee_shop_description"] = df["full_review_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

df["coffee_shop_description"] = df["coffee_shop_description"].str.replace(r'[^\w\s]+', '')
df1 = df.copy()

df = df.groupby(["coffee_shop_name"]).agg({"coffee_shop_description": "sum"}).reset_index()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel("count")
        plt.title(numerical_col)
        plt.show(block=True)

def content_based_recommender(coffee_shop_name, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['coffee_shop_name'])
    # title'ın index'ini yakalama
    cafe_index = indices[coffee_shop_name]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[cafe_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    cafe_indices = similarity_scores.sort_values("score", ascending=False)[1:4].index
    return dataframe['coffee_shop_name'].iloc[cafe_indices]


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['coffee_shop_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)

df1.groupby(['star_rating']).sum().plot(kind='pie', y='star_rating_num', autopct='%1.0f%%')
plt.show()

vader = SentimentIntensityAnalyzer()

function = lambda title: vader.polarity_scores(title)['compound']
df1['compound'] = df1['coffee_shop_description'].apply(function)

allWords = ' '.join([twts for twts in df_wordcloud['full_review_text']])

wordCloud = WordCloud(colormap= "flare", background_color="white", contour_color ="yellow",
                      width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


def getAnalysis(score):
 if score < 0:
    return 'Negative'
 elif score == 0:
    return 'Neutral'
 else:
    return 'Positive'

df1['coffee_shop_description'] = df1['compound'].apply(getAnalysis)

df1.head(5)


plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df1['coffee_shop_description'].value_counts().plot(kind = 'bar')
plt.show()


df1.coffee_shop_description.value_counts().plot(kind='pie', autopct='%1.0f%%',  fontsize=8, figsize=(9,6), colors=["purple", "yellow", "pink"])
plt.ylabel(" ", size=12)
plt.xlabel("Coffeeshop Reviews Sentiment ", size=12)
plt.show()



