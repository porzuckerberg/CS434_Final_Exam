import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns, numpy as np
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import time

anime = pd.read_csv('https://raw.githubusercontent.com/nestor94/animender/master/data/anime.csv')
anime.tail(5)
anime.info()
anime.duplicated().sum()
anime.isna().sum()
(anime.isna().sum()/anime.shape[0]).round(2)


df = anime.copy()
df.dropna(axis=0 ,inplace=True)
df.isna().sum()
df.reset_index(drop=True,inplace=True)
df.tail()

mapping = {'Sci-Fi':'SciFi','Slice of Life':'SliceOfLife','Martial Arts':'MartialArts','Super Power':'SuperPower','Shounen Ai':'ShounenAi','Shoujo Ai':'ShoujoAi'}
ndf = df.copy()
ndf['genre'] = df['genre'].replace(mapping,regex=True) 


def countgenre(data):
  dicts = {}
  for x in data:
    for y in x:
      if y not in dicts.keys():
        dicts[y] = 0
      dicts[y] = int(dicts[y])+1
  return dicts

GENRE = 'genre'
GENRE = df[GENRE].str.split(", ") 
GENRE = countgenre(GENRE)
GENRE

colGenre = pd.DataFrame(data=GENRE.keys(),columns=['genre'])
rowGenre = pd.DataFrame(data=GENRE.values(),columns=['count_genre'])
colGenre.reset_index(inplace=True)
rowGenre.reset_index(inplace=True)
Genre = pd.merge(colGenre,rowGenre,on=['index','index'])
Genre

alt.Chart(Genre, title='Anime Genre').mark_bar().encode(
    x='count_genre:Q',
    y=alt.Y('genre:O', sort='-x'),
    tooltip=list(Genre.columns)
).properties(width=900,height=600)


top10=df[['name', 'members']].sort_values(by = 'members',ascending = False).head(10)
plt.figure(figsize=(15,8))
ax=sns.barplot(x="name", y="members", data=top10, palette="gnuplot2")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40, ha="right")
ax.set_title('Top 10 Anime based on members',fontsize = 22)
ax.set_xlabel('Anime',fontsize = 20) 
ax.set_ylabel('Community Size', fontsize = 20)

df.sort_values(by = 'members',ascending = False).head(10)

plt.figure(figsize=(10,10))
sns.histplot(df, x='type');

plt.figure(figsize=(10,10))
sns.distplot(df['rating']);

tf_idf = TfidfVectorizer(lowercase=True, stop_words = 'english')

tf_idf_matrix = tf_idf.fit_transform(ndf['genre'])
tf_idf_matrix.shape
space = tf_idf.vocabulary_
print(space)

pd.DataFrame(tf_idf_matrix.toarray(),columns=tf_idf.get_feature_names())


class PipeLine():
    def __init__(self):       
        self.mapping = {'Sci-Fi':'SciFi','Slice of Life':'SliceOfLife','Martial Arts':'MartialArts','Super Power':'SuperPower','Shounen Ai':'ShounenAi','Shoujo Ai':'ShoujoAi'}

    def dropRows(self, data):
        data.dropna(axis=0 ,inplace=True)
        return data

    def reset_index(self,data):
        data.reset_index(drop=True,inplace=True)
        return data

    def genre_correction(self, data):
        data['genre'] = data['genre'].replace(self.mapping,regex=True)
        return data

    def Tf_idf(self,data):
        tf_idf = TfidfVectorizer(lowercase=True, stop_words = 'english')
        tf_idf_matrix = tf_idf.fit_transform(data['genre'])
        return tf_idf_matrix

    def execution(self,data):
        df = data.copy()
        df = self.dropRows(df)
        df = self.reset_index(df)
        df = self.genre_correction(df)
        return self.Tf_idf(df)

pipe = PipeLine()

tf_idf_matrix = pipe.execution(anime)

linear_ker = linear_kernel(tf_idf_matrix, tf_idf_matrix)

def recommendations(name, linear_ker = linear_ker):
    indices = pd.Series(df.index, index = df['name'])
    similarity_scores = list(enumerate(linear_ker[indices[name]]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[:11]
    anime_indices = [i[0] for i in similarity_scores]
    a = df[['anime_id','name','genre','type']].iloc[anime_indices]
    b = a.loc[a['name'] == name]
    a['match_score'] = [i[1] for i in similarity_scores]
    a['match_score'] = a['match_score'].apply(lambda x: round(x,4))
    a.drop(labels=indices[name],axis=0,inplace=True)  
    return b,a


#!pip install gradio -q
import gradio as gr

iface = gr.Interface(recommendations,inputs=[gr.inputs.Textbox(label='ANIME NAME', default="Aura: Maryuuin Kouga Saigo no Tatakai")],outputs=[gr.outputs.Dataframe(label='YOUR ANIME'),gr.outputs.Dataframe(label='SIMILAR ANIME')])
iface.launch()

"""
Reference
https://www.zhihu.com/question/19746144
https://www.datacamp.com/community/tutorials/recommender-systems-python
https://www.kaggle.com/lavanyaanandm/recommending-anime-s-using-all-recommendation-sys#10.-Reference-
"""