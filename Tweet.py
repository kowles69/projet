import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from numpy import nan
from math import log

class Tweets:
    # Initialisation des variables de la classe
    def __init__(self, date="",time="",username="", texte=""):
        self.username = username
        self.time=time
        self.date = date
        self.texte = texte
       

    def get_type(self):
        return self.type


    # Fonction qui renvoie le texte à afficher lorsqu'on tape str(classe)
    def __str__(self):
        return f"{self.texte}, par {self.username}, publié le {self.date} à {self.time}"
    
    
    stopwords = set(STOPWORDS)
#Ajouter des mots de francais à stopwords :
    for wds in ['http', 'https', 'www', 'fr', 'com', 'io', 'org', 'co', 'jo', 'edu', 'news', 'html', 'htm',\
            'github', 'youtube', 'google', 'blog', 'watch', 'de', 'le', 'la', 'en', 'sur', 'vous', 'les', \
           'ajouter', 'README','md', 'et', 'PROCESS', 'CMYK', 'des', 'chargement', 'playlists', 'endobj', \
           'obj','est', 'use', 'using', 'will', 'web', 'first','pour', 'du', 'une', 'que']:
      stopwords.add(wds)
 
    stopwords_fr_ntlk = set(nltk.corpus.stopwords.words('french'))
    stopwords_en_ntlk = set(nltk.corpus.stopwords.words('english'))
    stopwords_clean = [ l.lower() for l in list(stopwords.union(stopwords_fr_ntlk).union(stopwords_en_ntlk))]
    stopwords_clean[:50] + ['...'] 

#On applique tout ca a l'ensemble de données :
#cette fonction permet le nettoyage des données textuelles en éliminant les stopwords
    def words_cleaning(self,date,time,username,tweet):
     date_clean = nan_to_string(date)
     time_clean = nan_to_string(time)
     username_clean = nan_to_string(username)
     tweet_clean = nan_to_string(tweet)
     words = ' '.join([date_clean, time_clean, username_clean, tweet_clean])
     words_clean = re.sub('[^A-Za-z ]','', words)
     words_clean = re.sub('\s+',' ', words_clean)
     words_list = words_clean.split(' ')
     return ' '.join([w.lower() for w in words_list if w not in stopwords_clean])
 
#Cette fonction permet de calculer le nombre d'occurence d'un mot dans l'ensemble des tweets
    def word_count(self,doc,word):
        all_sentences = []
        for word in doc:
         all_sentences.append(word)
        lines = list()
        for line in all_sentences:    
          words = line.split()
        for w in words: 
         lines.append(w)
        i=0
        for w in lines : 
         if w == 'deserve':
          i =i+1
        return i  

#la fonction tf idf pur calculer le score qui correspond à l'importance d'un mot dans le corpus
    def tf(term, doc, normalize=True):
     doc = doc.lower().split()
     if normalize:
        return doc.count(term.lower()) / float(len(doc))
     else:
        return doc.count(term.lower()) / 1.0
 
    def idf(term, corpus):
     num_texts_with_term = len([True for text in corpus if term.lower()
                              in text.lower().split()])
     try:
        return 1.0 + log(float(len(corpus)) / num_texts_with_term)
     except ZeroDivisionError:
        return 1.0

    def tf_idf(term, doc, corpus):
      return tf(term, doc) * idf(term, corpus)        

    
   
df = pd.read_csv('balondor_tweets.csv')
df['words_string'] = np.vectorize(words_cleaning)(df['date'], \
                                                         df['time'], \
                                                         df['username'], \
                                                         df['tweet'])
#Instanciation de la classe Tweets : 
data = {}
data = data.fromkeys(range(20000),[])                                                     
for i in df.index:
 tw = Tweets(df["date"][i],df["time"][i],df["username"][i],df["tweet"][i])
 data[i].append(tw)
 

#Affichage du dictionnaire data qui contient toutes les instances Tweets 
for i in data.items():
    print(i)

#Execution de la fonction Tf Idf
terms = {}
 for cle, valeur in data.items():
     v = [i.lower() for i in data['cle'].split() ]
     terms[cle].append(v)

#Calculer le score Tf Idf pour les quatre mots :
QUERY_TERMS = ['deserve', 'steal', 'mérite','volé']

query_scores = ([i for i in data.keys()],0)
for term in [t.lower() for t in QUERY_TERMS]:
    for doc in sorted(data):
        score = tf_idf(term, data[doc], data.values())
        query_scores[doc] += score

print("Score TF-IDF total pour le terme '{}'".format(' '.join(QUERY_TERMS), ))

for (doc, score) in sorted(query_scores.items()):
    print(doc, score)
