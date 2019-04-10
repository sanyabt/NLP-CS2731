import numpy as np
import pandas as pd
from gensim.summarization.summarizer import summarize
from pytextrank import json_iter, parse_doc, pretty_print
from rouge import Rouge
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import os
stop_words = stopwords.words('english')
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class Grouping():

    def __init__(self):
        self.groups = {}
    
    def addTweetsToGroups(self):
        data = pd.read_csv("postProcessedText.csv")
        for col in data.columns[1:]:
            self.groups[col] = []
            for _ , row in data.loc[data[col] == 1].iterrows():
               self.groups[col].append(row.Tweet)
        
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def read_glove():
    word_embeddings = {}
    f = open(os.path.abspath(os.path.join("glove","glove.6B.300d.txt")),encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def create_csv_with_given_group(group_name, vect):
    csv = os.path.abspath(os.path.join("ranked_tweets_by_group",group_name + ".csv"))
    f = open(csv,"w+")
    f.write("rank,Tweet")
    f.write("\n")
    for i in vect:
        f.write(str(i[0])+","+i[1])
        f.write("\n")
    f.close()

def sort_Tweets_by_rank(dic, group_name, word_embeddings):
    reference = dic.groups[group_name]
    clean_sentences = [s.lower() for s in reference]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((300,))
        sentence_vectors.append(v)
    sim_mat = np.zeros([len(reference), len(reference)])
    for i in range(len(reference)):
        for j in range(len(reference)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(reference)), reverse=True)

if __name__ == "__main__":
    word_embeddings = read_glove()
    g = Grouping()
    g.addTweetsToGroups()
    v = sort_Tweets_by_rank(g, "trust", word_embeddings)
    create_csv_with_given_group("trust", v)

    