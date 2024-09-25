import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import multiprocessing
import logging
from node2vec import Node2Vec
from gensim.models.callbacks import CallbackAny2Vec

logger = logging.getLogger(__name__)


if __name__ =="__main__":
    # tags = pd.read_csv('./ml-latest-small/tags.csv')
    df_movies = pd.read_csv('./ml-latest-small/movies.csv')
    ratings = pd.read_csv('./ml-latest-small/ratings.csv')
    tags = pd.read_csv('./ml-latest-small/tags.csv')

    df_movies['movieId'] = df_movies['movieId'].astype(str)

    ratings['userId'] = ratings['userId'].astype(str)
    ratings['movieId'] = ratings['movieId'].astype(str)

    tags['movieId'] = tags['movieId'].astype(str)
    tags['userId'] = tags['userId'].astype(str)


    tags['tag'] = tags['tag'].apply(lambda x: x.lower())
    xs = tags.groupby('movieId')['tag'].apply(list)
    xs = xs.apply(lambda x: list(set(x)))
    df_tags_merge = pd.DataFrame(xs,columns=['tag'])
    dict_tags_merge = df_tags_merge.to_dict(orient = 'index')