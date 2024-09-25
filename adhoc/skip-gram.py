

import pandas as pd
import networkx as nx
import math
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import multiprocessing
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils import data as tud
from torch import nn

def get_time_block(timestamp):
    hour = timestamp.hour
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'midnight'


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, hidden):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden

        self.in_embedding = nn.Embedding(self.vocab_size, self.hidden)
        self.out_embedding = nn.Embedding(self.vocab_size, self.hidden)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embedding(input_labels)  # [batch, hidden]
        pos_embedding = self.out_embedding(pos_labels)  # [batch, window * 2, hidden]
        neg_embedding = self.out_embedding(neg_labels)  # [batch, window * 2 * k, hidden]

        input_embedding = input_embedding.unsqueeze(2)  # [batch, hidden, 1] must be the same dimension when use bmm

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch, window * 2, 1]
        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch, window * 2 * k, 1]

        pos_dot = pos_dot.squeeze(2)  # [batch, window * 2]
        neg_dot = neg_dot.squeeze(2)  # [batch, window * 2 * k]

        pos_loss = F.logsigmoid(pos_dot).sum(1)
        neg_loss = F.logsigmoid(neg_dot).sum(1)

        loss = neg_loss + pos_loss

        return -loss

    def get_input_embedding(self):
        return self.in_embedding.weight.detach()

if __name__ =="__main__":
    # tags = pd.read_csv('./ml-latest-small/tags.csv')
    df_movies = pd.read_csv('./ml-latest-small/movies.csv')
    ratings = pd.read_csv('./ml-latest-small/ratings.csv')
    ratings = ratings[0:2000]
    tags = pd.read_csv('./ml-latest-small/tags.csv')

    df_movies['movieId'] = df_movies['movieId'].astype(str)

    ratings['userId'] = ratings['userId'].astype(str)
    ratings['movieId'] = ratings['movieId'].astype(str)

    tags['movieId'] = tags['movieId'].astype(str)
    tags['userId'] = tags['userId'].astype(str)

    tags['tag'] = tags['tag'].apply(lambda x: x.lower())
    xs = tags.groupby('movieId')['tag'].apply(list)
    xs = xs.apply(lambda x: list(set(x)))
    df_tags_merge = pd.DataFrame(xs, columns=['tag'])
    dict_tags_merge = df_tags_merge.to_dict(orient='index')

    df_movies['release_year'] = df_movies['title'].apply(lambda x: x.split('(')[-1].replace(')', ''))
    df_movies['release_year'] = df_movies['title'].apply(
        lambda x: x.split('(')[-1].replace(')', '') if len(x.split('(')) > 1 else '')
    df_movies['release_year'] = df_movies['release_year'].apply(lambda x: 'release_' + x)
    df_movie_sort_by_release_year = df_movies.sort_values(by='release_year', ascending=False)

    df = pd.merge(ratings, df_movies, how='left', on="movieId")

    df.sort_values(by=['userId', 'timestamp'], ascending=[True, False], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['time_block'] = df['timestamp'].apply(get_time_block)
    df['year'] = df['timestamp'].dt.year
    df = df[~df['title'].isnull()]

    df['userId'] = df['userId'].apply(lambda x: 'user_' + x)

    print(f"ratings: {ratings.shape}")
    print(f"movies: {df_movies.shape}")
    # df_relation = df[['userId','movieId','rating']]

    for i in df['title'].values[0]:
        i.lower()

    ############
    df_movies = df_movies[df_movies['movieId'].isin(df['movieId'])]
    df_movies.reset_index(inplace=True)

    genres_all = list(set([j for i in df['genres'].apply(lambda x: x.split('|')) for j in i]))
    userlist = df['userId'].unique().tolist()
    movielist = df['movieId'].unique().tolist()
    df = df[df['movieId'].isin(df_movies['movieId'])]
    genreslist = [i.split('|') for i in df['genres'].tolist()]
    df['genres_list'] = genreslist
    genreslist = list(set([j for i in genreslist for j in i]))
    time_block = df['time_block'].unique().tolist()

    genreslist = [i.split('|') for i in df_movies['genres'].tolist()]
    df_movies['genres_list'] = genreslist

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(df_movies['genres_list'])

    # 将 one-hot 编码结果转换为 DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)
    one_hot_df = pd.concat([df_movies['movieId'], one_hot_df], axis=1)
    one_hot_dict = one_hot_df.set_index('movieId').to_dict(orient='index')

    side_info = dict(zip(df_movies['movieId'], df['genres_list']))

    G = nx.Graph()

    for u in userlist:
        G.add_node(u, attr='user')
    for i in movielist:
        G.add_node(i, attr='item')
    # for g in genreslist:
    #     G.add_node(g, attr='genres')
    # for t in time_block:
    #     G.add_node(t, attr='time_block')

    for _, row in df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        score = row['rating']
        genres_list = row['genres_list']
        time_block = row['time_block']
        G.add_edge(user_id, movie_id, score=score)
        # G.add_edge(movie_id, time_block)
        # for genre in genres_list:
        #     G.add_edge(movie_id, genre)

    node2vec = Node2Vec(G, dimensions=32, walk_length=5, num_walks=10, workers=multiprocessing.cpu_count() - 1,p= 2,q =4)

    walks_ = node2vec.walks

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# 創建假的用戶-物品互動數據
users = ['U' + str(i) for i in range(1, 101)]  # 100 個用戶
items = ['I' + str(i) for i in range(1, 51)]   # 50 個物品

np.random.seed(42)
interactions = [(np.random.choice(users), np.random.choice(items)) for _ in range(500)]

df_interactions = pd.DataFrame(interactions, columns=['user_id', 'item_id'])

# 創建假的用戶側面信息
user_info = pd.DataFrame({
    'user_id': users,
    'age': np.random.randint(18, 70, size=100),
    'gender': np.random.choice(['M', 'F'], size=100)
})

# 創建假的物品側面信息
item_info = pd.DataFrame({
    'item_id': items,
    'category': np.random.choice(['A', 'B', 'C', 'D'], size=50),
    'price': np.random.uniform(10, 1000, size=50).round(2)
})

print(df_interactions.head())
print(user_info.head())
print(item_info.head())


df = df_interactions.merge(user_info, on='user_id').merge(item_info, on='item_id')

# 對類別特徵進行編碼
le_gender = LabelEncoder()
le_category = LabelEncoder()
user_info['gender_encoded'] = le_gender.fit_transform(user_info['gender'])
item_info['category_encoded'] = le_category.fit_transform(item_info['category'])


# 創建圖
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['user_id'], row['item_id'])

# 準備側面信息
side_info = []
for node in G.nodes():
    if node.startswith('U'):
        user_data = user_info[user_info['user_id'] == node].iloc[0]
        side_info.append([user_data['age'], user_data['gender_encoded']])
    else:
        item_data = item_info[item_info['item_id'] == node].iloc[0]
        side_info.append([item_data['price'], item_data['category_encoded']])

# 只對連續數值特徵進行歸一化
scaler = MinMaxScaler()
side_info_normalized = np.zeros_like(side_info)
side_info_normalized[:, 0] = scaler.fit_transform(side_info[:, 0].reshape(-1, 1)).ravel()  # 年齡和價格
side_info_normalized[:, 1] = side_info[:, 1]  # 保持性別和類別的編碼不變


def generate_walks(graph, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        for node in graph.nodes():
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if neighbors:
                    walk.append(np.random.choice(neighbors))
                else:
                    break
            walks.append([str(node) for node in walk])
    return walks

def train_base_embeddings(walks, embedding_dim, window_size):
    model = Word2Vec(sentences=walks, vector_size=embedding_dim, window=window_size, min_count=0, sg=1, workers=4)
    return {node: model.wv[str(node)] for node in model.wv.index_to_key}

def enhance_embeddings(base_embeddings, side_info):
    enhanced_embeddings = {}
    for node, base_vector in base_embeddings.items():
        node_index = list(base_embeddings.keys()).index(node)
        side_vector = side_info[node_index]
        enhanced_vector = np.concatenate([base_vector, side_vector])
        enhanced_embeddings[node] = enhanced_vector
    return enhanced_embeddings

def train_eges(graph, side_info, embedding_dim=64, walk_length=10, num_walks=80, window_size=5):
    walks = generate_walks(graph, num_walks, walk_length)
    base_embeddings = train_base_embeddings(walks, embedding_dim, window_size)
    enhanced_embeddings = enhance_embeddings(base_embeddings, side_info)
    return enhanced_embeddings

walks = generate_walks(G, num_walks=80, walk_length=10)
base_emb = train_base_embeddings(walks, embedding_dim=64, window_size=5)



print("基礎嵌入示例:")
print(list(base_emb.items())[:2])