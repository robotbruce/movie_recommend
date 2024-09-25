from torch import nn
import torch
import pandas as pd
import networkx as nx
import math
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import multiprocessing
import torch.optim as optim

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


# class EGESModel(nn.Module):
#     def __init__(self, num_nodes,node_dim, genre_dim, num_genres):
#         super(EGESModel, self).__init__()
#         self.node_embedding = nn.Embedding(num_nodes, node_dim)
#         self.genre_embedding = nn.Embedding(num_genres, genre_dim)
#         self.weight_fc = nn.Linear(genre_dim, 1)  # 用於學習權重
#
#     def forward(self, node_idx, genre_idx):
#         node_emb = self.node_embedding(node_idx)
#         genre_emb = self.genre_embedding(genre_idx)
#
#         # 類型嵌入的加權和
#         weights = torch.softmax(self.weight_fc(genre_emb), dim=0)
#         weighted_genre_emb = torch.sum(weights * genre_emb, dim=0)
#
#         # 結合節點嵌入與side information
#         combined_emb = torch.cat((node_emb, weighted_genre_emb), dim=1)
#         return combined_emb


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

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(df_movies['genres_list'])

    # 将 one-hot 编码结果转换为 DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)
    one_hot_df = pd.concat([df_movies['movieId'], one_hot_df], axis=1)
    one_hot_dict = one_hot_df.set_index('movieId').to_dict(orient='index')


    G = nx.Graph()

    for u in userlist:
        G.add_node(u, attr='user')
    for i in movielist:
        G.add_node(i, attr='item')
    for g in genreslist:
        G.add_node(g, attr='genres')
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
        for genre in genres_list:
            G.add_edge(movie_id, genre)

    node2vec = Node2Vec(G, dimensions=32, walk_length=20, num_walks=50, workers=multiprocessing.cpu_count() - 1,p= 2,q =4)
    num_nodes = len(G.nodes)
    num_genres = len(genreslist)
    num_dim = 64
    genre_dim = 32

    model = EGESModel(num_nodes, num_dim, genre_dim, num_genres)

    df_movies[['movieId']] = df_movies[['movieId']].astype(str)

    genreslist = [i.split('|') for i in df_movies['genres'].tolist()]
    df_movies['genres_list'] = genreslist

    lr = 0.001
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss()

    window_size = 10
    node_pairs = []

    walks = node2vec.walks

    for walk in walks:
        for i, node in enumerate(walk):
            # 確定上下文範圍
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)

            # 為當前節點生成 (node, context) 對
            for j in range(start, end):
                if i != j:  # 避免將節點與自己配對
                    node_pairs.append((node, walk[j]))

    genre_info = dict(zip(df_movies['movieId'],df['genres_list']))


    for epoch in range(epochs):
        total_loss = 0
        for node_pair in node_pairs:
            optimizer.zero_grad()
            node_idx, context_idx = node_pair
            if 'user' in node:
                node_idx = torch.tensor([int(node.split('_')[1]) + num_nodes])
            else:
                node_idx = torch.tensor([int(node)])

            if 'user' in context_idx:
                context_idx = torch.tensor([int(context_idx.split('_')[1]) + num_nodes])
            else:
                context_idx = torch.tensor([int(num_nodes)])

            # 計算節點與上下文的嵌入
            node_emb = model(node_idx, genre_idx)
            context_emb = model(context_idx, genre_idx)

####---------------####

node_model = node2vec.fit(window=10, min_count=1, batch_words=4)
graph_embeddings = {node: node_model.wv[node] for node in G.nodes()}



# 6. 準備訓練數據
def prepare_data(entities, side_info = None):
    graph_emb = torch.tensor([graph_embeddings[e] for e in entities], dtype=torch.float32)
    if side_info:
        side_emb = torch.tensor([[info[k] for k in genres_all if k in info] for info in side_info.values()], dtype=torch.float32)
        return graph_emb, side_emb
    return graph_emb

item_graph_emb, item_side_emb = prepare_data(movielist, one_hot_dict)
user_graph_emb = prepare_data(userlist)

# 6. 訓練EGES模型
model = EGES(32,20)

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(item_graph_emb, item_side_emb)
    loss = criterion(output, item_graph_emb)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')


# 7. 生成最終的用戶嵌入
with torch.no_grad():
    item_embeddings = model(item_graph_emb, item_side_emb).numpy()


side_info_list = [['Action']]
side_info_key = 'age'
value = 1
top_n = 10
# 9. 基於側面信息的推薦函數
def recommend_items_by_side_info(side_info_list, top_n=10):
    mock_side_info = torch.tensor(mlb.transform(side_info_list)[0])
    mock_item_graph_emb = torch.mean(item_graph_emb, dim=0).unsqueeze(0)

    # 生成模擬用戶的嵌入
    with torch.no_grad():
        mock_user_embedding = model(mock_item_graph_emb, mock_side_info.unsqueeze(0)).numpy()

    similarities = cosine_similarity(mock_user_embedding, item_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:]

    return [movielist[i] for i in top_indices]

recommends = recommend_items_by_side_info(side_info_list = [['Children']],top_n=10)

def search_recommend_title(recommends):
    recommend_title = []
    for i in recommends:
        title = df_movies[df_movies['movieId']==i]['title'].values[0]
        recommend_title.append(title)
    return(recommend_title)

recommends_title = search_recommend_title(recommends)

pprint(recommends_title)
