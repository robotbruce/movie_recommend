from torch import nn
import torch
import pandas as pd
import networkx as nx
import math
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import multiprocessing
import torch.optim as optim
from pprint import pprint
import torch.nn.functional as F
import math
from node2vec import Node2Vec

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
    df_movies['genres_list'] = df_movies['genres'].apply(lambda x: x.split('|'))

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



    ####train test split

    df['rank_latest'] = df.groupby(['userId'])['timestamp'] \
        .rank(method='first', ascending=False)

    train_df = df[df['rank_latest'] != 1]
    test_df = df[df['rank_latest'] == 1]


    print(f"ratings: {train_df.shape}")
    print(f"movies: {df_movies.shape}")
    # df_relation = df[['userId','movieId','rating']]

    for i in df['title'].values[0]:
        i.lower()
    from collections import Counter

    train_user_id = train_df['userId'].unique().tolist()
    user_prefer_dict = {}
    for uid in train_user_id:
        user_prefer = []
        filter_user = df[df['userId']==uid]
        for user_genres in filter_user['genres_list']:
            user_prefer.extend(user_genres)
        user_prefer_counter = Counter(user_prefer)
        user_most_common = user_prefer_counter.most_common()
        new_user_prefer = [genres for genres,_ in user_most_common[0:5]]
        user_prefer_dict.update({uid:new_user_prefer})


    ############
    df_movies = df_movies[df_movies['movieId'].isin(df['movieId'])]
    df_movies.reset_index(drop = True,inplace=True)

    genres_all = list(set([j for i in train_df['genres'].apply(lambda x: x.split('|')) for j in i]))
    userlist = train_df['userId'].unique().tolist()
    movielist = train_df['movieId'].unique().tolist()
    train_df = train_df[train_df['movieId'].isin(df_movies['movieId'])]
    genreslist = [i.split('|') for i in df['genres'].tolist()]
    train_df['genres_list'] = genreslist
    genreslist = list(set([j for i in genreslist for j in i]))
    time_block = df['time_block'].unique().tolist()

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(df_movies['genres_list'])

    # 将 one-hot 编码结果转换为 DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)
    one_hot_df = pd.concat([df_movies['movieId'], one_hot_df], axis=1)
    one_hot_dict = one_hot_df.set_index('movieId').to_dict(orient='index')


    df_user_prefer = pd.DataFrame(0, index=train_user_id, columns=genreslist)
    for uid in train_user_id:
        user_prefer = []
        filter_user = df[df['userId']==uid]
        for user_genres in filter_user['genres_list']:
            user_prefer.extend(user_genres)
        user_prefer_counter = Counter(user_prefer)
        user_most_common = user_prefer_counter.most_common()
        for genres_cat, counter in user_most_common:
            df_user_prefer.at[uid, genres_cat] = counter

    df_user_prefer['total_count'] = df_user_prefer[genreslist].sum(axis=1)
    preference_df = df_user_prefer[genreslist].div(df_user_prefer['total_count'], axis=0)

    # Create a dictionary to store the top 5 users for each genre
    top_users_by_genre = {}

    # Sort and get the top 5 users for each genre
    for genre in genreslist:
        # Sort the users by the count of the genre, in descending order
        top_users = preference_df[[genre]].sort_values(by=genre, ascending=False).head(5)

        # Convert to list of tuples (user, count)
        top_users_by_genre[genre] = list(top_users.itertuples(index=True, name=None))

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


    for user in user_prefer_dict:
        user_prefer_genres = user_prefer_dict[user]
        for category in user_prefer_genres:
            G.add_edge(user,category)

    for genre in top_users_by_genre:
        users = top_users_by_genre[genre]
        for user_id,score in users:
            G.add_edge(user_id, genre,score=math.exp(score))

    node2vec = Node2Vec(
        G,
        dimensions=64,
        walk_length=20,
        num_walks=50,
        workers=multiprocessing.cpu_count() - 1,p= 2,q =4)


    num_nodes = len(G.nodes)
    num_genres = len(genreslist)

    walks = node2vec.walks

    # for walk in walks:
    #     for i, node in enumerate(walk):
    #         # 確定上下文範圍
    #         start = max(0, i - window_size)
    #         end = min(len(walk), i + window_size + 1)
    #
    #         # 為當前節點生成 (node, context) 對
    #         for j in range(start, end):
    #             if i != j:  # 避免將節點與自己配對
    #                 node_pairs.append((node, walk[j]))

    # genre_info = dict(zip(df_movies['movieId'],df['genres_list']))

model = node2vec.fit(window=20, min_count=1, batch_words=4,sg=1,epochs = 5)
graph_embeddings = {node: torch.tensor(model.wv[node]) for node in G.nodes()}

####movie genres 實驗
movie_id = df_movies['movieId'].tolist()
for id_ in movie_id:
    avg_embed_list = []
    if id_ in graph_embeddings:
        avg_embed_list.append(graph_embeddings[id_])
        movie_genres = df_movies[df_movies['movieId']==id_]['genres_list'].values
        movie_genres = movie_genres[0]
        for genre in movie_genres:
            avg_embed_list.append(graph_embeddings[genre])
        enhance_embedding = torch.mean(torch.stack(avg_embed_list), dim=0)
        graph_embeddings[id_] = enhance_embedding
####user genres 實驗

for uer_id in user_prefer_dict:
    avg_embed_list = []
    if uer_id in graph_embeddings:
        avg_embed_list.append(graph_embeddings[uer_id])
        like_genres = user_prefer_dict[uer_id]
        for genre in like_genres:
            avg_embed_list.append(graph_embeddings[genre])
        enhance_embedding = torch.mean(torch.stack(avg_embed_list), dim=0)
        graph_embeddings[uer_id] = enhance_embedding

for genre in top_users_by_genre:
    users = top_users_by_genre[genre]
    avg_embed_list = []
    avg_embed_list.append(graph_embeddings[genre])
    for user_id,_ in users:
        avg_embed_list.append(graph_embeddings[user_id])
    enhance_embedding = torch.mean(torch.stack(avg_embed_list), dim=0)
    graph_embeddings[genre] = enhance_embedding


# 假設 embeddings_dict 是你 skip-gram 訓練後的所有 embeddings 字典
class model_embed:
    def __init__(self,embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.embeddings_tensor = None
        self.embeddings_tensor_keys = []
        self.embeddings_movie_keys = []
        self.embeddings_movie_tensor = None

    def initialize_embeddings(self):
        self.embeddings_tensor_keys = list(self.embeddings_dict.keys())
        self.embeddings_tensor = torch.stack([self.embeddings_dict[key] for key in self.embeddings_tensor_keys])

    def create_movies_embedding_space(self,movie_list):
        for movie_id in movie_list:
            self.embeddings_movie_keys.append(movie_id)
        self.embeddings_movie_tensor = torch.stack([self.embeddings_dict[key] for key in self.embeddings_movie_keys])

    # 查詢相似的 top k embeddings
    def find_top_k_similar_embeddings(self,search_list,top_k=10):
        # 提取目標 embedding

        target_list = list(set(search_list) & set(self.embeddings_tensor_keys))

        item_key_indices = []
        if target_list:
            for item in target_list:
                item_key_indice = self.embeddings_tensor_keys.index(item)
                item_key_indices.append(item_key_indice)
            targe_embeddings = self.embeddings_tensor[item_key_indices]
            targe_mean_embeddings = torch.mean(targe_embeddings,dim = 0)
            cos_sim = F.cosine_similarity(self.embeddings_tensor, targe_mean_embeddings.unsqueeze(0), dim=1)
            # 過濾掉自身
            for idx in item_key_indices:
                cos_sim[idx] = -float('inf')  # 將自身相似度設置為負無限大，以便在 top k 中排除

            # 找出相似度最高的前 top k 個 index
            top_k_indices = torch.topk(cos_sim, top_k).indices

            # 提取這些 index 對應的 keys 和相似度，並組成 (key, similarity) 元組
            top_k_results = [(self.embeddings_tensor_keys[idx], cos_sim[idx].item()) for idx in top_k_indices]
            return top_k_results

    def find_top_k_movie_similar_embeddings(self,search_list,top_k=10):
        # 提取目標 embedding
        target_list = list(set(search_list) & set(self.embeddings_tensor_keys))
        target_list_in_movie = list(set(search_list) & set(self.embeddings_movie_keys))

        item_key_indices = []
        if target_list:
            for item in target_list:
                item_key_indice = self.embeddings_tensor_keys.index(item)
                item_key_indices.append(item_key_indice)
            targe_embeddings = self.embeddings_tensor[item_key_indices]
            targe_mean_embeddings = torch.mean(targe_embeddings,dim = 0)
            cos_sim = F.cosine_similarity(self.embeddings_movie_tensor, targe_mean_embeddings.unsqueeze(0), dim=1)
            # 過濾掉自身
            if target_list_in_movie:
                item_key_indices_in_movie = []
                for movie_item in target_list_in_movie:
                    movie_item_key_indices = self.embeddings_movie_keys.index(movie_item)
                    item_key_indices_in_movie.append(movie_item_key_indices)
                for idx in item_key_indices_in_movie:
                    cos_sim[idx] = -float('inf')  # 將自身相似度設置為負無限大，以便在 top k 中排除

            # 找出相似度最高的前 top k 個 index
            top_k_indices = torch.topk(cos_sim, top_k).indices

            # 提取這些 index 對應的 keys 和相似度，並組成 (key, similarity) 元組
            top_k_results = [(self.embeddings_movie_keys[idx], cos_sim[idx].item()) for idx in top_k_indices]
            return top_k_results

def search_recommend_title(recommends,search_method = ""):
    """
    :param recommends: search_list
    :param search_method: id,title
    :return: movie_info dict
    """
    recommend_title = []
    for i in recommends:
        if search_method == "id":
            if i in set(df_movies['movieId']):
                title = df_movies[df_movies['movieId']==i]['title'].values[0]
                movieId = df_movies[df_movies['movieId'] == i]['movieId'].values[0]
                genres = df_movies[df_movies['movieId'] == i]['genres'].values[0]
                recommend_title.append((title,movieId,genres))
        elif search_method == "title":
            for t in set(df_movies['title']):
                if i.lower() in t.lower():
                    title = df_movies[df_movies['title'].isin([t])]['title'].values[0]
                    movieId = df_movies[df_movies['title'].isin([t])]['movieId'].values[0]
                    genres = df_movies[df_movies['title'].isin([t])]['genres'].values[0]
                    recommend_title.append((title, movieId, genres))
    return(recommend_title)

gemb = model_embed(graph_embeddings)
gemb.initialize_embeddings()
gemb.create_movies_embedding_space(movielist)

search_list = ['Children']
inputs = search_recommend_title(search_list)
pprint(inputs)

recommend_results = gemb.find_top_k_movie_similar_embeddings(search_list=search_list,top_k=10)
recommend_ids = [i for i,_ in recommend_results]
titles = search_recommend_title(recommend_ids)
pprint(titles)

recommend_results_origin = model.wv.most_similar(positive=search_list,topn=10)
recommend_origin_ids = [i for i,_ in recommend_results_origin]
titles_origin = search_recommend_title(recommend_origin_ids)
pprint(titles_origin)

# User-item pairs for testing
test_user_item_set = set(zip(test_df['userId'], test_df['movieId']))


from tqdm.auto import tqdm
import numpy as np
# Dict of all items that are interacted with by each user
user_interacted_items = df.groupby('userId')['movieId'].apply(list).to_dict()



from pprint import pprint
hits = []
recalls = []
for (u, i) in tqdm(test_user_item_set):
    user_prefer_list = user_prefer_dict[u]
    movie_genres = df_movies[df_movies['movieId']==i]['genres_list'].values[0]

    interacted_items = user_interacted_items[u]

    input_items = user_prefer_list+[i]

    # input = []
    # input = input + [u]
    # input = input + [i]
    title_input = search_recommend_title([i],search_method='id')
    pprint("#"*10+f"input_title     user_id: {u}; title_input: {title_input}")
    pprint(f"user_prefer: {user_prefer_list}")

    # predicted_labels = gemb.find_top_k_movie_similar_embeddings(search_list=input_items,top_k=10)
    predicted_labels = model.wv.most_similar(positive=input_items,topn=10)


    top10_items = [item for item,_ in predicted_labels][0:10]

    # top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
    hit_ = set(interacted_items) & set(top10_items)
    print(f'hit : {hit_}')
    title_predict = search_recommend_title(hit_,search_method='id')
    recall_count = len(hit_)
    pprint(title_predict)
    if hit_:
        hits.append(1)
        recalls.append(recall_count)
    else:
        hits.append(0)
        recalls.append(0)

    # if i in top10_items:
    #     hits.append(1)
    # else:
    #     hits.append(0)

print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))
print("The Hit Ratio @ 10 is {:.2f}".format(np.average(recalls)))

input_items = ['Drama', 'Comedy', 'Thriller', 'Mystery', 'Romance']
input_items = input_items + ['296']
title_input = search_recommend_title(['296'],'id')

input_items = ['IMAX','185585']

# predicted_labels = gemb.find_top_k_movie_similar_embeddings(search_list=input_items, top_k=10)
predicted_labels = model.wv.most_similar(positive=input_items,topn=10)
top10_items = [item for item, _ in predicted_labels][0:10]
title_predict = search_recommend_title(top10_items,'id')
pprint(title_predict)

title_predict = search_recommend_title(['Pacific'],'title')
pprint('185585')


