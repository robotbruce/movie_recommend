import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler

# 假設的電影觀看數據
data = {
    'user_id': ['1', '1', '2', '2', '3', '3'],
    'movie_id': ['101', '102', '101', '103', '102', '103'],
    'timestamp': [1609459200, 1609545600, 1609645600, 1609745600, 1609845600, 1609945600],
    'score': [4.5, 3.0, 4.0, 5.0, 3.5, 4.0]
}

# 轉換成 DataFrame
df = pd.DataFrame(data)

# 創建一個無向圖形
G = nx.Graph()

# 添加用戶節點和電影節點以及邊
for _, row in df.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']
    timestamp = row['timestamp']
    score = row['score']

    # 添加用戶節點
    if not G.has_node(user_id):
        G.add_node(user_id, type='user')

    # 添加電影節點
    if not G.has_node(movie_id):
        G.add_node(movie_id, type='movie')

    # 添加邊
    G.add_edge(user_id, movie_id, timestamp=timestamp, score=score)


# 定義節點特徵
# 將 timestamp 和 score 轉換成特徵向量
def create_feature_vector(timestamp, score):
    # 將 timestamp 轉換成年份
    year = pd.to_datetime(timestamp, unit='s').year
    # 特徵向量
    return np.array([year, score])


# 為每個節點添加特徵
for node in G.nodes():
    print(node)
    if G.nodes[node]['type'] == 'user':
        # 用戶節點
        G.nodes[node]['features'] = create_feature_vector(df[df['user_id'] == node]['timestamp'].values[0],
                                                          df[df['user_id'] == node]['score'].values[0])
    elif G.nodes[node]['type'] == 'movie':
        # 電影節點
        G.nodes[node]['features'] = create_feature_vector(df[df['movie_id'] == node]['timestamp'].values[0],
                                                          df[df['movie_id'] == node]['score'].values[0])

# 使用 node2vec 算法進行隨機遊走和節點表示學習
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
node2vec_walks = node2vec.walks
model = node2vec.fit(window=10, min_count=1)

# 獲取節點表示
node_embeddings = {str(node): model.wv[node] for node in G.nodes()}

# 打印節點表示
for node, embedding in node_embeddings.items():
    print(f"Node {node} -> Embedding: {embedding}")
