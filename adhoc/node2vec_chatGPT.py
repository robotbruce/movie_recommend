import networkx as nx
from node2vec import Node2Vec
import random

# 建立一個空的有向圖形
G = nx.DiGraph()

# 建立使用者和商品的列表
users = ['User{}'.format(i) for i in range(1, 101)]
products = ['Product{}'.format(i) for i in range(1, 101)]

# 增加使用者節點，並隨機指定角色
for user in users:
    role = random.choice(['Manager', 'Developer', 'Analyst'])
    G.add_node(user, role=role)

# 增加商品節點，並隨機指定類型
for product in products:
    product_type = random.choice(['Electronics', 'Clothing', 'Books'])
    G.add_node(product, type=product_type)

# 增加隨機的交互作用
for user in users:
    for _ in range(random.randint(1, 10)):
        product = random.choice(products)
        G.add_edge(user, product)

# 定義node2vec演算法參數
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# 訓練node2vec模型
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 取得用戶向量
user_vectors = {user: model.wv[user] for user in users}

# 取得商品向量
product_vectors = {product: model.wv[product] for product in products}

# 在此可以進行推薦系統的其他操作，例如計算用戶與商品之間的相似度，然後推薦相似的商品給用戶等等

# 定義根據職務名稱推薦商品的函數
def recommend_products_by_role(role, num_recommendations=5):
    # 找到具有指定職務名稱的使用者
    relevant_users = [user for user in users if G.nodes[user]['role'] == role]

    # 找到這些使用者喜歡的商品
    liked_products = []
    for user in relevant_users:
        for product in G.neighbors(user):
            liked_products.append(product)

    # 移除使用者已經喜歡的商品
    liked_products = set(liked_products)
    for neighbor_product in G.neighbors('User1'):
        if neighbor_product in liked_products:
            liked_products.remove(neighbor_product)

    # 隨機推薦商品
    recommended_products = random.sample(liked_products, min(num_recommendations, len(liked_products)))
    return recommended_products


# 試著根據使用者的職務名稱來推薦商品
role = 'Manager'  # 指定職務名稱
recommended_products = recommend_products_by_role(role)
print("Recommended products for users with role '{}' :".format(role))
for product in recommended_products:
    print(product)


similar_products = []
product_vector = model.wv['Product1']
model.wv.most_similar(['User1'])