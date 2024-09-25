import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import multiprocessing
import logging
from node2vec import Node2Vec
from gensim.models.callbacks import CallbackAny2Vec

logger = logging.getLogger(__name__)

def add_nodes(G, df, col, type_name):
    """Add entities to G from the 'col' column of the 'df' DataFrame. The new nodes are annotated with 'type_name' label."""
    nodes = list(df[~df[col].isnull()][col].unique())
    G.add_nodes_from([(n, dict(type=type_name)) for n in nodes])
    print("Nodes (%s,%s) were added" % (col, type_name))


def add_links(G, df, col1, col2, type_name):
    """Add links to G from the 'df' DataFrame. The new edges are annotated with 'type_name' label."""
    df_tmp = df[(~df[col1].isnull()) & (~df[col2].isnull())]
    links = list(zip(df_tmp[col1], df_tmp[col2]))
    G.add_edges_from([(src, trg, dict(type=type_name)) for src, trg in links])
    print("Edges (%s->%s,%s) were added" % (col1, col2, type_name))


def encode_graph(G):
    """Encode the nodes of the network into integers"""
    nodes = [(n, d.get("type", None)) for n, d in G.nodes(data=True)]
    nodes_df = pd.DataFrame(nodes, columns=["id", "type"]).reset_index()
    node2idx = dict(zip(nodes_df["id"], nodes_df["index"]))
    edges = [(node2idx[src], node2idx[trg], d.get("type", None)) for src, trg, d in G.edges(data=True)]
    edges_df = pd.DataFrame(edges, columns=["src", "trg", "type"])
    return nodes_df, edges_df

def add_movie_nodes(G,movieFitGenres_list):
    """Add entities to G from the 'col' column of the 'df' DataFrame. The new nodes are annotated with 'type_name' label."""
    for movieid,genres in movieFitGenres_list:
        G.add_node(movieid, genres=genres)
    print("Nodes movie,genres were added")


def add_links(G, df, col1, col2, type_name):
    """Add links to G from the 'df' DataFrame. The new edges are annotated with 'type_name' label."""
    df_tmp = df[(~df[col1].isnull()) & (~df[col2].isnull())]
    links = list(zip(df_tmp[col1], df_tmp[col2]))
    G.add_edges_from([(src, trg, dict(type=type_name)) for src, trg in links])
    print("Edges (%s->%s,%s) were added" % (col1, col2, type_name))

def add_user_movie_edges(G,df,nodeName1,nodeName2,weightName):
    links = list(zip(df[nodeName1],df[nodeName2],df[weightName]))
    for link in links:
        G.add_edge(link[0],link[1],weight = link[2])

def searh_similar(model_,search_list,search_target_df,topn = 10):
    search_target_df['movieId'] = search_target_df['movieId'].astype(str)
    df_search_info = search_target_df[search_target_df['movieId'].isin(search_list)]
    score_list = []
    item_list = []
    for item, score in model_.wv.most_similar(search_list, topn=topn):
        item_list.append(item)
        print(f'movieId:{item}')
        score_list.append(score)

    # similar_items = [item for item,_ in model.wv.most_similar(search_list,topn = topn)]
    similar_items_filter = [similar_item for similar_item in item_list if similar_item in search_target_df['movieId'].values.tolist()]
    sorterIndex = dict(zip(similar_items_filter, range(len(similar_items_filter))))
    df_similar = search_target_df[search_target_df['movieId'].isin(sorterIndex)]
    df_similar['movie_Rank'] = df_similar['movieId'].map(sorterIndex)
    df_similar.sort_values(by = ['movie_Rank'],ascending=True,inplace = True)
    df_similar.reset_index(drop=True,inplace=True)
    df_similar['recom_score'] = pd.DataFrame(score_list)

    search_info_and_results = dict(
        search_df = df_search_info,
        similar_results = df_similar
    )
    return search_info_and_results


def searh_similar_by_genre(model, search_list, search_target_df, topn=10):
    # df_search_info = search_target_df[search_target_df['movieId'].astype(str).isin(search_list)]
    score_list = []
    item_list = []
    for item, score in model.wv.most_similar(search_list, topn=topn):
        item_list.append(item)
        print(f'movieId:{item}')
        score_list.append(score)

    # similar_items = [item for item,_ in model.wv.most_similar(search_list,topn = topn)]
    similar_items_filter = [similar_item for similar_item in item_list if
                            similar_item in search_target_df['movieId'].values.tolist()]
    sorterIndex = dict(zip(similar_items_filter, range(len(similar_items_filter))))
    df_similar = df_movies[df_movies['movieId'].isin(sorterIndex)]
    df_similar['movie_Rank'] = df_similar['movieId'].map(sorterIndex)
    df_similar.sort_values(by=['movie_Rank'], ascending=True, inplace=True)
    df_similar.reset_index(drop=True, inplace=True)
    df_similar['recom_score'] = pd.DataFrame(score_list)

    search_info_and_results = dict(
        similar_results=df_similar
    )
    return search_info_and_results

class Callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logger.info("Loss after epoch {}: {}".format(self.epoch, loss))
        self.epoch += 1

def replace_str_in_tags_lambda(list_in_lambda):
    for index,movie in enumerate(list_in_lambda):
        list_in_lambda[index] = movie.replace('^^',' ').lower()
    return list_in_lambda


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


class vector4recommend:
    def __init__(self, _model_=None, item_category_dict=None):

        self.node2vec_model = _model_
        self.item_dict = item_category_dict

        self.item_embeddings = {}
        self.user_embeddings = {}
        self.embeddings_all = {}
        self.side_info_embeddings = {}

        self.all_index = set(self.node2vec_model.wv.index_to_key)
        self.item_index = set()
        self.user_index = set()
        self.sideinfo_index = set()

        for w_ in self.all_index:
            self.embeddings_all[w_] = self.node2vec_model.wv[w_]

    def list_of_items(self):
        user_idlist = set(self.item_dict['user_idlist'])
        sideinfo_list = set(self.item_dict['sideinfo_list'])
        items = set(self.item_dict['items'])
        return items, user_idlist, sideinfo_list

    def create_vector_space(self):
        """
        items, resources , posts, users, sideinfo
        """
        items, users, sideinfo = self.list_of_items()

        for word in self.all_index:
            if word in items:
                self.item_embeddings[word] = self.node2vec_model.wv[word]
            if word in users:
                self.user_embeddings[word] = self.node2vec_model.wv[word]
            if word in sideinfo:
                self.side_info_embeddings[word] = self.node2vec_model.wv[word]

        self.sideinfo_index = set(self.side_info_embeddings.keys())
        self.user_index = set(self.user_embeddings.keys())
        self.item_index = set(self.item_embeddings.keys())

    def calculate_similar(self, item1, item2):
        return np.dot(item1, item2) / (np.linalg.norm(item1) * np.linalg.norm(item2))

    def aggregated_items_embedding(self, items_list):
        embeddings = []
        if items_list:
            for item in items_list:
                embeddings.append(self.embeddings_all[str(item)])
            aggregated_embedding = np.mean(embeddings, axis=0)
            return aggregated_embedding

    def search_similar(self, search_list, data_type='item', topn=20):
        if data_type == 'item':
            search_space = self.item_embeddings
        if data_type == 'user':
            search_space = self.user_embeddings
        if data_type == 'sideinfo':
            search_space = self.side_info_embeddings

        search_items_in_model = list(set(search_list) & set(self.all_index))
        if search_items_in_model:
            aggregated_embedding = self.aggregated_items_embedding(search_items_in_model)
            similarities = {}
            for other_item, item_embed in search_space.items():
                if other_item not in search_items_in_model:
                    similarity = self.calculate_similar(item1=aggregated_embedding, item2=item_embed)
                    similarities[other_item] = similarity
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            return sorted_similarities[:topn]
        else:
            return []

def search_movie_name(input,df_dictionary):
    df_dictionary['movieId'] = df_dictionary['movieId'].astype('str')
    recom_movie_ids = [recom[0] for recom in input]
    df_search = df_dictionary[df_dictionary['movieId'].isin(recom_movie_ids)]
    df_search['movieId'] = pd.Categorical(df_search['movieId'], categories=recom_movie_ids, ordered=True)
    df_search.sort_values('movieId',inplace= True)
    search_response = df_search[['movieId','title','genres']]
    titles = search_response['title'].values
    for title in titles:
        print(title)
    return search_response

if __name__ =="__main__":
    # tags = pd.read_csv('./ml-latest-small/tags.csv')
    df_movies = pd.read_csv('./ml-latest-small/movies.csv')
    ratings = pd.read_csv('./ml-latest-small/ratings.csv')
    tags = pd.read_csv('./ml-latest-small/tags.csv')

    df_movies['movieId'] = df_movies['movieId'].astype(str)

    ratings['userId'] = ratings['userId'].astype(str)
    ratings['movieId'] = ratings['movieId'].astype(str)

    ratings['datetime'] = pd.to_datetime(ratings['timestamp'])

    tags['movieId'] = tags['movieId'].astype(str)
    tags['userId'] = tags['userId'].astype(str)


    tags['tag'] = tags['tag'].apply(lambda x: x.lower())
    xs = tags.groupby('movieId')['tag'].apply(list)
    xs = xs.apply(lambda x: list(set(x)))
    df_tags_merge = pd.DataFrame(xs,columns=['tag'])
    dict_tags_merge = df_tags_merge.to_dict(orient = 'index')

    df_movies['release_year'] = df_movies['title'].apply(lambda x: x.split('(')[-1].replace(')',''))
    df_movies['release_year'] = df_movies['title'].apply(lambda x: x.split('(')[-1].replace(')','') if len(x.split('('))>1 else '')
    df_movies['release_year'] = df_movies['release_year'].apply(lambda x: 'release_'+x)
    df_movie_sort_by_release_year = df_movies.sort_values(by = 'release_year',ascending=False)

    df = pd.merge(ratings,df_movies,how = 'left',on="movieId")

    df.sort_values(by = ['userId','timestamp'],ascending=[True,False],inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['time_block'] = df['timestamp'].apply(get_time_block)
    df['year'] = df['timestamp'].dt.year
    df = df[~df['title'].isnull()]

    user_demo_df = df[df['timestamp'].dt.year>=2016]
    user_demo_df = df[df['userId'] == 'user_77']

    # df['year'].unique()
    # df_movies['movieId'] = df_movies['movieId'].astype(str)

    # df['title'] = df['title'].astype(str)
    # df['movieId'] = df['movieId'].astype(str)
    # df['userId'] = df['userId'].astype(str)

    # df_movies['movieId'] = df_movies['movieId'].apply(lambda x: 'movie_' + x)
    # df['genres_new'] = df['genres'].apply(lambda x : x.split('|')[0])
    # df['userId'] = df['userId'].apply(lambda x: 'usr_'+ x)
    # df['movieId'] = df['movieId'].apply(lambda x: 'movie_'+ x)

    df['userId'] = df['userId'].apply(lambda x :'user_'+x)

    print(f"ratings: {ratings.shape}")
    print(f"movies: {df_movies.shape}")
    # df_relation = df[['userId','movieId','rating']]

    for i in df['title'].values[0]:
        i.lower()



    ############
    userlist = df['userId'].unique().tolist()
    movielist = df['movieId'].unique().tolist()
    genreslist = [i.split('|') for i in df['genres'].tolist()]
    df['genres_list'] = genreslist
    genreslist = list(set([j for i in genreslist for j in i]))
    time_block = df['time_block'].unique().tolist()

    df['movie_release_year'] = df['release_year'].apply(lambda x : x.split('_')[1])
    release_yearlist = list(set(df['movie_release_year']))
    G = nx.Graph()

    for u in userlist:
        G.add_node(u,attr = 'user')
    for i in movielist:
        G.add_node(i,attr = 'item')
    for g in genreslist:
        G.add_node(g,attr = 'genres')
    for t in time_block:
        G.add_node(t,attr = 'time_block')
    for y in release_yearlist:
        G.add_node(y,attr = 'release_year')



    for _,row in df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        score = row['rating']
        genres_list = row['genres_list']
        time_block = row['time_block']
        release_year = row['movie_release_year']
        G.add_edge(user_id, movie_id, score=score)
        G.add_edge(movie_id, time_block)
        G.add_edge(movie_id,release_year)
        for genre in genres_list:
            G.add_edge(movie_id, genre)

    node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=50, workers=multiprocessing.cpu_count()-1)
    sequences = node2vec.walks
    model = node2vec.fit(window=15, min_count=1, batch_words=5,sg=1,epochs = 2)



    from pprint import pprint
    def search_recommend_title(recommends):
        recommend_title = []
        for i in recommends:
            title = df_movies[df_movies['movieId'] == i]['title'].values[0]
            recommend_title.append(title)
        result = list(zip(recommends,recommend_title))
        return result

    recom = model.wv.most_similar(['Children'])
    recom = [i for i,_ in recom]
    recom_title = search_recommend_title(recom)
    pprint(recom_title)

    recom_d2 = model.wv.most_similar(['Children','Action'])
    recom_d2 = [i for i, _ in recom_d2]
    recom_title2 = search_recommend_title(recom_d2)
    pprint(recom_title2)


    recom_d2 = model.wv.most_similar(['Horror'])
    recom_d2 = [i for i, _ in recom_d2]
    recom_title2 = search_recommend_title(recom_d2)
    pprint(recom_title2)
'a'.lower()
catch_id = []
catch_title = []
for index,row in df_movies[['movieId','title']].iterrows():
    title = row['title']
    idd = row['movieId']
    if 'alien' in title.lower():
        print(title)
        catch_title.append(title)
        catch_id.append(idd)
    alien = list(zip(catch_id,catch_title))
    pprint(alien)

    recom_d2 = model.wv.most_similar(['169984','1214','1200'])
    recom_d2 = [i for i, _ in recom_d2]
    recom_title2 = search_recommend_title(recom_d2)
    pprint(recom_title2)




    items_dict = {
        'user_idlist': userlist,
        'sideinfo_list':genreslist,
        'items': movielist
    }

    vectors = vector4recommend(_model_ = model,item_category_dict=items_dict)
    vectors.create_vector_space()


    recom1 = vectors.search_similar(search_list=['1'],data_type='item')
    res1 = search_movie_name(recom1,df_dictionary=df_movies)

    recom2 = vectors.search_similar(search_list=['364','588'],data_type='item')
    res2 = search_movie_name(recom2,df_dictionary=df_movies)

    recom3 = vectors.search_similar(search_list=['Sci-Fi','Childen'],data_type='item')
    res3 = search_movie_name(recom3,df_dictionary=df_movies)


    ############


    for _, row in df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        score = row['rating']
        release_year = row['release_year']
        time_session = row['time_block']

        # 添加用戶節點
        if not G.has_node(user_id):
            G.add_node(user_id, type='user_node')

        # 添加電影節點
        if not G.has_node(movie_id):
            G.add_node(movie_id, type='movie_node')

        # 添加邊
        G.add_edge(user_id, movie_id, time_session=time_session, score=score)


    # 定義節點特徵
    # 將 timestamp 和 score 轉換成特徵向量
    def create_feature_vector(time_session, score):
        # # 將 timestamp 轉換成年份
        # year = pd.to_datetime(timestamp, unit='s').year
        # # 特徵向量
        return np.array([time_session, score])

    # 為每個節點添加特徵
    for node in G.nodes():
        if G.nodes[node]['type'] == 'user_node':
            # 用戶節點
            G.nodes[node]['features'] = create_feature_vector(df[df['userId'] == node]['time_block'].values[0],
                                                              df[df['userId'] == node]['rating'].values[0])
        elif G.nodes[node]['type'] == 'movie_node':
            # 電影節點
            G.nodes[node]['features'] = create_feature_vector(df[df['movieId'] == node]['time_block'].values[0],
                                                              df[df['movieId'] == node]['rating'].values[0])

    G = nx.from_pandas_edgelist(df, source  = "userId", target = "movieId", edge_attr=['rating'], create_using=nx.Graph())


    # function that will create edges for given movie title and its genres
    def addToGraph(movieid, graph,df):
        if movieid in df['movieId'].values:
            users = df[df['movieId'] == movieid]['userId'].values[0].rstrip().lower().split(', ')
            for user in users:
                graph.add_edge(movieid.strip(), user)
        return graph


    # function that will create graph for all the movies name
    def createGraph(df,df_2):
        graph = nx.Graph()
        for movieid in df['movieId']:
            graph = addToGraph(movieid, graph,df_2)
        return graph

    graph=createGraph(df_movies,df)

    # node2vec = Node2Vec(G, dimensions=64, walk_length=70, num_walks=200, workers=multiprocessing.cpu_count()-1)
    node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=10, workers=multiprocessing.cpu_count()-1)
    sequences = node2vec.walks
    model = node2vec.fit(window=10, min_count=2, batch_words=4)

    # model.wv.vector_size

    search_list_0 = ['260','1196','1210']
    search_list_1 = ['34','1','364']
    search_list_2 = ['648','2745','3354','8387']

    df_similars_movies = searh_similar(model= model,search_list=search_list_2,search_target_df=df_movies,topn=20)
    for i in (df_similars_movies.get('similar_results')['title'].values.tolist()):
        print(i)

    ddd = df_similars_movies.get('search_df')

    df_movies['one_genre'] = df_movies['genres'].apply(lambda x : x.split('|')[0])

    df_new_movies = df_movies[['movieId','one_genre','release_year']]
    df_new_movies.set_index(keys =['movieId'],inplace = True)
    genres_dict = df_new_movies.to_dict(orient='index')

    sequences_for_word2vec = []
    for sequence in sequences:
        temp_sequence = []
        for item in sequence:
            if item in genres_dict:
                genre_info = genres_dict[str(item)]
                genre = genre_info.get('one_genre')
                release_year = genre_info.get('release_year')
                temp_sequence.extend([str(item),genre,release_year])
            else:
                temp_sequence.append(item)
            if item in dict_tags_merge:
                tags_info = dict_tags_merge[item]
                tags = tags_info.get('tag')
                if item not in temp_sequence:
                    temp_sequence.append(item)
                    temp_sequence.extend(tags)
                else:
                    temp_sequence.extend(tags)
            else:
                if item not in temp_sequence:
                    temp_sequence.append(item)
        sequences_for_word2vec.append(temp_sequence)

    from gensim.models import Word2Vec
    # 使用 Skip-gram 模型訓練商品和類別之間的關係
    model_word2vec = Word2Vec(
        sequences_for_word2vec,
        vector_size=64,
        epochs=10,
        window=10,
        min_count=2,
        sg=1,
        workers = multiprocessing.cpu_count()-1,
        compute_loss=True,
        callbacks=[Callback()],
        # negative=30
    )

    type = 'Children'
    type = 'Sci-Fi'

    user_demo_df = df[df['timestamp'].dt.year>=2016]
    user_demo_df = df[df['userId'] == 'user_77']
    demo_search_rating = user_demo_df[user_demo_df['rating']==5]
    search = demo_search_rating['movieId'].tolist()

    test = searh_similar_by_genre(model=model_word2vec, search_list=['baseball','release_2017'],search_target_df=df_movies, topn=20)
    test = searh_similar_by_genre(model= model_word2vec,search_list=['Children','baseball'],search_target_df=df_movies,topn=20)
    test = searh_similar_by_genre(model= model_word2vec,search_list=['boxing','Action'],search_target_df=df_movies,topn=20)
    test = searh_similar_by_genre(model= model_word2vec,search_list=['3114','2355'],search_target_df=df_movies,topn=20)
    test = searh_similar_by_genre(model= model_word2vec,search_list=['disney','fish'],search_target_df=df_movies,topn=20)

    ##batman
    test = searh_similar_by_genre(model=model_word2vec, search_list=['58559','91529','136864'],search_target_df=df_movies, topn=20)
    test = searh_similar_by_genre(model= model_word2vec,search_list=search,search_target_df=df_movies,topn=20)

    print(test['similar_results']['title'])

    print(tags[tags['tag']=='basketball']['movieId'])

    search_list_0 = ['260','1196','1210']
    search_list_1 = ['34','1','364']
    search_list_2 = ['648','2745','3354','8387']
    kk = model_word2vec.wv.key_to_index
    search_list_1 = ['990578']
    search_list_1 = ['Sci-Fi']

    similars_movies = searh_similar(model_ = model_word2vec,search_list=search_list_1,search_target_df=df_movies,topn=20)
    for i in (similars_movies.get('similar_results')['title'].values.tolist()):
        print(i)
    search_df = similars_movies.get('similar_results')

    userSearch = df[df['userId']==990578].sort_values(by = 'movieId')
    recom = search_df.get('similar_results').sort_values(by= 'movieId')



    ###toys_story
    user = '990148'

    # children
    user = '990006'
    test = model_word2vec.wv.most_similar([user,'Sci-Fi'],topn=30)

    ###by type
    test = model_word2vec.wv.most_similar(['Romance','War'],topn=30)

    ###by type and movieid
    test = model_word2vec.wv.most_similar(['Romance', '84954'], topn=30)

    temp = []
    scores = []
    for item,score in test:
        if item in df_movies['movieId'].values:
            temp.append(item)
            scores.append(float(f'{score:.3}'))
    sorterIndex = dict(zip(temp, range(len(temp))))
    df_similar = df_movies[df_movies['movieId'].isin(sorterIndex)]
    df_similar['movie_Rank'] = df_similar['movieId'].map(sorterIndex)
    df_similar.sort_values(by=['movie_Rank'], ascending=True, inplace=True)
    df_similar.reset_index(drop=True,inplace=True)
    df_similar['score'] = pd.DataFrame(scores)
    for index,title in enumerate(df_similar['title']):
        print(title, scores[index])

    user_view_log[user_view_log['movieId'].isin(temp)]


    df_movies[df_movies['one_genre'].str.contains("Children")]