#graphlab

#from readers import *
from graphlab import recommender, SFrame, aws

aws.set_credentials('AKIAIWD7RQTJXCCB72BA', 'E0/g0PtP7d9jkjrSyOKmQNpVKXm6FajpF34A55tN')

def test_graphlab():
    ''' test the graphlab install '''
    data = SFrame('s3:///graphlab-sample-data/example.csv')
    m = recommender.create(data, user='user_id', item='item_id')
    recs = m.recommend(k=5, user_ids=['u1', 'u3', 'u15'])
    print recs

    m = recommender.matrix_factorization.create(data, user='user_id', item='item_id',
            D=7, regularizer=0.05, nmf=True,use_bias=False)
                                        
    m = recommender.item_similarity.create(data, user='user_id', item='item_id',
                                               similarity_type='jaccard')        