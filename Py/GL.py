#graphlab

#from readers import *
from graphlab import recommender, SFrame, aws

creds=open('../creds.txt','r').readlines()

aws.set_credentials(creds[0], creds[1])

def test_graphlab():
    ''' test the graphlab install '''
    url='http://s3.amazonaws.com/GraphLab-Datasets/movie_ratings/training_data.csv'    
    data = SFrame(url)
	
    mfac = recommender.matrix_factorization.create(data, 'user', 'movie',7, regularizer=0.05)
                                           
