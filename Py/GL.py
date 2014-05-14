#graphlab

#from readers import *
from graphlab import recommender, SFrame, aws

creds=open('../creds.txt','r').readlines()

aws.set_credentials(creds[0], creds[1])

def test_graphlab(num_factors=10,reg=0.01,niter=50):
    ''' test the graphlab install '''
    url='http://s3.amazonaws.com/GraphLab-Datasets/movie_ratings/training_data.csv'    
    data = SFrame(url)
    mfac = recommender.matrix_factorization.create(data, 'user', 'movie','rating',num_factors, 
            reg=reg,nmf=True,use_bias=True,holdout_probability=0.2,niter=niter,random_seed=42)
    print mfac.summary
    return mfac

def test_graphlab2(num_factors=10,reg=0.01,niter=50):
    ''' test the graphlab install with our data'''
    url='http://s3.amazonaws.com/GraphLab-Datasets/movie_ratings/training_data.csv'
    data = SFrame(url)
    mfac = recommender.matrix_factorization.create(data, 'user', 'movie','rating',num_factors,
            reg=reg,nmf=True,use_bias=True,holdout_probability=0.2,niter=niter,random_seed=42)
    print mfac.summary
    return mfac
