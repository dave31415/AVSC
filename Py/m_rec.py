from readers import PARS
import time

def test_mrec(d=5,num_iters=3,reg=0.015):
    #d is dimension of subspace, i.e. groups
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix
    from mrec.mf.wrmf import WRMFRecommender

    alpha=1.0
    suffix="_mrec_d%s_iter%s_reg%0.4f.npz"%(d,num_iters,reg)
    start=time.time()

    file_format = "csv"
    #file shoule be csv, with: row,col,data
    #data may just be ones
    filepath = PARS['data_dir']+"reduced.v1_numbers.csv"
    #filepath = PARS['data_dir']+"test_10_mill.csv" 
    outfile = filepath.replace('.csv',suffix)
    print outfile
    print 'reading file: %s'%filepath
    # load training set as scipy sparse matrix
    print "loading file"
    train = load_sparse_matrix(file_format,filepath)
    print "loaded file"
    print (time.time()-start),"seconds"
    print "size:",train.shape

    print "creating recommender"
    model = WRMFRecommender(d=5,num_iters=num_iters,alpha=alpha,lbda=reg)
    print "training on data"
    print time.time()-start
    model.fit(train)
    print "done training"
    print time.time()-start
    print "saving model"
    save_recommender(model,outfile)
    print "wrote model to: %s"%outfile
    print "done"
    run_time=(time.time()-start)/60.0
    print "runtime: %0.3f minutes"%run_time

