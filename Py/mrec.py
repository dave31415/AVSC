from readers import PARS

def test_mrec():
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix
    from mrec.mf.wrmf import WRMFRecommender

    file_format = "csv"
    filepath = PARS['data_dir']+"reduced.v1_numbers.csv"
    outfile = filepath.replace('.csv','_mrec.csv')

    # load training set as scipy sparse matrix
    train = load_sparse_matrix(file_format,filepath)

    model = WRMFRecommender(d=5)
    model.fit(train)
    save_recommender(model,outfile)
    print "wrote model to: %s"%outfile
    print "done"
