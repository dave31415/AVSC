from kaggleShopper import *

if __name__=="__main__":
       data_dir= 'data/'
       train_file= 'trainHistory.csv'
       test_file= 'testHistory.csv'
       offer_file= 'offers.csv'
       transaction_file= 'transactions.csv'

       trainhist_offers_file= 'trainhist_offers.csv'
       reduced_file= 'reduced1.csv'
       reduced_dated= 'reduced-dated1.csv'
       transaction_aggr_file= 'transactions-aggr.csv'
       trainhist_offers_transaction_file= 'trainhist_offers_transaggr-dated1.csv'

       testhist_offers_file= 'testhist_offers.csv'
       testhist_offers_transaction_file= 'testhist_offers_transaggr-dated1.csv'

       model_choice=[linear_model.LogisticRegression(),GaussianNB(),svm.SVC(),
                     ExtraTreesClassifier(n_estimators=50), AdaBoostClassifier(n_estimators=50),
                     GradientBoostingClassifier()]
       dense_model= [1,2]
       model_no=0

       model = model_choice[model_no]
       date_to_split_begin= '04-01-2013'
       date_to_split_end= '04-20-2013'
       case= None
       add_data=None
       best_c=None

       if len(sys.argv)!=2:
           print 'python kaggle-shopper.py {wrangling|analyze|both}'
           exit()
       else:
           case=sys.argv[1]

       if case =='wrangling' or case=='both' or case=='both-submit':
          t_i=time.time()
          dw=DataWrangling(data_dir, offer_file, transaction_file)
          #dw.reduce_transaction(outfile=reduced_transaction_file)
          #dw.add_past_purchase_last_n_days(infile=train_file, infile1=reduced_transaction_file,
                                          # outfile=reduced_dated)

          dw.combine_history_offers(infile=train_file, outfile=trainhist_offers_file)
          dw.combine_history_offers(infile=test_file, outfile=testhist_offers_file)

          dw.combine_hist_offer_transaggr_v1(infile=reduced_dated, infile1=trainhist_offers_file,
                                    infile2=testhist_offers_file,outfile1=trainhist_offers_transaction_file,
                                    outfile2=testhist_offers_transaction_file)
          #dw.combine_hist_offer_transaggr(infile= transaction_aggr_file, outfile= trainhist_offers_transaction_file)

          #dw.combine_hist_offer_transaggr(infile= transaction_aggr_file, outfile= testhist_offers_transaction_file)
          print ' time takes to wrangle: ', time.time()-t_i

       if case=='analyze' or case=='both' or case=='analyze-submit' or case=='both-submit':
           t_i=time.time()
           # initialize
           An=AnalyzePredict(data_dir,date_begin=date_to_split_begin, date_end=date_to_split_end)
           # read train
           An.read_aggr_data(file=trainhist_offers_transaction_file, case='train')
           # read test
           An.read_aggr_data(file=testhist_offers_transaction_file, case='test')
           #An.feature_importance(model=model_for_feature_imp)

           # feature gen. of degree n
           #add_data=An.group_data(degree=2)
           # normalize the data
           data_aggr_scaled = An.normalize_data(add_data, case='train')

           if model_no>0: data_aggr_scaled=data_aggr_scaled.toarray()
           # do a hyperparameter search
           best_c, best_auc= An.hyperparam(model, data=data_aggr_scaled, n_cv_loop=1)
           print 'best C, best AUC: ', best_c, best_auc
           # cv test using the optimized hyperparameter
           mean_auc, std_auc=An.cv_loop(model, bestc=best_c, data= data_aggr_scaled, n_cv_loop=8)
           print 'mean, std AUC: ', mean_auc, std_auc
           if case=='analyze' or case=='both': exit()

           # submission on test data
           print "Training full model..."
           totaldata_agr_scaled = An.normalize_data(add_data, case='test')
           preds=An.predict_test(model, totaldata_agr_scaled)
           An.create_test_submission(filename='submit/run6.csv', prediction=preds)
           print 'time takes to analyze: ', time.time() - t_i
