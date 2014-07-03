import matplotlib
#matplotlib.use('Agg') # Plots go to files not screen    

from readers import PARS, make_item_category_company_brand,make_customer_offer_lookup
from readers import data_files,stream_data
import time
from mrec import load_recommender
import numpy as np
import mrec
import matplotlib.pyplot as plt
import csv
from scipy.cluster.vq import kmeans,vq
from readers import roc as roc_curve

def make_mrec_outfile(infile,d,num_iters,reg):
    suffix="_mrec_d%s_iter%s_reg%0.4f.npz"%(d,num_iters,reg)
    outfile = infile.replace('.csv',suffix)
    return outfile

def recall_precision_plot(thresh,recall,precision,plot_file=None):
    ep=1e-11
    prec=np.array(precision)
    rec=np.array(recall)
    fscore=2*prec*rec/(prec+rec+ep)
    fig=plt.figure()
    plt.plot(thresh,precision,'ro')
    plt.plot(thresh,recall,'bo')
    plt.plot(thresh,fscore)
    plt.legend(['precision','recall','fscore'])
    
    if plot_file: 
        print 'saving plot to: %s'%plot_file
        fig.savefig(plot_file)
    else :
        print 'not plotting to a file'
        #plt.show()
        pass

def check_validation(outfile='reduced.v1_numbers_mrec_d5_iter15_reg0.0150.npz'):
    #outfile without path TODO: fix
    data,U,V=read_mrec(outfile)
    model=np.dot(U,V.transpose())
    plot_file=outfile.replace('.npz','.png')
    vals=multi_thresh(data,model,plot_file=plot_file)
    return vals

def multi_thresh(data,model,thresh_list=None,plot_file=None):
    import pickle
    if thresh_list==None: thresh_list=np.linspace(0.1,1.,7)
    vals=[]
    print " thresh  precision  recall  fscore"
    print "-----------------------------------------"
    for thresh in thresh_list:
        val=validate_matrices(data,model,thresh=thresh)
        val['thresh']=thresh

        vals.append(val)
        line="%0.4f  %0.4f  %0.4f  %0.4f)"%(thresh, val['precision'],val['recall'],val['fscore'])
        print line
    
    pickle_file=plot_file.replace('.png','.pickle')    
    pickle.dump(vals,open('data.pickle','wb'))
    thresh=[d['thresh'] for d in vals]
    precision=[d['precision']for d in vals]
    recall=[d['recall']for d in vals]
    recall_precision_plot(thresh,recall,precision,plot_file=plot_file)

    return vals

def threshing(M_sparse,U,V,thresh=0.5):
    ''' 
    Calculate the sufficient statistics between a 
    sparse matrix and a UV decomposed one  
    We want: 
    d= sum_of M_sparse
    c= sum_of ((M_sparse * U*V^T) > thresh)
    s= sum_of (U*V^T > thresh)
    These can be used to calculate precision, recall, AUROC etc
    for matrix factorization models without expanding into full matrices
    assumes M_sparse is a list of (row,col) tupes where the value is 1 , else 0
    U and V are the U-V decomposition via a matrix factorization.
    This should be both memory efficient and fast, thus, allowing for huge data sets.
    TODO:this needs work
    '''
       
    num_U=U.shape[0]
    num_V=V.shape[0]
    num_tot=float(num_U*num_V)
    num_d=U.shape[1]
    assert(V.shape[1] == num_d)

    mean_data=len(M_sparse)/num_tot
    print 'mean data: %s'%mean_data

    sum_cross=0.0
    for row, col in M_sparse:
        u=U[row,:]
        v=V[col,:]
        sum_cross+= (np.dot(u,v))> thresh
        
    x=[d[0] for d in M_sparse]
    y=[d[1] for d in M_sparse]
    U_filt=U[x,:]
    V_filt=V[y,:]
    

    mean_cross=sum_cross/num_tot
    print 'mean cross: %s'%mean_cross

    sum_model=0.0
    for i in xrange(num_U):
        sum_model+=(np.dot(V,U[i,:]) > thresh).sum()
    sum_cross2=0.0
    for i in x:
        sum_cross2+=(np.dot(V_filt,U[i,:]) > thresh).sum()
    mean_cross=sum_cross2/num_tot
    print 'mean cross2: %s'%mean_cross2
        

    mean_model=sum_model/num_tot
    print 'mean model: %s'%mean_model
    #return means not sums
    return (mean_data,mean_model,mean_cross)

def validate_matrices(data,Model,thresh=0.5,show=False):
    model=np.dot(Model.U,Model.V.transpose())
    rms=np.sqrt(((data-model)**2).mean())
    mean_data=data.mean()
    mean_model=model.mean()
    mean_ratio=mean_model/float(mean_data)
    sim=(model>thresh)*1
    mean_sim=sim.mean()
    mean_cross=(data*sim).mean()

    TP=mean_cross                                                                                
    TN=1-mean_data-mean_sim+mean_cross                                                          
    FP=mean_sim-mean_cross                                                                                 
    FN=mean_data-mean_cross     

    ep=1e-11
    precision=TP/(TP+FP+ep)
    recall=TP/(TP+FN+ep)
    fallout=FP/(FP+TN+ep)
    fscore=2*precision*recall/(precision+recall)

    valid={'rms':rms,'mean_data':mean_data,'mean_model':mean_model,'mean_ratio':mean_ratio,
           'TP':TP,'FP':FP,'TN':TN,'FN':FN,'mean_cross':mean_cross,
           'precision':precision,'recall':recall,'fallout':fallout,'fscore':fscore}
    if show:
        for k,v in valid.iteritems(): print k,":",v
    return valid

def read_mrec(mrec_file='reduced.v1_numbers_mrec_d5_iter9_reg0.0150.npz'):
    file_name=mrec_file
    data_file_name=file_name.split('_mrec_')[0]+'.csv'
    model=mrec.load_recommender(file_name)
    U=model.U
    V=model.V
    model_matrix=np.dot(U,V.transpose())
    shape=model_matrix.shape
    shape=(U.shape[0],V.shape[0])
    data_matrix=np.ndarray(shape,dtype=int)
    line_num=0
    for line in open(data_file_name,'r'):
        line_num+=1
        if line_num % 1000000 ==0 : print line_num
        dat=line.strip().split(',')
        row=int(dat[0])-1
        col=int(dat[1])-1
        val=int(float(dat[2]))
        data_matrix[row,col]=val
    return (data_matrix,U,V)

def cut_number_file(cutoff=1.5):
    filepath = PARS['data_dir']+"/reduced_row_col_num.csv"
    f=open(filepath,'rU')
    outfile=filepath.replace('.csv','_cutoff_%s.csv'%cutoff)
    fout=open(outfile,'w')
    for line in f:
        data=line.strip().split(',')
        num=float(data[2])
        if num> cutoff: fout.write(line)
    f.close()
    fout.close()    

def run_mrec(d=10,num_iters=4,reg=0.02):
    #d is dimension of subspace, i.e. groups
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix
    from mrec.mf.wrmf import WRMFRecommender

    alpha=1.0
    start=time.time()

    file_format = "csv"
    #file shoule be csv, with: row,col,data
    #data may just be ones
    filepath = PARS['data_dir']+"/reduced_row_col_num_cutoff_1.5.csv"
    #filepath = PARS['data_dir']+"test_10_mill.csv" 
    outfile = make_mrec_outfile(filepath,d,num_iters,reg)
    print outfile
    print 'reading file: %s'%filepath
    # load training set as scipy sparse matrix
    print "loading file"
    train = load_sparse_matrix(file_format,filepath)
    print "loaded file"
    print (time.time()-start),"seconds"
    print "size:",train.shape

    print "creating recommender"
    model = WRMFRecommender(d=d,num_iters=num_iters,alpha=alpha,lbda=reg)
    print "training on data"
    print time.time()-start
    model.fit(train)
    print "done training"
    print time.time()-start
    print "saving model"
    save_recommender(model,outfile)
    print "wrote model to: %s"%outfile
    print time.time()-start

    return

    print "validating"
    data,U,V=read_mrec(mrec_file=outfile)
    plot_file=outfile.replace('.npz','.png')
    multi_thresh(data,model,thresh_list=None,plot_file=plot_file)
    run_time=(time.time()-start)/60.0
    print "runtime: %0.3f minutes"%run_time
    print 'done'

class junk:
    pass

class mf_model:
    def __init__(self,d=10,num_iters=4,reg=0.02):
        #TODO: clean this up
        filepath = PARS['data_dir']+"/reduced_row_col_num_cutoff_1.5.csv"
        file_name='/Users/davej/data/AVSC/reduced.csv'
        
        self.model_file = make_mrec_outfile(filepath,d=d,num_iters=num_iters,reg=reg)
        self.dictfile_user=file_name.replace('.csv','_dict_user.csv')
        self.dictfile_item=file_name.replace('.csv','_dict_item.csv')
        print "loading model in : %s" % self.model_file
        self.model=load_recommender(self.model_file)
        print "loading dictionaries"
        self.dict_user=dict(list(csv.reader(open(self.dictfile_user,'rU'))))
        self.dict_item=dict(list(csv.reader(open(self.dictfile_item,'rU'))))
        self.nbad=0
        #kmeans stuff
        self.k_default=75
        self.alpha=5.0
        self.mu=0.31
    
    def features(self,offer):
        bad=False
        user=offer['id']
        item=make_item_category_company_brand(offer)
        if user not in self.dict_user:
            print 'Warning- unknown user : %s' % user
            bad=True
        if item not in self.dict_item:
            print 'Warning- unknown item : %s' % item
            bad=True  
            
        if bad :
            self.nbad+=1
            return np.zeros(self.model.d)
            
        user_num=int(self.dict_user[user])
        item_num=int(self.dict_item[item])
        row=user_num-1
        col=item_num-1
        if row < self.model.U.shape[0] and col < self.model.U.shape[0]:
            u=self.model.U[row,:]
            v=self.model.V[col,:]
        else :
            print "Warning - row=%s, col=%s not in range of data" % (row,col)
            u=np.zeros(self.model.d)
            v=np.zeros(self.model.d)
        #return the point-wise product NOT dot product
        return u*v

    def score(self,offer):
        scores=[]
        score_min=1e-8   # greater than zero
        score_max=10.0   # might not be needed
        features=self.features(offer)
        raw_score=features.sum()
        score=min(max(raw_score,score_min),score_max)
        return score

    def add_features_to_dic(self,data):
        #data a dictionary
        ndim=self.model.d
        feature_names=['MF'+str(i) for i in range(ndim)]
        feature_names.append('score')
        features=self.features(data)
        score=features.sum()
        features=list(features)
        features.append(score)
        #add to dict
        #make float with trimmed digits
        features_str=["%0.6f"%i for i in features]
        
        for i in range(ndim+1): data[feature_names[i]]=features_str[i]
        #modified state, no need to return
        return features[0:-1]

    def add_features_to_files(self,name='history',return_features=False):
        self.nbad=0
        file=data_files(name)
        outfile=file.replace('.csv','_with_MF_features.csv')
        if name == 'reduced':
            data=stream_data('reduced')
        else :
            data=make_customer_offer_lookup(name).itervalues()
            
        key_names=None
        nlines=0
        custs=set()
        feature_list=[]
        for dat in data:
            nlines+=1
            custs.add(dat['id'])
            features=self.add_features_to_dic(dat)
            
            if return_features: feature_list.append(features)
            if not key_names:
                #first one
                key_names=dat.keys()
                W=csv.DictWriter(open(outfile,'w'),key_names)
                W.writeheader()
            
            W.writerow(dat)
        print 'nlines : %s' % nlines    
        print 'n customers : %s' % len(custs)
        print 'nbad : %s' % self.nbad
        self.nbad=0
        if return_features: return(np.array(feature_list))

    def add_features_to_both(self):
        self.add_features_to_files(name='history')
        self.add_features_to_files(name='history_test')

    def kmeans(self,k=None):
        #think about whether whitening is needed
        if k == None : 
            k=self.k_default
        self.k=k
            
        features_train=self.add_features_to_files(name='history',return_features=True)
        features_test=self.add_features_to_files(name='history_test',return_features=True)
        #combine them
        n_train=features_train.shape[0]
        n_test=features_test.shape[0]
        n_combined=n_train+n_test
        features=np.zeros(shape=(n_combined,self.model.d))
        for i in xrange(n_train):
            features[i]=features_train[i]
        for i in xrange(n_test):
            features[i+n_train]=features_test[i]
            
        print "Running kmeans with k=%s" % self.k
        centroids,_ = kmeans(features,k)
        # assign each sample to a cluster
        self.centroids=centroids
        idx=self.kmeans_assign(features)
        print "done kmeans"
        
    def kmeans_assign(self,features):
        idx,_ = vq(features,self.centroids)
        return idx
        
    def calibrate_kmeans(self):
        features=self.add_features_to_files(name='history',return_features=True)
        data=make_customer_offer_lookup(name='history').itervalues()
        n_train=features.shape[0]
        idx=self.kmeans_assign(features)
        num_tot=np.zeros(n_train)
        num_repeater=np.zeros(n_train)
        for i in xrange(n_train):
            num_tot[idx[i]]+=1.0
            d=data.next()
            if d['repeater']=='t':
                num_repeater[idx[i]]+=1.0
        #use additive (Laplace) smoothing
        frac_repeater=(num_repeater+self.alpha)/(num_tot+self.alpha/self.mu)
        kmeans_dic={}
        
        for i in xrange(self.k):
            dic={'idx': i, 'num_train':num_tot[i],'num_train_repeat':num_repeater[i],
                'frac_repeater':frac_repeater[i]}
            kmeans_dic[i] = dic
        self.kmeans_dic=kmeans_dic
        
        
    def frac_repeater_idx(self,idx): 
        return self.kmeans_dic[idx]['frac_repeater']

    def score_offer(self,offer):
        #compute features, not tested
        features=self.features(offer)
        #assign a kmeans cluster index
        idx=self.kmeans_assign(features)
        #look up fraction of repeaters (in train) for this cluster
        frac=self.kmeans_dic[idx]['frac_repeater']
        return frac,idx

    def plot_fracs(self):
        d=sorted(self.kmeans_dic.values(),key=lambda x : x['frac_repeater'])
        frac=np.array([i['frac_repeater'] for i in d])
        num=np.array([i['num_train'] for i in d])
        num_repeat=np.array([i['num_train_repeat'] for i in d])
        #calculate errors, find my formula from Catalina writeup which included Laplace term
        print "cluster fraction  num_tot num_repeat"
        for i in xrange(len(frac)):
            print i,frac[i],num[i],num_repeat[i]
        plt.plot(frac)
        plt.show()
        
    def roc(self):
        features=self.add_features_to_files(name='history',return_features=True)
        data=make_customer_offer_lookup(name='history').itervalues()
        n_train=features.shape[0]
        idx=self.kmeans_assign(features)
        frac=np.array([self.kmeans_dic[i]['frac_repeater'] for i in idx])
        actual=np.array([d['repeater'] == 't' for d in data])
        roc_curve(frac,actual)

    def write_feature_files(self):
        file_name='/Users/davej/data/AVSC/reduced.csv'
        outfile_name=file_name.replace('.csv','_MF_kmeans.csv')
        W=open(outfile_name,'w')
        W.write("id,cluster,frac_repeater\n")
    
        for name in ['history','history_test']:    
            data=make_customer_offer_lookup(name=name).itervalues()
            features=self.add_features_to_files(name=name,return_features=True)
            idx=self.kmeans_assign(features)
            frac=[self.frac_repeater_idx(i) for i in idx] 
            for d,i,f in zip(data,idx,frac):
                id=d['id']
                W.write("%s,%s,%s\n" % (id,i,f))
    
    def make_submission(self):
        #got  0.51096 on first submission try
        file_name='/Users/davej/data/AVSC/reduced.csv'
        outfile_name=file_name.replace('.csv','_MF_kmeans_submission.csv')
        W=open(outfile_name,'w')
        W.write("id,repeatProbability\n")
    
        for name in ['history_test']:    
            data=make_customer_offer_lookup(name=name).itervalues()
            features=self.add_features_to_files(name=name,return_features=True)
            idx=self.kmeans_assign(features)
            frac=[self.frac_repeater_idx(i) for i in idx] 
            for d,i,f in zip(data,idx,frac):
                id=d['id']
                W.write("%s,%s\n" % (id,f*100.0))
    

    def build(self):
        print "Building kmeans and repeater calibration with k=%s, alpha=%s, mu=%s" % (self.k_default,self.alpha,self.mu)
        self.kmeans()
        self.calibrate_kmeans()  
    
    def doall(self):
        self.build()
        self.write_feature_files()
        self.make_submission()
        print 'did all'

def test_mf_train():
    train=make_customer_offer_lookup(name='history')
    model=mf_model()
    
    dic={}
    for cust, data in train.iteritems() :
        score=model.score(data)
        data['score']=score
        dic[cust]=data
    return dic
    
    
    
    
    
    
    
    
    
    
    




