#new Bayesian data structure
#a tree with nice syntactic sugar for loading data
#and performing aggregated rollups

from collections import defaultdict
import numpy as np
import csv

class Node(defaultdict):
    def __init__(self,dim=2):
        self.dim=dim
        self.data=np.zeros(self.dim,dtype=np.int64)
        self.cache = None
        super(Node, self).__init__(Node)
    def __repr__(self):
        return "Node: data = %s, rollup_cache = %s" % (self.data,self.cache)
        
    def rollup(self,use_cache=True):
        if use_cache and not self.cache == None:
            return self.cache
             
        if len(self) == 0:
            #no children, return data
            res=self.data
        else :
            #recur
            roll=np.zeros(self.dim,dtype=np.int64)
            for k in self.keys():
                roll+=self[k].rollup(use_cache=use_cache)
            res=roll+self.data
    
        self.cache=res
        return res

class BayesCube(Node):
    def __init__(self,hierarchy,target,target_true=1):
        self.nbad=0
        self.target=target
        self.hierarchy=hierarchy
        self.target_true=target_true
        super(BayesCube, self).__init__()
        
    def loader(self,file):
        R=csv.DictReader(open(file,'rU'))
        nlines=0
        for line in R:
            nlines+=1
            target_value=line[self.target]
            target_boolean = (target_value == self.target_true)
            
            #self node to root node
            node=self
            line_is_good=True
            for field in self.hierarchy:
                #walk the tree all the way down
                val=line[field]
                if not(isinstance(val,int) or isinstance(val,str)):
                    #this line is messed up 
                    line_is_good=False
                    continue 
                node=node[val]  
            if line_is_good: 
                node.data+=np.array([1,target_boolean])
            else :
                self.bad_lines+=1    
    
        print 'done loading'
        print '%s lines' % nlines
        print '%s bad lines' % self.nbad     
    
def test_load():
    file="/Users/davej/data/AVSC/test_cube.csv"    
    hierarchy=['offer','chain','market','offerdate']
    target='repeater'
    cube=BayesCube(hierarchy,target,target_true='t')
    cube.loader(file)
    return cube
        
        
          
                