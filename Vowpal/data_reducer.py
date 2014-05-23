from datetime import datetime, date

loc_offers = '../data/offers.csv'
loc_transactions = '../data/transactions.csv'
loc_reduced = '../data/reduced.csv'

class DataReducer:
    def __init__(self, loc_offers, loc_transactions, loc_reduced):
        self.loc_offers = loc_offers
        self.loc_transactions = loc_transactions
        self.loc_reduced = loc_reduced

    def reduce_data(self, loc_transactions):
        start = datetime.now()
        print start
        #get all categories and comps on offer in a dict
        offers_category = {}
        for e, line in enumerate( open(self.loc_offers) ):
            offers_category[ line.split(",")[1] ] = 1
        with open(self.loc_reduced, "wb") as outfile:
            for e, line in enumerate( open(loc_transactions) ):
                cols = line.split(",")
                if e == 0:
                    outfile.write( line ) #print header
                elif cols[3] in offers_category:
                    outfile.write( line )
                #progress
                if e % 5000000 == 0:
                    print e, datetime.now() - start
        print datetime.now() - start
        return self
        
        
    def reduce_by_ten(self, loc_transactions):
        with open(self.loc_reduced, "wb") as outfile, open(loc_transactions) as infile:
            next(infile)
            int_p = int
            my_split = str.split
            my_filter = lambda x: int_p(my_split(x,",", 1)[0]) % 10 == 2
            writer = lambda x: outfile.write(x)
            map(writer,filter(my_filter, infile))        
        return self
    
if __name__ == '__main__':
    DataReducer(loc_offers, loc_transactions, loc_reduced).reduce_by_ten(loc_transactions)
    #DataReducer(loc_offers, loc_transactions, loc_reduced).reduce_data(loc_reduced)
