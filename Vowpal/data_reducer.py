from datetime import datetime, date

loc_offers = '../data/offers.csv'
loc_transactions = '../data/transactions.csv'
loc_reduced_to_tenth = '../data/reduced_to_tenth.csv'
loc_reduced = '../data/reduced.csv'

class DataReducer:
    def __init__(self, loc_offers, loc_transactions, loc_reduced, loc_reduced_to_tenth):
        self.loc_offers = loc_offers
        self.loc_transactions = loc_transactions
        self.loc_reduced = loc_reduced
        self.loc_reduced_to_tenth = loc_reduced_to_tenth

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
                if e % 500000 == 0:
                    print e, datetime.now() - start
        print datetime.now() - start
        return self
        
        
    def reduce_by_ten(self, loc_transactions):
        with open(self.loc_reduced_to_tenth, "wb") as outfile, open(loc_transactions) as infile:
            start = datetime.now()
            print start
            next(infile)
            int_p = int
            my_split = str.split      
            for e, line in enumerate(infile):
                id = int_p(my_split(line,",",1)[0])
                if (id % 10) == 2:
                    outfile.write(line)
                if e % 5000000 == 0:
                    print e, datetime.now() - start
        return self
    
if __name__ == '__main__':
    #DataReducer(loc_offers, loc_transactions, loc_reduced, loc_reduced_to_tenth).reduce_by_ten(loc_transactions).reduce_data(loc_reduced_to_tenth)
    DataReducer(loc_offers, loc_transactions, loc_reduced, loc_reduced_to_tenth).reduce_data(loc_transactions)
