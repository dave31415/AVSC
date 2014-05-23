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
        offers_company = {}
        offers_brand = {}
        for e, line in enumerate( open(self.loc_offers) ):
            offers_category[ line.split(",")[1] ] = 1
            offers_company[ line.split(",")[3] ] = 1
            offers_brand[ line.split(",")[5] ] = 1
        with open(self.loc_reduced, "wb") as outfile:
            for e, line in enumerate( open(loc_transactions) ):
                cols = line.split(",")
                if e == 0:
                    outfile.write( line ) #print header
                elif cols[3] in offers_category or cols[4] in offers_company or cols[5] in offers_brand:
                    outfile.write( line )
                #progress
                if e % 5000000 == 0:
                    print e, datetime.now() - start
        print datetime.now() - start
        return self
        
    def reduce_by_ten(self, loc_transactions):
        with open(self.loc_reduced, "wb") as outfile:
            start = datetime.now()
            print start
            for e, line in enumerate( open(loc_transactions) ):
                if e > 0:
                    row = line.split(",")
                    id = int( row[0] )
                    if id % 10 == 2:
                        outfile.write( line ) #print header
                    if e % 5000000 == 0:
                        print e, datetime.now() - start
        return self
    
if __name__ == '__main__':
    generate_features(loc_train, loc_test, loc_reduced, loc_out_train, loc_out_test)
