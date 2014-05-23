class TransactionRow:
    def __init__(self, row_text):
        row = row_text.split(",")
        self.id = int( row[0] )
        self.chain = int( row[1] )
        self.dept = int( row[2] )
        self.category = int( row[3] )
        self.company = int( row[4] )
        self.brand = int( row[5] )
        self.date = row[6]
        self.productsize = row[7]
        self.productmeasure = row[8]
        self.purchasequantity = int( row[9] )
        self.purchaseamount = float( row[10] )