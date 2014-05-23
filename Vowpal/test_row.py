class TestRow:
    def __init__(self, row_text):
        row = row_text.split(",")
        self.id = int( row[0] ) 
        self.chain = int( row[1] )
        self.offer = int( row[2] )
        self.market = int( row[3] )
        self.date = row[4]