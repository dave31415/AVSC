class OfferRow:
    def __init__(self, row_text):
        row = row_text.split(",")
        self.id = int( row[0] )
        self.category = int( row[1] )
        self.quantity = int( row[2] )
        self.company = int( row[3] )
        self.value = float( row[4] )
        self.brand = int( row[5] )