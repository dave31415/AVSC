import argparse

parser = argparse.ArgumentParser(description = 'Convert CSV file to vw format.')
parser.add_argument("input_file", help = "path to csv input file")
parser.add_argument("output_file", help = "path to output file")
parser.add_argument("-i", "--ignore", help = "columns to ignore", type=str)
parser.add_argument("-l", "--label_index", help = "specify index of label col", type=int, default = -1)
args = parser.parse_args()
def print_line(features, outfile):
    outline = ""
    for k, v in features.items():
        if k == "label":
            outline = str(v) + " '" + features['id'] + " |f" + outline
        elif (k not in args.ignore) and (v != ''):
            outline += " " + k+":"+str(v) 
    outline += "|\n"
    outfile.write( outline )
    
with open(args.input_file, "r") as infile, open(args.output_file, "wb") as outfile:
    headers = []
    for e, line in enumerate(infile):
        if e == 0:
            headers = line.strip().split(",")
        else:
            features = dict()
            row = line.strip().split(",")
            zipped = zip(headers, row)
            for i in zipped:
                features[i[0]] = i[1]
            print_line(features, outfile)