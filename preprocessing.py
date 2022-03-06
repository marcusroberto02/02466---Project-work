import gzip

f = gzip.open("data/Data_API.csv.gz","rb").read()

print(f)

