import csv

data = []

with open("data/Data_API.csv","r") as fin:
    reader = csv.reader(fin)
    headers = next(reader)
    data = [{h:x for (h,x) in zip(headers,row)} for row in reader]

print(data[0]["Seller_address"])



