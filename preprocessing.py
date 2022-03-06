import csv
import sys

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

data = []

with open("data/Data_API.csv","r") as f:
    reader = csv.reader(f)
    headers = next(reader)
    data = [{h:x for (h,x) in zip(headers,row)} for row in reader]


# create toy data set:
header = ["NFT_ID","Seller_address","Buyer_address"]
N = 1000

with open("data/toyset.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for i in range(N):
        writer.writerow([data[i]["Unique_id_collection"],data[i]["Seller_address"],data[i]["Buyer_address"]])






