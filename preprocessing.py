import csv
from enum import unique
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

unique_sellers = set()

with open("data/toyset.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    i = 0
    while len(unique_sellers) < 1000 and i <= 1000000:
        nft_id = data[i]["Unique_id_collection"]
        seller_id = data[i]["Seller_address"]
        buyer_id = data[i]["Buyer_address"]
        unique_sellers.add(seller_id)
        writer.writerow([nft_id,seller_id,buyer_id])
        i += 1
        if i % 1000 == 0:
            print(i,len(unique_sellers))


print(len(data))


