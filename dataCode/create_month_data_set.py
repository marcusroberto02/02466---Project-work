import os
import pandas
import datetime

# link til data 
# https://osf.io/wsnzr/?view_only=319a53cf1bf542bbbe538aba37916537

dataset = pandas.DataFrame()
i = 0
for chunk in pandas.read_csv("./data/Data_API.csv", chunksize=10000, parse_dates=[18]):
    temp = chunk[chunk["Datetime_updated"] >= datetime.datetime(2017,11,1)]
    dataset = pandas.concat([dataset, temp[temp["Datetime_updated"] < datetime.datetime(2017,12,1)]])

# sellers_buyers_table = small_dataset[['Seller_address','Buyer_address','Unique_id_collection']]

dataset.to_csv('./data/2018_01/train.csv')

