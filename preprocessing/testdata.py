import gzip
import pandas as pd
import numpy as np
import datetime

#df = pd.read_csv('./data/Data_API.csv')

#print(len(np.unique(df["Unique_id_collection"])))     
#print(len(np.unique(df["Seller_address"])))
#print(len(np.unique(df["Buyer_address"])))     

#start_date=datetime.datetime(2021,2,1)
#end_date=datetime.datetime(2021,3,1)
#df["Datetime_updated"] = pd.to_datetime(df['Datetime_updated'], format='%Y-%m-%d')
#df = df.loc[df["Datetime_updated"] >= start_date]
#df = df.loc[df["Datetime_updated"] < end_date]
#df = df.groupby(["Unique_id_collection","Category"]).size().reset_index()
#categories = ["Art","Collectible","Games","Metaverse","Other","Utility"]
#print([sum(df["Category"] == c) for c in categories])
#print([sum(df["Category"] == c) / len(df["Category"])*100 for c in categories])
#date=datetime.datetime(2020,1,1)
#df['Datetime_updated'] = pd.to_datetime(df['Datetime_updated'], format='%Y-%m-%d')
#print(len(df[df['Datetime_updated'] < date]))
#print(len(df[df['Datetime_updated'] >= date]))

#print(len(df[df['Datetime_updated'] < date]) / len(df['Datetime_updated']) * 100)
#print(len(df[df['Datetime_updated'] >= date]) / len(df['Datetime_updated']) * 100)


sparse_w_bi = np.loadtxt("./data/ETH/2021-03/bi/test/sparse_w.txt",dtype=int)

max_w_bi = np.max(sparse_w_bi)

for i in range(1,max_w_bi+1):
    print(f"{i}: {np.sum([sparse_w_bi == i])}")

print("\n")

sparse_w_tri = np.loadtxt("./data/ETH/2021-03/tri/test/sparse_w.txt",dtype=int)

max_w_tri = np.max(sparse_w_tri)

for i in range(1,max_w_tri+1):
    print(f"{i}: {np.sum([sparse_w_tri == i])}")

