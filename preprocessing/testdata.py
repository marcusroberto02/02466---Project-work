import gzip
import pandas as pd
import numpy as np
import datetime

"""
df = pd.read_csv('./data/Data_ETH.csv')

#print(len(np.unique(df["Unique_id_collection"])))     
#print(len(np.unique(df["Seller_address"])))
#print(len(np.unique(df["Buyer_address"])))     


date=datetime.datetime(2020,1,1)
df['Datetime_updated'] = pd.to_datetime(df['Datetime_updated'], format='%Y-%m-%d')
print(len(df[df['Datetime_updated'] < date]))
print(len(df[df['Datetime_updated'] >= date]))

print(len(df[df['Datetime_updated'] < date]) / len(df['Datetime_updated']) * 100)
print(len(df[df['Datetime_updated'] >= date]) / len(df['Datetime_updated']) * 100)

"""

sparse_i = np.loadtxt("./data/WAX/2020-07/bi/train/sparse_i.txt")
sparse_j = np.loadtxt("./data/WAX/2020-07/bi/train/sparse_j.txt")

print("hey")