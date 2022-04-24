import enum
import pandas as pd
from scipy import sparse
import numpy as np

# path to data folder
data_path = "./data/"
dataset = "small_toy_data_set.csv"

df = pd.read_csv(data_path + dataset)[["Unique_id_collection","Seller_address","Buyer_address","Category"]]

# combine seller and buyer address into traders
data_dict = {"Unique_id_collection" : df["Unique_id_collection"]+df["Unique_id_collection"],
             "Trader_address" : df["Seller_address"]+df["Buyer_address"],
             "Category" : df["Category"]+df["Category"]}

df = pd.DataFrame(data_dict)

df = df.apply(lambda x : pd.factorize(x)[0])

#Removes dublicate trades and increments counter
df = df.groupby(["Unique_id_collection","Trader_address","Category"]).size().reset_index()
# rename count column
df.columns = ["Unique_id_collection","Trader_address","Category","Count"]

# save files
df["Unique_id_collection"].to_csv(data_path + 'sparse_bi/sparse_i.txt',header=None,index=None)
df["Trader_address"].to_csv(data_path + 'sparse_bi/sparse_j.txt',header=None,index=None)
df["Count"].to_csv(data_path + 'sparse_bi/sparse_w.txt',header=None,index=None)

#print("Unique Traders:", len(set(df["Trader_address"])))
#print("Unique NFTS:", len(set(df["Unique_id_collection"])))
#print("Unique categories", len(set(df["Category"])))

