import enum
import pandas as pd
from scipy import sparse
import numpy as np

df = pd.read_csv('./data/small_toy_data_set.csv')[["Unique_id_collection","Seller_address","Buyer_address","Category"]]

df = df.apply(lambda x : pd.factorize(x)[0])

#Removes dublicate trades and increments counter
df = df.groupby(["Unique_id_collection","Seller_address","Buyer_address","Category"]).size().reset_index()
# rename count column
df.columns = ["Unique_id_collection","Seller_address","Buyer_address","Category","Count"]

input = input("Want to save file: y/n: ")
if input == "y":
    df["Unique_id_collection"].to_csv('./data/sparse_tri/sparse_i.txt',header=None,index=None)
    df["Seller_address"].to_csv('./data/sparse_trisparse_j.txt',header=None,index=None)
    df["Buyer_address"].to_csv('./data/sparse_tri/sparse_k.txt',header=None,index=None)
    df["Category"].to_csv('./data/sparse_tri/sparse_v.txt',header=None,index=None)
    df["Count"].to_csv('./data/sparse_tri/sparse_w.txt',header=None,index=None)


print("Unique Sellers:", len(set(df["Seller_address"])))
print("Unique Buyers:", len(set(df["Buyer_address"])))
print("Unique NFTS:", len(set(df["Unique_id_collection"])))
print("Unique Categories:", len(set(df["Category"])))



