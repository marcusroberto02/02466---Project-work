import enum
import pandas
from scipy import sparse
import numpy as np

small_dataset = pandas.read_csv('./data/small_toy_data_set.csv')
df = small_dataset

print

#Creates an index for all of the NFTS
NFT_ids = {NFT : i for i, NFT in enumerate(dict.fromkeys(df.iloc[:,3]).keys())}
#Creates an index for all of the Traders
Trader_ids = {Trader : i for i, Trader in enumerate(dict.fromkeys(df.iloc[:,1]).keys())}

#Pairs trades
new_df = pandas.DataFrame({'groups': list(zip(df.iloc[:,3],df.iloc[:,1])),'count': 1})
print(new_df.shape)
#Removes dublicate trades and increments counter
new_df = new_df.groupby('groups').sum()
print(new_df.shape)
#Creates new columns with the respective ids for the traders and nfts
new_df.insert(0,"NFT_idx",[NFT_ids[NFT[0]] for NFT in new_df.index])
new_df.insert(1,"Trader_idx",[Trader_ids[Trader[1]] for Trader in new_df.index])

input = input("Want to save file: y/n: ")
if input == "y":
    new_df.to_csv('data/sparse_bi/sparse_matrix_bi_toy.csv',index=None)
    new_df["NFT_idx"].to_csv('./data/sparse_bi/sparse_i.txt',header=None,index=None)
    new_df["Trader_idx"].to_csv('./data/sparse_bi/sparse_j.txt',header=None,index=None)
    new_df["count"].to_csv('./data/sparse_bi/sparse_w.txt',header=None,index=None)

# print("Unique Trader:", len(Trader_ids))
# print("Unique NFTS:", len(NFT_ids))
# print("Check in new_df:")
# print(len(new_df['NFT_idx']))
# print(len(new_df['Trader_idx']))



"""
from pandas.api.types import CategoricalDtype


seller = df["Seller_address"].unique()
NFT_id = df["Unique_id_collection"].unique()
shape = (len(seller), len(NFT_id))

# Create indices for users and movies
seller_cat = CategoricalDtype(categories=sorted(seller), ordered=True)
NFT_cat = CategoricalDtype(categories=sorted(NFT_id), ordered=True)
seller_index = df["Seller_address"].astype(seller_cat).cat.codes
NFT_index = df["Unique_id_collection"].astype(NFT_cat).cat.codes
# Conversion via COO matrix
coo = sparse.coo_matrix((np.ones(len(seller_index)),(seller_index, NFT_index)), shape=shape)

rows = set(coo.row)
cols = set(coo.col)
"""