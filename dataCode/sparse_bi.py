import enum
import pandas
from scipy import sparse
import numpy as np

small_dataset = pandas.read_csv('./data/small_toy_data_set.csv')
df = small_dataset

#Creates an index for all of the NFTS
NFT_ids = {NFT : i for i, NFT in enumerate(dict.fromkeys(df['Unique_id_collection']).keys())}
#Creates an index for all of the Sellers
Seller_ids = {Seller : i for i, Seller in enumerate(dict.fromkeys(df['Seller_address']).keys())}
#Creates an index for all of the Buyers
Buyer_ids = {Buyer : i for i, Buyer in enumerate(dict.fromkeys(df['Buyer_address']).keys())}
#Trader ids
Trader_ids = {Trader : i for i, Trader in enumerate(dict.fromkeys(list(df['Seller_address'])+list(df['Buyer_address'])).keys())}


NFT_list = [nft for nft in df["Unique_id_collection"]] + [nft for nft in df["Unique_id_collection"]]
Trader_list = [seller for seller in df["Seller_address"]] + [buyer for buyer in df["Buyer_address"]]
temp_df = pandas.DataFrame({"NFT_idx" : NFT_list, "Trader_idx" : Trader_list})

#Pairs trades
new_df = pandas.DataFrame({'groups': list(zip(temp_df['NFT_idx'],temp_df["Trader_idx"])),'count': 1})
print(new_df.shape)
#Removes dublicate trades and increments counter
new_df = new_df.groupby('groups').sum()
print(new_df.shape)
#Creates new columns with the respective ids for the traders and nfts
new_df.insert(0,"NFT_idx",[NFT_ids[trade[0]] for trade in new_df.index])
new_df.insert(1,"Trader_idx",[Trader_ids[trade[1]] for trade in new_df.index])

input = input("Want to save file: y/n: ")
if input == "y":
    new_df["NFT_idx"].to_csv('./data/sparse_bi/sparse_i.txt',header=None,index=None)
    new_df["Trader_idx"].to_csv('./data/sparse_bi/sparse_j.txt',header=None,index=None)
    new_df["count"].to_csv('./data/sparse_bi/sparse_w.txt',header=None,index=None)

print("Unique traders:", len(Trader_ids))
print("Unique NFTS:", len(NFT_ids))
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