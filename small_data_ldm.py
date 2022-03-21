import pandas
from scipy import sparse
import numpy as np

small_dataset = pandas.read_csv('data/toy.csv')
df = small_dataset

#Creates an index for all of the Traders
Trader_ids = {j: i for i,j in enumerate(set(df.iloc[:,1]))}
#Creates an index for all of the NFTS
NFT_ids = {j:i for i,j in enumerate(set(df.iloc[:,3]))}

#Pairs trades
new_df = pandas.DataFrame({'groups': list(zip(df.iloc[:,1],df.iloc[:,3])),'count':1})
print(new_df.shape)
#Removes dublicate trades and incremetns counter
new_df = new_df.groupby('groups').sum()
print(new_df.shape)
#Creates new columns with the respective ids for the traders and nfts
new_df['Traders_idx'] = [Trader_ids[Trader[0]] for Trader in new_df.index]
new_df['Nft_idx'] = [NFT_ids[NFT[1]] for NFT in new_df.index]

input = input("Want to save file: y/n: ")
if input == "y":
    new_df.to_csv('data/COO_matrix.csv')
print(new_df[new_df['count']>1])

print("Unique Trader:", len(Trader_ids))
print("Unique NFTS:", len(NFT_ids))
print("Check in new_df:")
print(len(new_df['Traders_idx']))
print(len(new_df['Nft_idx']))



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