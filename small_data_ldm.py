import pandas
from scipy import sparse

from pandas.api.types import CategoricalDtype

small_dataset = pandas.read_csv('../../simple_ldm/sellers_buyers_NFTs.csv')
df = small_dataset
seller = df["Seller_address"].unique()
buyer = df["Buyer_address"].unique()
shape = (len(seller), len(buyer))

# Create indices for Sellers and Buyers
seller_cat = CategoricalDtype(categories=sorted(seller), ordered=True)
buyer_cat = CategoricalDtype(categories=sorted(buyer), ordered=True)
seller_index = df["Seller_address"].astype(seller_cat).cat.codes
buyer_index = df["Buyer_address"].astype(buyer_cat).cat.codes

# Conversion via COO matrix
coo = sparse.coo_matrix((df["Unique_id_collection"], (seller_index, buyer_index)), shape=shape)

print(coo)