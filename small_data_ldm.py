import pandas
from scipy import sparse

from pandas.api.types import CategoricalDtype

small_dataset = pandas.read_csv('../../simple_ldm/sellers_buyers_NFTs.csv')
df = small_dataset
seller = df["Seller_address"].unique()
buyer = df["Buyer_address"].unique()
shape = (len(seller), len(buyer))

# Create indices for users and movies
seller_cat = CategoricalDtype(categories=sorted(seller), ordered=True)
buyer_cat = CategoricalDtype(categories=sorted(buyer), ordered=True)
seller_index = df["Seller_address"].astype(seller_cat).cat.codes
buyer_index = df["Buyer_address"].astype(buyer_cat).cat.codes

# Conversion via COO matrix
coo = sparse.coo_matrix((df["Unique_id_collection"], (seller_index, buyer_index)), shape=shape)
csr = to_csr

#adj_matrix = pd.crosstab(small_dataset['Seller_address'], small_dataset['Buyer_address'])
#idx = adj_matrix.columns.union(adj_matrix.index)
#adj_matrix = adj_matrix.reindex(index = idx, columns=idx, fill_value=0)
#print(adj_matrix)
print(coo)