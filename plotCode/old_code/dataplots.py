import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt

df = pd.read_csv("Data/Data_API.csv", low_memory=True)
print(df.describe())


seller = df["Seller_address"].unique()
buyer = df["Buyer_address"].unique()
nft = df["Unique_id_collection"].unique()

traders = np.append(seller,buyer)
trad_df=pd.DataFrame(traders)
trads = trad_df[0].unique()

trader_cat = CategoricalDtype(categories=sorted(trads), ordered=True)
nft_cat = CategoricalDtype(categories=sorted(nft), ordered=True)

seller_index = df["Seller_address"].astype(trader_cat).cat.codes
buyer_index = df["Buyer_address"].astype(trader_cat).cat.codes

nft_index = df["Unique_id_collection"].astype(nft_cat).cat.codes

traders_index = pd.concat([seller_index,buyer_index], ignore_index=True)

seller_dict = dict(seller_index.value_counts())
trader_dict = dict(traders_index.value_counts())
buyer_dict = dict(buyer_index.value_counts())
nft_dict = dict(nft_index.value_counts())


# small_dataset.to_csv('small_dataset.csv')
plt.hist(traders_index, bins = 100, density=True)
plt.title("Trader frequency histogram")
plt.xlabel('Traders (Both buyers and sellers)')
plt.ylabel('Frequency (As either buyer or seller)')
plt.grid(True)
plt.tight_layout()
plt.savefig('traders_histogram.png')

plt.hist(buyer_index, bins = 100, density=False)
plt.title('Buyer frequency histogram')
plt.xlabel('Buyers (Encoded ID)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('buyers_histogram.png')

plt.hist(seller_index, bins = 100, density=False)
plt.title('Seller frequency histogram')
plt.xlabel('Sellers (Encoded ID)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('sellers_histogram.png')

plt.hist(nft_index, bins = 100, density=False)
plt.xlabel('Non-fungible tokens (Encoded ID)')
plt.ylabel('Frequency')
plt.title('NFT frequency histogram')
plt.grid(True)
plt.tight_layout()
plt.ticklabel_format(style='plain')
plt.savefig('nft_histogram.png')


values = list(seller_dict.values())
keys = list(seller_dict.keys())
k50 = keys[:50]
v50 = values[:50]
plt.bar(range(len(k50)), v50, tick_label = k50)
plt.xlabel('Seller IDs')
plt.ylabel('Frequency')
plt.title('Seller frequency bar plot (top 50)')
plt.grid(True)
plt.xticks(rotation = -60, fontsize = 6)
plt.tight_layout()
plt.savefig('seller_barplot.png')
plt.show()
plt.close()

values = list(buyer_dict.values())
keys = list(buyer_dict.keys())
k50 = keys[:50]
v50 = values[:50]
plt.bar(range(len(k50)), v50, tick_label = k50)
plt.xlabel('Buyer IDs')
plt.ylabel('Frequency')
plt.title('Buyer frequency bar plot (top 50)')
plt.grid(True)
plt.xticks(rotation = -60, fontsize = 6)
plt.tight_layout()
plt.savefig('buyer_barplot.png')
plt.show()
plt.close()

values = list(trader_dict.values())
keys = list(trader_dict.keys())
k50 = keys[:50]
v50 = values[:50]
plt.bar(range(len(k50)), v50, tick_label = k50)
plt.xlabel('Trader IDs')
plt.ylabel('Frequency')
plt.title('Trader frequency bar plot (top 50)')
plt.grid(True)
plt.xticks(rotation = -60, fontsize = 6)
plt.tight_layout()
plt.savefig('Trader_barplot.png')
plt.show()
plt.close()


values = list(nft_dict.values())
keys = list(nft_dict.keys())
k50 = keys[:50]
v50 = values[:50]
plt.bar(range(len(k50)), v50, tick_label = k50)
plt.xlabel('NFT IDs')
plt.ylabel('Frequency')
plt.title('NFT frequency bar plot (top 50)')
plt.grid(True)
plt.xticks(rotation = -60, fontsize = 6)
plt.tight_layout()
plt.savefig('NFT_barplot.png')
plt.show()

