import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl


#data = pd.read_csv("../data/toy.csv", index_col = 0)
data = pd.read_csv("./data/Data_API.csv", low_memory=True)

seller_count = data.groupby('Seller_address').size().values
buyer_count = data.groupby('Buyer_address').size().values
nft_count = data.groupby('Unique_id_collection').size().values

traders = np.append(data['Seller_address'], data['Buyer_address'])
trad_df=pd.DataFrame(traders)
trad_count = trad_df.groupby(0).size().values

counts = [seller_count,buyer_count,]

min_bin,max_bin = min(seller_count),max(seller_count)
bins = np.logspace(np.log10(min_bin),np.log10(max_bin),30)
hist, edges = np.histogram(seller_count,bins = bins, density=False)
x = (edges[1:] + edges[:-1])/2

fig, axs = plt.subplots(2,2)
axs[0,0].plot(x,hist,marker = '.')
axs[0,0].set_xlabel('Sellers')
axs[0,0].set_ylabel('Counts')
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[0,0].grid(True)

min_bin,max_bin = min(buyer_count),max(buyer_count)
bins = np.logspace(np.log10(min_bin),np.log10(max_bin),30)
hist_b, edges_b = np.histogram(seller_count,bins = bins, density=False)
x_b = (edges_b[1:] + edges_b[:-1])/2

axs[0,1].plot(x_b,hist_b,marker = '.')
axs[0,1].set_xlabel('Buyers')
axs[0,1].set_ylabel('Counts')
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')
axs[0,1].grid(True)

min_bin,max_bin = min(nft_count),max(nft_count)
bins = np.logspace(np.log10(min_bin),np.log10(max_bin),24)
hist_n, edges_n = np.histogram(seller_count,bins = bins, density=False)
x_n = (edges_n[1:] + edges_n[:-1])/2

axs[1,1].plot(x_n,hist_n,marker = '.')
axs[1,1].set_xlabel('NFTs')
axs[1,1].set_ylabel('Counts')
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')
axs[1,1].grid(True)

min_bin,max_bin = min(trad_count),max(trad_count)
bins = np.logspace(np.log10(min_bin),np.log10(max_bin),24)
hist_n, edges_n = np.histogram(trad_count,bins = bins, density=False)
x_n = (edges_n[1:] + edges_n[:-1])/2

axs[1,0].plot(x_n,hist_n,marker = '.')
axs[1,0].set_xlabel('Traders')
axs[1,0].set_ylabel('Counts')
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[1,0].grid(True)
plt.tight_layout()
plt.rcParams['figure.figsize'] = [12,8]
plt.show()

