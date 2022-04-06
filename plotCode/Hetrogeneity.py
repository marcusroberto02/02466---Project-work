import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl


data = pd.read_csv("../data/toy.csv", index_col = 0)
#data = pd.read_csv("data/Data_API.csv")

seller_count = data.groupby('Seller_address').size().values
buyer_count = data.groupby('Buyer_address').size().values

min_bin,max_bin = min(seller_count),max(seller_count)
bins = np.logspace(np.log10(min_bin),np.log10(max_bin),50)
hist, edges = np.histogram(seller_count,bins = bins, density=True)
x = (edges[1:] + edges[:-1])/2

fig, (ax,ax2) = plt.subplots(2)
ax.plot(x,hist,marker = '.')
ax.set_xlabel('Sellers')
ax.set_ylabel('Probaility density')
ax.set_xscale('log')
ax.set_yscale('log')

min_bin,max_bin = min(buyer_count),max(buyer_count)
bins = np.logspace(np.log10(min_bin),np.log10(max_bin),50)
hist_b, edges_b = np.histogram(seller_count,bins = bins, density=True)
x_b = (edges_b[1:] + edges_b[:-1])/2

ax2.plot(x_b,hist_b,marker = '.')
ax2.set_xlabel('Buyers')
ax2.set_ylabel('Probaility density')
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.show()

#TEST!