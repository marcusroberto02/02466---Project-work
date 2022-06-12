import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


path = "../results_final/ETH/2021-02"

path_2 = "../data/ETH/2021-02/tri/"

nft_id = np.loadtxt(path_2 + "train/sparse_i.txt")
seller_id = np.loadtxt(path_2 + "train/sparse_j.txt")
buyer_id = np.loadtxt(path_2 + "train/sparse_k.txt")


l2 = torch.load(path + "/tri/results/D2/nft_embeddings").detach().numpy()
r2 = torch.load(path + "/tri/results/D2/seller_embeddings").detach().numpy()
u2 = torch.load(path + "/tri/results/D2/buyer_embeddings").detach().numpy()

sellerbuyertable = pd.read_csv(path_2 + "sellerbuyeridtable.csv")


# choose seller
# seller 30 har fart på
seller = sellerbuyertable['ei_seller'][30]
seller = 30
# create seller to buyer dict
sellertobuyer = dict(zip(sellerbuyertable['ei_seller'],sellerbuyertable['ei_buyer']))

# find buyer id
buyer = sellertobuyer[seller]
# find sold nfts of the trader
sold_nfts = [int(i) for i in nft_id[np.where(seller_id == seller)[0]]]
# find bought nfts of the trader
bought_nfts = [int(i) for i in nft_id[np.where(buyer_id == buyer)[0]]]
print(len(sold_nfts),len(bought_nfts))

# remove bought and sold NFTs to a common list
common = list(set(sold_nfts) & set(bought_nfts))
sold_nfts, bought_nfts = list(set(sold_nfts) - set(bought_nfts)), list(set(bought_nfts) - set(sold_nfts))

s = 10
# plot sold NFTs
plt.scatter(*zip(*l2[sold_nfts]), marker="v", s =s, color = "green", label = "Sold NFTs")
# plot bought NFTs
plt.scatter(*zip(*l2[bought_nfts]), marker="*", s = s, color = "red", label = "Bought NFTs")
# plot traded NFTs
if len(common) > 0:
    plt.scatter(*zip(*l2[common]), marker = "P", s = s, color = "black", label = "Traded NFTs")
# plot the seller
plt.scatter((*r2[seller]), marker = "o",s = 80, color = "blue", label = "Seller")
# plot the buyer
plt.scatter(*u2[buyer], marker="o",s=80, color = "purple", label = "Buyer")
plt.legend()
plt.show()



