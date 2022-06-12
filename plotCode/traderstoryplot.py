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

# remove NFTs that have both been bought and sold to a common list
common = list(set(sold_nfts) & set(bought_nfts))
sold_nfts, bought_nfts = list(set(sold_nfts) - set(bought_nfts)), list(set(bought_nfts) - set(sold_nfts))

s = 10
def plot_2D_story():
    l = torch.load(path + "/tri/results/D2/nft_embeddings").detach().numpy()
    r = torch.load(path + "/tri/results/D2/seller_embeddings").detach().numpy()
    u = torch.load(path + "/tri/results/D2/buyer_embeddings").detach().numpy()
    # plot sold NFTs
    plt.scatter(*zip(*l[sold_nfts]), marker="v", s =s, color = "green", label = "Sold NFTs")
    # plot bought NFTs
    plt.scatter(*zip(*l[bought_nfts]), marker="*", s = s, color = "red", label = "Bought NFTs")
    # plot traded NFTs
    if len(common) > 0:
        plt.scatter(*zip(*l[common]), marker = "P", s = s, color = "black", label = "Traded NFTs")
    # plot the seller
    plt.scatter((*r[seller]), marker = "o",s = 80, color = "blue", label = "Seller")
    # plot the buyer
    plt.scatter(*u[buyer], marker="o",s=80, color = "purple", label = "Buyer")
    plt.legend()
    plt.show()

def plot_3D_story():
    l = torch.load(path + "/tri/results/D3/nft_embeddings").detach().numpy()
    r = torch.load(path + "/tri/results/D3/seller_embeddings").detach().numpy()
    u = torch.load(path + "/tri/results/D3/buyer_embeddings").detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plot sold NFTs
    ax.scatter(*zip(*l[sold_nfts]), marker="v", s =s, color = "green", label = "Sold NFTs")
    # plot bought NFTs
    ax.scatter(*zip(*l[bought_nfts]), marker="*", s = s, color = "red", label = "Bought NFTs")
    # plot traded NFTs
    if len(common) > 0:
        ax.scatter(*zip(*l[common]), marker = "P", s = s, color = "black", label = "Traded NFTs")
    # plot the seller
    ax.scatter((*r[seller]), marker = "o",s = 80, color = "blue", label = "Seller")
    # plot the buyer
    ax.scatter(*u[buyer], marker="o",s=80, color = "purple", label = "Buyer")
    ax.legend()
    plt.show()
plot_2D_story()
plot_3D_story()

