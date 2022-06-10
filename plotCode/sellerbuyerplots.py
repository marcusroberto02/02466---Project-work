import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
path = "./data/ETH/2020-10"

specific = "/tri/results/D2"

seller_biases = torch.load(path + specific + "/seller_biases").detach().numpy()
buyer_biases = torch.load(path + specific + "/buyer_biases").detach().numpy()

seller_biases = (seller_biases - np.mean(seller_biases)) / np.std(seller_biases)
buyer_biases = (buyer_biases - np.mean(buyer_biases)) / np.std(buyer_biases)

df = pd.read_csv(path + "/tri/sellerbuyeridtable.csv")
sellers = df["ei_seller"]
buyers = df['ei_buyer']

plt.scatter(seller_biases[sellers],buyer_biases[buyers],s=5)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
plt.show()
"""
# define lower limit for vast inactivity for sellers 
# (i.e. if the seller bias is below this threshold the seller is considered inactive)
sl = -5
# define upper limit for highly active sellers
sh = 5

# define lower limit for vast inactivity for buyers
# (i.e. if the seller bias is below this threshold the seller is considered inactive)
bl = -5
# define upper limit for highly active buyers
bh = 2
# bh = np.where(buyer_biases > bh)
# idx = np.where(seller_biases < -sl)
# return bh[idx]
# collectors is defined as people who buys a lot and doesn't sell that much
def discover_collectors():
    # active buyers
    acb = set(list(np.where(buyer_biases > bh)[0]))
    # inactive sellers
    ins = set(list(np.where(seller_biases < sl)[0]))
    return list(acb & ins)

def discover_miners():
    acb = set(list(np.where(buyer_biases < bl)[0]))
    # inactive sellers
    ins = set(list(np.where(seller_biases > sh)[0]))
    return list(acb & ins)
"""
# load embeddings
l = torch.load(path + specific + "/nft_embeddings").detach().numpy()
r = torch.load(path + specific + "/seller_embeddings").detach().numpy()
u = torch.load(path + specific + "/buyer_embeddings").detach().numpy()
"""
# find collectors
collectors = discover_collectors()

# find collector NFT interaction

nft_ids = np.loadtxt(path+'/tri/train/sparse_i.txt')
seller_ids = np.loadtxt(path+'/tri/train/sparse_j.txt')
buyer_ids = np.loadtxt(path+'/tri/train/sparse_k.txt')

sold_items = {}
bought_items = {}

for collector in collectors:
    indexes_seller = np.where(seller_ids == collector)[0]
    indexes_buyer = np.where(buyer_ids == collector)[0]
    sold_items[collector] = [int(nft) for nft in nft_ids[indexes_seller]]
    bought_items[collector] = [int(nft) for nft in nft_ids[indexes_buyer]]


# plot seller and buyer location for specific collector
collector = collectors[0]
#plt.scatter(r[collector,0],r[collector,1],s=50,color="yellow",label="seller")
#plt.scatter(u[collector,0],u[collector,1],s=50,color="black",label="buyer")

# categories
categories = np.loadtxt(path + "/tri/train/sparse_c.txt",dtype="str")

colors = {'Art':'green', 'Collectible':'blue', 'Games':'red','Metaverse':'orange','Other':'purple','Utility':'brown'}

# sold items by the collector

cls = l[sold_items[collector]]
# bought items by the collector
clb = l[bought_items[collector]]


#plt.scatter(*zip(*cls[:,:2]),s=10,c="blue",marker="P",label="Sold items")
#plt.scatter(*zip(*clb[:,:2]),s=10,c="red",marker="*",label="Bought items")

#plt.legend(loc="lower left", markerscale=15)
#plt.title("True classification - Tripartite model")
#plt.show()

"""

# Seller buyer plots
# embeddings loaded as l = nft, r = sellers, u = buyers
# Scatter plots of the embeddings
inactive_sellers = r[seller_biases <= np.percentile(seller_biases, 20)]
active_sellers = r[seller_biases >= np.percentile(seller_biases, 80)]
inactive_buyers = u[buyer_biases <= np.percentile(buyer_biases, 20)]
active_buyers = u[buyer_biases >= np.percentile(buyer_biases, 80)]

plt.scatter(inactive_sellers.T[0], inactive_sellers.T[1], s=2, c = 'red', marker='o', label="Inactive sellers")
plt.scatter(active_sellers.T[0], active_sellers.T[1], s=2, color = 'green',marker='o', label="Active sellers")
#plt.legend(loc="upper right")
#plt.title("Seller embeddings")
#plt.show()
plt.scatter(inactive_buyers.T[0], inactive_buyers.T[1], s=2, c = 'orange', marker='*', label="Inactive buyers")
plt.scatter(active_buyers.T[0], active_buyers.T[1], s=2, color = 'purple', marker='*', label="Active buyers")
plt.legend(loc="upper right")
plt.title("Seller and buyer embeddings")
plt.show()


def plot_clusters(data):
    kmeans_kwargs = {
        "init" :"random",
        "n_init":10,
        "max_iter":100,
        "random_state":33
    }

    kmeans = KMeans(n_clusters=5, **kmeans_kwargs)
    kmeans.fit(data)

    labels = kmeans.fit_predict(data)
    u_labels = np.unique(labels)

    labelled_sellers = np.c_[data,labels]

    for j in u_labels:
        clust = [i for i in labelled_sellers if i[2]==j]
        plt.scatter(np.transpose(clust)[0], np.transpose(clust)[1], s = 5,label = j)
    plt.legend()
    plt.show()
'''
    silhouette_coefficients = []

    for k in range (2,11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        score = silhouette_score(u, kmeans.labels_)
        silhouette_coefficients.append(score)

        plt.style.use("fivethirtyeight")
        plt.plot(list(range(2, 11)), silhouette_coefficients)
        plt.xticks(range(2, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.show()
'''
plot_clusters(u)
plot_clusters(r)


## Bias histograms 
fig, axs = plt.subplots(2)
axs[1].hist(seller_biases, bins = 100, density=False)
axs[1].set_title('Seller Bias histogram')
axs[1].xlabel('Seller Biases')
axs[1].ylabel('Frequency')
axs[1].grid(True)

axs[2].hist(buyer_biases, bins = 100, density=False)
axs[2].set_title('Buyer Bias histogram')
axs[2].xlabel('Buyer Biases')
axs[2].ylabel('Frequency')
axs[2].grid(True)


plt.tight_layout()


plt.show()