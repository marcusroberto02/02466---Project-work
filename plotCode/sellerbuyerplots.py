import matplotlib.pyplot as plt
import torch
import numpy as np

path = "./data/ETH/2020-10"

specific = "/tri/results/D2"

seller_biases = torch.load(path + specific + "/seller_biases").detach().numpy()
buyer_biases = torch.load(path + specific + "/buyer_biases").detach().numpy()

n_sellers = len(seller_biases)
n_buyers = len(buyer_biases)

if n_sellers > n_buyers:
    seller_biases = seller_biases[:n_buyers]
else:
    buyer_biases = buyer_biases[:n_sellers]

print(min(seller_biases),max(seller_biases))
print(min(buyer_biases),max(buyer_biases))

# define lower limit for vast inactivity for sellers 
# (i.e. if the seller bias is below this threshold the seller is considered inactive)
sl = -5
# define upper limit for highly active sellers
sh = 10

# define lower limit for vast inactivity for buyers
# (i.e. if the seller bias is below this threshold the seller is considered inactive)
bl = -8
# define upper limit for highly active buyers
bh = 5
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

# load embeddings
l = torch.load(path + specific + "/nft_embeddings").detach().numpy()
r = torch.load(path + specific + "/seller_embeddings").detach().numpy()
u = torch.load(path + specific + "/buyer_embeddings").detach().numpy()

# categories
categories = np.loadtxt(path + "/tri/train/sparse_c.txt",dtype="str")

colors = {'Art':'green', 'Collectible':'blue', 'Games':'red','Metaverse':'orange','Other':'purple','Utility':'brown'}

for category, color in colors.items():
    plt.scatter(*zip(*l[categories==category][:,:2]),s=0.1,c=color,label=category)

# find collectors
collectors = discover_collectors()

"""
for c in collectors:
    print("Trader: " + str(c))
    print("Seller bias: " + str(seller_biases[c]))
    print("Buyer bias: " + str(buyer_biases[c]))
"""

# plot seller and buyer location for specific collector
collector = collectors[0]
plt.scatter(r[collector,0],r[collector,1],s=50,color="pink")
plt.scatter(u[collector,0],u[collector,1],s=50,color="cyan")

plt.legend(loc="lower left", markerscale=15)
plt.title("True classification - Tripartite model")
plt.show()