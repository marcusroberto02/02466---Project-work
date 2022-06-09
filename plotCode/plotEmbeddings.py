import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

colors = {'Games':'red', 'Art':'green', 'Collectible':'blue', 'Metaverse':'orange','Other':'purple','Utility':'brown'}

class embedPlotter:
    def __init__(self, path):
        self.path = path
        self.load_embeddings()

    def load_embeddings(self):
        pass

    def scatter(self):
        pass


class embedPlotterBi(embedPlotter):
    def __init__(self,path):
        super().__init__(path)

    def load_embeddings(self):
        self.z = torch.load(self.path + "/bi/results/D2/nft_embeddings").detach().numpy()
        self.q = torch.load(self.path + "/bi/results/D2/trader_embeddings").detach().numpy()
    
    def scatter(self):
        plt.scatter(*zip(*self.z[:,:2]),s=0.1,label="NFTs")
        plt.scatter(*zip(*self.q[:,:2]),s=0.1,label="Traders")
        plt.legend(markerscale=15)
        plt.title("Scatter plot - Bipartite model")
        plt.show()

    def categoryPlot(self):
        categories = np.loadtxt(path + "/bi/sparse_c.txt",dtype='str')
        for category, color in colors.items():
            plt.scatter(*zip(*self.z[categories==category][:,:2]),s=0.1,c=color,label=category)
        plt.legend(markerscale=15)
        plt.title("Category plot - Bipartite model")
        plt.show()

class embedPlotterTri(embedPlotter):
    def __init__(self,path):
        super().__init__(path)

    def load_embeddings(self):
        self.l = torch.load(self.path + "/tri/results/D2/nft_embeddings").detach().numpy()
        self.r = torch.load(self.path + "/tri/results/D2/seller_embeddings").detach().numpy()
        self.u = torch.load(self.path + "/tri/results/D2/buyer_embeddings").detach().numpy()
    
    def scatter(self):
        plt.scatter(*zip(*self.l[:,:2]),s=0.1,label="NFTs")
        plt.scatter(*zip(*self.r[:,:2]),s=0.1,label="Sellers")
        plt.scatter(*zip(*self.u[:,:2]),s=0.1,label="Buyers")
        plt.legend(markerscale=15)
        plt.title("Scatter plot - Tripartite model")
        plt.show()
    
    def categoryPlot(self):
        categories = np.loadtxt(path + "/tri/sparse_c.txt",dtype='str')
        for category, color in colors.items():
            plt.scatter(*zip(*self.l[categories==category][:,:2]),s=0.1,c=color,label=category)
        plt.legend(markerscale=15)
        plt.title("Category plot - Tripartite model")
        plt.show()

        
path = "./data/ETH/2020-10"

# embed plot structure bi
epb = embedPlotterBi(path)
epb.scatter()
epb.categoryPlot()

# embed plot structure tri
ept = embedPlotterTri(path)
ept.scatter()
ept.categoryPlot()



#bipartite
'''

# nft
z = z.detach().numpy()
zx = [el[0] for el in z]
zy = [el[1] for el in z]
plt.scatter(zx,zy,s=1,color="blue")
# trader
q = q.detach().numpy()
qx = [el[0] for el in q]
qy = [el[1] for el in q]
plt.scatter(qx,qy,s=1,color="red")
plt.show()

print(len(z))
print(len(q))

nft_z = torch.load("../results/seller_embeddings")

print(nft_z)
'''
"""
l = torch.load('./data/ETH/2020-10/results/tri/nft_embeddings')
r = torch.load('./data/ETH/2020-10/results/tri/seller_embeddings')
u = torch.load('./data/ETH/2020-10/results/tri/buyer_embeddings')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# nft
l = l.detach().numpy()[:1000]
lx = [el[0] for el in l]
ly = [el[1] for el in l]
lz = [el[2] for el in l]
ax.scatter(lx,ly,lz,s=0.1,color="red")

# seller
r = r.detach().numpy()[:1000]
rx = [el[0] for el in r]
ry = [el[1] for el in r]
rz = [el[2] for el in r]
ax.scatter(rx,ry,rz,s=0.1,color="green")

# buyer
u = u.detach().numpy()[:1000]
ux = [el[0] for el in u]
uy = [el[1] for el in u]
uz = [el[2] for el in u]
ax.scatter(ux,uy,uz,s=0.1,color="blue")

#plt.ion()
plt.show()

print(len(l))
print(len(r))
print(len(u))
"""
"""
# nft
l = l.detach().numpy()
lx = [el[0] for el in l]
ly = [el[1] for el in l]
plt.scatter(lx,ly,s=0.1,color="red")

# seller
r = r.detach().numpy()
rx = [el[0] for el in r]
ry = [el[1] for el in r]
plt.scatter(rx,ry,s=0.1,color="green")

# buyer
u = u.detach().numpy()
ux = [el[0] for el in u]
uy = [el[1] for el in u]
plt.scatter(ux,uy,s=0.1,color="blue")

#plt.ion()
#plt.show()
#plt.ioff()
plt.show()
"""
