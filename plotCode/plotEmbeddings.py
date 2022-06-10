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
        self.z = torch.load(self.path + "/bi/D2/nft_embeddings").detach().numpy()
        self.q = torch.load(self.path + "/bi/D2/trader_embeddings").detach().numpy()
    
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
        self.l = torch.load(self.path + "/tri/D2/nft_embeddings").detach().numpy()
        self.r = torch.load(self.path + "/tri/D2/seller_embeddings").detach().numpy()
        self.u = torch.load(self.path + "/tri/D2/buyer_embeddings").detach().numpy()
    
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

        
path = "./results_final/ETH/2021-02"

# embed plot structure bi
epb = embedPlotterBi(path)
epb.scatter()
epb.categoryPlot()

# embed plot structure tri
ept = embedPlotterTri(path)
ept.scatter()
ept.categoryPlot()
