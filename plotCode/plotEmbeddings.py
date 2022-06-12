import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


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
        # 2d embeddings
        self.z2 = torch.load(self.path + "/bi/results/D2/nft_embeddings").detach().numpy()
        self.q2 = torch.load(self.path + "/bi/results/D2/trader_embeddings").detach().numpy()

        # 3d embeddings
        self.z3 = torch.load(self.path + "/bi/results/D3/nft_embeddings").detach().numpy()
        self.q3 = torch.load(self.path + "/bi/results/D3/trader_embeddings").detach().numpy()
    
    def scatter(self):
        plt.scatter(*zip(*self.z2),s=0.1,label="NFTs")
        plt.scatter(*zip(*self.q2),s=0.1,label="Traders")
        plt.legend(markerscale=15)
        plt.title("Scatter plot 2D - Bipartite model")
        plt.show()
    
    def scatter3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*zip(*self.z3),s=0.1,label="NFTs")
        ax.scatter(*zip(*self.q3),s=0.1,label="Traders")
        ax.legend(markerscale=15)
        ax.set_title("Scatter plot 3D - Bipartite model")
        plt.show()

    def categoryPlot(self):
        categories = np.loadtxt(path + "/bi/sparse_c.txt",dtype='str')
        for category, color in colors.items():
            plt.scatter(*zip(*self.z2[categories==category]),s=0.1,c=color,label=category)
        plt.legend(markerscale=15)
        plt.title("Category plot 2D - Bipartite model")
        plt.show()
    
    def categoryPlot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        categories = np.loadtxt(path + "/bi/sparse_c.txt",dtype='str')
        for category, color in colors.items():
            ax.scatter(*zip(*self.z3[categories==category]),s=0.1,c=color,label=category)
        ax.legend(markerscale=15)
        ax.set_title("Category plot 3D - Bipartite model")
        plt.show()

class embedPlotterTri(embedPlotter):
    def __init__(self,path):
        super().__init__(path)

    def load_embeddings(self):
        # 2d embeddings
        self.l2 = torch.load(self.path + "/tri/results/D2/nft_embeddings").detach().numpy()
        self.r2 = torch.load(self.path + "/tri/results/D2/seller_embeddings").detach().numpy()
        self.u2 = torch.load(self.path + "/tri/results/D2/buyer_embeddings").detach().numpy()

        # 3d embeddings
        self.l3 = torch.load(self.path + "/tri/results/D3/nft_embeddings").detach().numpy()
        self.r3 = torch.load(self.path + "/tri/results/D3/seller_embeddings").detach().numpy()
        self.u3 = torch.load(self.path + "/tri/results/D3/buyer_embeddings").detach().numpy()
    
    def scatter(self):
        plt.scatter(*zip(*self.l2),s=0.1,label="NFTs")
        plt.scatter(*zip(*self.r2),s=0.1,label="Sellers")
        plt.scatter(*zip(*self.u2),s=0.1,label="Buyers")
        plt.legend(markerscale=15)
        plt.title("Scatter plot 2D - Tripartite model")
        plt.show()

    def scatter3D(self,n_rot):
        fig = plt.figure(figsize=(16,16),dpi=60)
        for i in range(n_rot*n_rot):
            ax= fig.add_subplot(n_rot,n_rot,i+1,projection='3d')
            ax.scatter(*zip(*self.l3),s=0.1,label="NFTs")
            ax.scatter(*zip(*self.r3),s=0.1,label="Sellers")
            ax.scatter(*zip(*self.u3),s=0.1,label="Buyers")
            ax.set_title("Rotation: " + str(i*360/(n_rot*n_rot)),y=-0.01)
            ax.view_init(azim=i*360/(n_rot*n_rot))

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.suptitle("Scatter plot 3D - Tripartite model",fontsize=30,weight="bold")
        fig.legend(lines, labels, loc = 'upper right',markerscale=30,borderpad=2,fontsize=20)
        plt.show()
    
    def categoryPlot(self):
        categories = np.loadtxt(path + "/tri/sparse_c.txt",dtype='str')
        for category, color in colors.items():
            plt.scatter(*zip(*self.l2[categories==category]),s=0.1,c=color,label=category)
        plt.legend(markerscale=15)
        plt.title("Category plot 2D - Tripartite model")
        plt.show()

    def categoryPlot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        categories = np.loadtxt(path + "/tri/sparse_c.txt",dtype='str')
        for category, color in colors.items():
            ax.scatter(*zip(*self.l3[categories==category]),s=0.1,c=color,label=category)
        ax.legend(markerscale=15)
        ax.set_title("Category plot 3D - Tripartite model")
        plt.show()

        
path = "../results_final/ETH/2021-02"

# embed plot structure bi
epb = embedPlotterBi(path)
#epb.scatter()
#epb.scatter3D()
#epb.categoryPlot()
#epb.categoryPlot3D()

# embed plot structure tri
ept = embedPlotterTri(path)
#ept.scatter()
ept.scatter3D(4)
#ept.categoryPlot()
#ept.categoryPlot3D()
