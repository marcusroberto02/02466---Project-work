from plot_formatting import Formatter
from collections import defaultdict, Counter
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

###################
# EMBEDDING PLOTS #
###################

# 2D

class EmbeddingPlotter2D(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.95,0.90) 

    # color for embeddings
    colors = {'Art':'green','Collectible':'blue','Games':'red','Metaverse':'orange','Other':'purple','Utility':'brown'}
    
    # empty embedding variables for bi
    z = None
    q = None

    # empty embedding variables for tri
    l = None
    r = None
    u = None

    # coordinate name
    csuffix = defaultdict(lambda: "th")
    csuffix[1] = "st"
    csuffix[2] = "nd"
    csuffix[3] = "rd"

    # size of each data point in plot
    s_big = 4
    s_small = 1

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/EmbeddingPlots/2D"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)    
    
    def load_embeddings_bi(self):
        self.z = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.q = torch.load("{path}/results/D{dim}/trader_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
    
    def load_embeddings_tri(self):
        self.l = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.r = torch.load("{path}/results/D{dim}/seller_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.u = torch.load("{path}/results/D{dim}/buyer_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()

    def make_scatter_plot_bi(self,d1=1,d2=2):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.z is None or self.q is None:
            self.load_embeddings_bi()
        
        self.fig = plt.figure(figsize=self.figsize)
        plt.scatter(self.z[:,d1-1],self.z[:,d2-1],s=self.s_big,label="NFTs")
        plt.scatter(self.q[:,d1-1],self.q[:,d2-1],s=self.s_big,label="Traders")
        plt.legend(loc="upper right",markerscale=self.markerscale)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Scatter plot of embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

    def make_scatter_plot_tri(self,d1=1,d2=2,d3=3,plot3d=False,n_rot=4):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.l is None or self.r is None or self.u is None:
            self.load_embeddings_tri()

        self.fig = plt.figure(figsize=self.figsize)
        plt.scatter(self.l[:,d1-1],self.l[:,d2-1],s=self.s_big,label="NFTs")
        plt.scatter(self.r[:,d1-1],self.r[:,d2-1],s=self.s_big,label="Sellers")
        plt.scatter(self.u[:,d1-1],self.u[:,d2-1],s=self.s_big,label="Buyers")
        plt.legend(loc="upper right",markerscale=self.markerscale)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Scatter plot of embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

    def make_scatter_plot(self,d1=1,d2=2,save=False,show=False):        
        if self.mtype == "bi":
            self.make_scatter_plot_bi(d1=d1,d2=d2)
        elif self.mtype == "tri":
            self.make_scatter_plot_tri(d1=d1,d2=d2)

        if save:
            plt.savefig("{path}/scatter_plot_2D_{d1}_{d2}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_scatter_plot_all_bi(self,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()
        if self.z is None or self.q is None:
            self.load_embeddings_bi()
        
        self.fig, axes = plt.subplots(nrows=self.dim, ncols=self.dim, sharex=True, sharey=True,figsize=self.figsize)
        
        for d1 in range(1,self.dim+1):
            for d2 in range(1,self.dim+1):
                axes[d2-1,d1-1].scatter(self.z[:,d1-1],self.z[:,d2-1],s=self.s_small,label="NFTs")
                axes[d2-1,d1-1].scatter(self.q[:,d1-1],self.q[:,d2-1],s=self.s_small,label="Traders")
                if d1 == 1:
                    ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
                    axes[d2-1,d1-1].set_ylabel(ylabel)
                if d2 == self.dim:
                    xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
                    axes[d2-1,d1-1].set_xlabel(xlabel)

        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale*2,fontsize=self.fontsize_legend)


        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles(title="Scatter plot of embeddings for all pairs",subtitle=self.dataname,title_y=self.fig_title_y)

    def make_scatter_plot_all_tri(self,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()
        if self.l is None or self.r is None and self.u is None:
            self.load_embeddings_tri()
        
        self.fig, axes = plt.subplots(nrows=self.dim, ncols=self.dim, sharex=True, sharey=True,figsize=self.figsize)
        
        for d1 in range(1,self.dim+1):
            for d2 in range(1,self.dim+1):
                axes[d2-1,d1-1].scatter(self.l[:,d1-1],self.l[:,d2-1],s=self.s_small,label="NFTs")
                axes[d2-1,d1-1].scatter(self.r[:,d1-1],self.r[:,d2-1],s=self.s_small,label="Sellers")
                axes[d2-1,d1-1].scatter(self.u[:,d1-1],self.u[:,d2-1],s=self.s_small,label="Buyers")
                if d1 == 1:
                    ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
                    axes[d2-1,d1-1].set_ylabel(ylabel)
                if d2 == self.dim:
                    xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
                    axes[d2-1,d1-1].set_xlabel(xlabel)

        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale*2,fontsize=self.fontsize_legend)

        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles(title="Scatter plot of embeddings for all pairs",subtitle=self.dataname,title_y=self.fig_title_y)

    def make_scatter_plot_all(self,save=False,show=False):
        if self.mtype == "bi":
            self.make_scatter_plot_all_bi(save=save,show=show)
        elif self.mtype == "tri":
            self.make_scatter_plot_all_tri(save=save,show=show)

        if save:
            plt.savefig("{path}/scatter_plot_all_2D_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_category_nft_plot(self,d1=1,d2=2,save=False,show=False):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.mtype == "bi":
            if self.z is None:
                self.load_embeddings_bi()
        elif self.mtype == "tri":
            if self.l is None:
                self.load_embeddings_tri()

        # load correct nft embedding
        nft_embeddings = self.z if self.mtype == "bi" else self.l

        self.fig = plt.figure(figsize=self.figsize)

        categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
        for category, color in self.colors.items():
            plt.scatter(nft_embeddings[categories==category,d1-1],nft_embeddings[categories==category,d2-1],s=self.s_big,c=color,label=category)
        plt.legend(loc="upper right",markerscale=15)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Category plot for the NFT embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

        if save:
            plt.savefig("{path}/category_plot_nfts_2D_{d1}_{d2}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_category_plot_all(self,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()
        
        if self.mtype == "bi":
            if self.z is None:
                self.load_embeddings_bi()
        elif self.mtype == "tri":
            if self.l is None:
                self.load_embeddings_tri()

        # load correct nft embedding
        nft_embeddings = self.z if self.mtype == "bi" else self.l
        
        self.fig, axes = plt.subplots(nrows=self.dim, ncols=self.dim, sharex=True, sharey=True,figsize=self.figsize)

        categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
        for d1 in range(1,self.dim+1):
            for d2 in range(1,self.dim+1):
                for category, color in self.colors.items():
                    axes[d2-1,d1-1].scatter(nft_embeddings[categories==category,d1-1],nft_embeddings[categories==category,d2-1],s=self.s_small,c=color,label=category)
                if d1 == 1:
                    ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
                    axes[d2-1,d1-1].set_ylabel(ylabel)
                if d2 == self.dim:
                    xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
                    axes[d2-1,d1-1].set_xlabel(xlabel)

        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale*2,fontsize=self.fontsize_legend)
        
        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles(title="Category plot for embeddings for all pairs",subtitle=self.dataname,title_y=self.fig_title_y)

        if save:
            plt.savefig("{path}/category_plot_all_2D_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_collection_plot(self,d1=1,d2=2,category="Collectible",top=10,save=False,show=False):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.mtype == "bi":
            if self.z is None:
                self.load_embeddings_bi()
        elif self.mtype == "tri":
            if self.l is None:
                self.load_embeddings_tri()

        # load correct nft embedding
        nft_embeddings = self.z if self.mtype == "bi" else self.l

        self.fig = plt.figure(figsize=self.figsize)

        categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
        nft_embeddings = nft_embeddings[categories==category]

        # get 10 most abundant collections from collectible
        collections = np.loadtxt("{path}/collections.txt".format(path=self.results_path),dtype='str')
        collections = collections[categories==category]
        cc = Counter(collections)
        unique_collections = np.unique(collections)
        counts = [cc[c] for c in unique_collections]
        top_collections = unique_collections[np.argsort(counts)[-top:]]
        for collection in top_collections:
            plt.scatter(nft_embeddings[collections==collection,d1-1],nft_embeddings[collections==collection,d2-1],s=self.s_big,label=collection)
        plt.legend(loc="upper right",markerscale=15)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title=f"Top {top} collections plot for the NFT embeddings in 2D",subtitle=f"{self.dataname} - {category}",title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

        if save:
            plt.savefig("{path}/{category}_top{top}_collection_plot_2D_{d1}_{d2}_{mtype}_D{dim:d}".format(path=self.store_path,category=category,top=top,d1=d1,d2=d2,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
    def make_category_trader_plot_bi(self,d1=1,d2=2,save=False,show=False):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.q is None:
            self.load_embeddings_bi()
        
        self.fig = plt.figure(figsize=self.figsize)

        # load sparse matrices to represent each trade
        sparse_i = np.loadtxt(self.results_path + "/train/sparse_i.txt",dtype=int)
        sparse_j = np.loadtxt(self.results_path + "/train/sparse_j.txt",dtype=int)

        traders = np.unique(sparse_j)

        trader_embeddings = self.q

        print(len(traders))

        categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
        top_category = []

        for i,t in enumerate(traders):
            if i % 1000 == 0:
                print(i)
            nfts = sparse_i[sparse_j == t]
            cc = Counter([categories[nft] for nft in nfts])
            top_category.append(max(cc))

        top_category = np.array(top_category)

        for category, color in self.colors.items():
            # boolean array that keeps track if the jth trader has traded the most in given category
            plt.scatter(trader_embeddings[top_category==category,d1-1],trader_embeddings[top_category==category,d2-1],s=self.s_big,c=color,label=category)
        plt.legend(loc="upper right",markerscale=15)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Category plot for trader embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

        if save:
            plt.savefig("{path}/category_plot_trader_2D_{d1}_{d2}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_category_trader_plot_tri(self,d1=1,d2=2,save=False,show=False):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.r is None or self.u is None:
            self.load_embeddings_tri()
        
        self.fig = plt.figure(figsize=self.figsize)

        # load sparse matrices to represent each trade
        sparse_i = np.loadtxt(self.results_path + "/train/sparse_i.txt",dtype=int)
        sparse_j = np.loadtxt(self.results_path + "/train/sparse_j.txt",dtype=int)
        sparse_k = np.loadtxt(self.results_path + "/train/sparse_k.txt",dtype=int)

        sellers = np.unique(sparse_j)
        buyers = np.unique(sparse_k)

        seller_embeddings = self.r
        buyer_embeddings = self.u

        categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')

        print(len(sellers))
        
        top_category_seller = []

        for i,s in enumerate(sellers):
            if i % 1000 == 0:
                print(i)
            nfts = sparse_i[sparse_j == s]
            cc = Counter([categories[nft] for nft in nfts])
            top_category_seller.append(max(cc))

        top_category_seller = np.array(top_category_seller)

        print(len(buyers))
        
        top_category_buyer = []

        for i,b in enumerate(buyers):
            if i % 1000 == 0:
                print(i)
            nfts = sparse_i[sparse_k == b]
            cc = Counter([categories[nft] for nft in nfts])
            top_category_buyer.append(max(cc))

        top_category_buyer = np.array(top_category_buyer)

        for category, color in self.colors.items():
            # boolean array that keeps track if the jth trader has traded the most in given category
            plt.scatter(seller_embeddings[top_category_seller==category,d1-1],seller_embeddings[top_category_seller==category,d2-1],s=self.s_big,c=color,label=category)
            plt.scatter(buyer_embeddings[top_category_buyer==category,d1-1],buyer_embeddings[top_category_buyer==category,d2-1],s=self.s_big,c=color,label='_nolegend_')
        plt.legend(loc="upper right",markerscale=15)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Category plot for trader embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

        if save:
            plt.savefig("{path}/category_plot_trader_2D_{d1}_{d2}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def print_average_count_from_origin(self,stepsize=5):
        # print the average number of trades an nft who is
        # at least mindist away from the origin and at most maxdist
        # away from the origin
        if self.mtype == "bi":
            if self.z is None or self.q is None:
                self.load_embeddings_bi()
            # load sparse matrices to represent each trade
            sparse_i = np.loadtxt(self.results_path + "/train/sparse_i.txt",dtype=int)
            sparse_j = np.loadtxt(self.results_path + "/train/sparse_j.txt",dtype=int)
            sparse_w = np.loadtxt(self.results_path + "/train/sparse_w.txt",dtype=int)
            nft_dists = [np.linalg.norm(v) for v in self.z]
            trader_dists = [np.linalg.norm(v) for v in self.q]

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nNFT results: {len(nft_dists)}\n")

            for ni in np.argsort(nft_dists):
                dist = nft_dists[ni]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_i == ni])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_i == ni])
                    nums = 1

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nTrader results: {len(trader_dists)}\n")

            for ti in np.argsort(trader_dists):
                dist = trader_dists[ti]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_j == ti])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_j == ti])
                    nums = 1
            
            

        elif self.mtype == "tri":
            if self.l is None or self.r is None or self.u is None:
                self.load_embeddings_tri()
            
            # load sparse matrices to represent each trade
            sparse_i = np.loadtxt(self.results_path + "/train/sparse_i.txt",dtype=int)
            sparse_j = np.loadtxt(self.results_path + "/train/sparse_j.txt",dtype=int)
            sparse_k = np.loadtxt(self.results_path + "/train/sparse_k.txt",dtype=int)
            sparse_w = np.loadtxt(self.results_path + "/train/sparse_w.txt",dtype=int)
            nft_dists = [np.linalg.norm(v) for v in self.l]
            seller_dists = [np.linalg.norm(v) for v in self.r]
            buyer_dists = [np.linalg.norm(v) for v in self.u]

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nNFT results: {len(nft_dists)}\n")

            for ni in np.argsort(nft_dists):
                dist = nft_dists[ni]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_i == ni])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_i == ni])
                    nums = 1

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nSeller results: {len(seller_dists)}\n")

            for si in np.argsort(seller_dists):
                dist = seller_dists[si]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_j == si])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_j == si])
                    nums = 1

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nBuyer results: {len(buyer_dists)}\n")

            for bi in np.argsort(buyer_dists):
                dist = buyer_dists[bi]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_k == bi])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_k == bi])
                    nums = 1

        


# 3D

class EmbeddingPlotter3D(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.95,0.90) 

    # color for embeddings
    colors = {'Games':'red','Art':'green','Collectible':'blue','Metaverse':'orange','Other':'purple','Utility':'brown'}
    
    # empty embedding variables for bi
    z = None
    q = None

    # empty embedding variables for tri
    l = None
    r = None
    u = None

    # coordinate name
    csuffix = defaultdict(lambda: "th")
    csuffix[1] = "st"
    csuffix[2] = "nd"
    csuffix[3] = "rd"

    # size of each data point in plot
    s_big = 0.05
    s_small = 0.05

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/EmbeddingPlots/3D"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)    
    
    def load_embeddings_bi(self):
        self.z = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.q = torch.load("{path}/results/D{dim}/trader_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
    
    def load_embeddings_tri(self):
        self.l = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.r = torch.load("{path}/results/D{dim}/seller_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.u = torch.load("{path}/results/D{dim}/buyer_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()


    def make_scatter_plot_bi(self,d1=1,d2=2,d3=3,n_rot=3,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()

        if self.dim < 3:
            raise Exception("Cannot plot {dim}D embeddings in 3D".format(dim=self.dim))
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1) and d3 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.z is None or self.q is None:
            self.load_embeddings_bi()
        
        self.fig = plt.figure(figsize=self.figsize)
        
        for i in range(n_rot*n_rot):
            ax = self.fig.add_subplot(n_rot,n_rot,i+1,projection='3d')
            ax.scatter(self.z[:,d1-1],self.z[:,d2-1],self.z[:,d3-1],s=self.s_big,label="NFTs")
            ax.scatter(self.q[:,d1-1],self.q[:,d2-1],self.q[:,d3-1],s=self.s_big,label="Traders")
            #ax.set_title("Rotation: " + str(i*360/(n_rot*n_rot)),weight="bold")
            ax.view_init(azim=i*360/(n_rot*n_rot))
            xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
            ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
            zlabel = "{c3} coordinate".format(c3=str(d3)+self.csuffix[d3])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)

    def make_scatter_plot_tri(self,d1=1,d2=2,d3=3,n_rot=3,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()

        if self.dim < 3:
            raise Exception("Cannot plot {dim}D embeddings in 3D".format(dim=self.dim))
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1) and d3 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.l is None or self.r is None or self.u is None:
            self.load_embeddings_tri()
        
        self.fig = plt.figure(figsize=self.figsize)
        
        for i in range(n_rot*n_rot):
            ax = self.fig.add_subplot(n_rot,n_rot,i+1,projection='3d')
            ax.scatter(self.l[:,d1-1],self.l[:,d2-1],self.l[:,d3-1],s=self.s_big,label="NFTs")
            ax.scatter(self.r[:,d1-1],self.r[:,d2-1],self.r[:,d3-1],s=self.s_big,label="Sellers")
            ax.scatter(self.u[:,d1-1],self.u[:,d2-1],self.u[:,d3-1],s=self.s_big,label="Buyers")
            #ax.set_title("Rotation: " + str(i*360/(n_rot*n_rot)),weight="bold")
            ax.view_init(azim=i*360/(n_rot*n_rot))
            xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
            ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
            zlabel = "{c3} coordinate".format(c3=str(d3)+self.csuffix[d3])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        
    def make_scatter_plot(self,d1=1,d2=2,d3=3,n_rot=3,save=False,show=False):        
        if self.mtype == "bi":
            self.make_scatter_plot_bi(d1=d1,d2=d2,d3=d3,n_rot=n_rot)
        elif self.mtype == "tri":
            self.make_scatter_plot_tri(d1=d1,d2=d2,d3=d3,n_rot=n_rot)

        # prepare legend
        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale,borderpad=2,fontsize=self.fontsize_legend)

        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles_3D(title="Scatter plot of embeddings in 3D",subtitle=self.dataname,title_y=self.fig_title_y)

        if save:
            plt.savefig("{path}/scatter_plot_3D_{d1}_{d2}_{d3}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,d3=d3,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
    def make_category_plot(self,d1=1,d2=2,d3=3,n_rot=3,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()

        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1) and d3 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.mtype == "bi":
            if self.z is None:
                self.load_embeddings_bi()
        elif self.mtype == "tri":
            if self.l is None:
                self.load_embeddings_tri()

        # load correct nft embedding
        nft_embeddings = self.z if self.mtype == "bi" else self.l

        self.fig = plt.figure(figsize=self.figsize)

        for i in range(n_rot*n_rot):
            ax = self.fig.add_subplot(n_rot,n_rot,i+1,projection='3d')
            categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
            for category, color in self.colors.items():
                ax.scatter(nft_embeddings[categories==category,d1-1],nft_embeddings[categories==category,d2-1],nft_embeddings[categories==category,d3-1],s=self.s_big,c=color,label=category)
            #ax.set_title("Rotation: " + str(i*360/(n_rot*n_rot)),weight="bold")
            ax.view_init(azim=i*360/(n_rot*n_rot))
            xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
            ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
            zlabel = "{c3} coordinate".format(c3=str(d3)+self.csuffix[d3])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)

        # get legend
        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale,fontsize=self.fontsize_legend)

        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles_3D(title="Category plot for the NFT embeddings in 3D",subtitle=self.dataname,title_y=self.fig_title_y)
        
        if save:
            plt.savefig("{path}/category_plot_nfts_3D_{d1}_{d2}_{d3}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,d3=d3,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
    def print_average_count_from_origin(self,stepsize=5):
        # print the average number of trades an nft who is
        # at least mindist away from the origin and at most maxdist
        # away from the origin
        if self.mtype == "bi":
            if self.z is None or self.q is None:
                self.load_embeddings_bi()
            # load sparse matrices to represent each trade
            sparse_i = np.loadtxt(self.results_path + "/train/sparse_i.txt",dtype=int)
            sparse_j = np.loadtxt(self.results_path + "/train/sparse_j.txt",dtype=int)
            sparse_w = np.loadtxt(self.results_path + "/train/sparse_w.txt",dtype=int)
            nft_dists = [np.linalg.norm(v) for v in self.z]
            trader_dists = [np.linalg.norm(v) for v in self.q]

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nNFT results: {len(nft_dists)}\n")

            for ni in np.argsort(nft_dists):
                dist = nft_dists[ni]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_i == ni])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_i == ni])
                    nums = 1

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nTrader results: {len(trader_dists)}\n")

            for ti in np.argsort(trader_dists):
                dist = trader_dists[ti]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_j == ti])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_j == ti])
                    nums = 1
            
            

        elif self.mtype == "tri":
            if self.l is None or self.r is None or self.u is None:
                self.load_embeddings_tri()
            
            # load sparse matrices to represent each trade
            sparse_i = np.loadtxt(self.results_path + "/train/sparse_i.txt",dtype=int)
            sparse_j = np.loadtxt(self.results_path + "/train/sparse_j.txt",dtype=int)
            sparse_k = np.loadtxt(self.results_path + "/train/sparse_k.txt",dtype=int)
            sparse_w = np.loadtxt(self.results_path + "/train/sparse_w.txt",dtype=int)
            nft_dists = [np.linalg.norm(v) for v in self.l]
            seller_dists = [np.linalg.norm(v) for v in self.r]
            buyer_dists = [np.linalg.norm(v) for v in self.u]

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nNFT results: {len(nft_dists)}\n")

            for ni in np.argsort(nft_dists):
                dist = nft_dists[ni]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_i == ni])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_i == ni])
                    nums = 1

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nSeller results: {len(seller_dists)}\n")

            for si in np.argsort(seller_dists):
                dist = seller_dists[si]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_j == si])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_j == si])
                    nums = 1

            count = 0
            nums = 0
        
            mindist = 0
            
            print(f"\nBuyer results: {len(buyer_dists)}\n")

            for bi in np.argsort(buyer_dists):
                dist = buyer_dists[bi]
                if dist >= mindist and dist <= mindist + stepsize:
                    count += sum(sparse_w[sparse_k == bi])
                    nums += 1
                else:
                    if nums != 0:
                        print(f"\n{mindist}-{mindist+stepsize}: {count / nums} and {nums}\n")
                    while dist > mindist + stepsize:
                        mindist += stepsize
                    count = sum(sparse_w[sparse_k == bi])
                    nums = 1




# choose data set to investigate
blockchain="ETH"
month="2021-03"
mtypes=["bi","tri"]
dims=[3]

categories = ["Art","Collectible","Games","Metaverse","Other","Utility"]

#for mtype in mtypes:
#    for dim in dims:
#        ep = EmbeddingPlotter2D(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        #for category in categories:
        #    ep.make_collection_plot(category=category,top=5,save=True)
#        ep.print_average_count_from_origin(stepsize=5) 
        #if mtype == "bi":
        #   ep.make_category_trader_plot_bi(save=True)
        #elif mtype == "tri":
        #    ep.make_category_trader_plot_tri(save=True)
#       ep.make_scatter_plot(save=True)
        #ep.make_category_nft_plot(save=True)
#       ep.make_scatter_plot_all(save=True)
#       ep.make_category_plot_all(save=True)


for mtype in mtypes:
    for dim in dims:
        ep = EmbeddingPlotter3D(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        #ep.make_scatter_plot(save=True)
        #ep.make_category_plot(save=True)
        ep.print_average_count_from_origin()

