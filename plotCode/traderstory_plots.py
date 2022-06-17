from logging import raiseExceptions
from torch_sparse import remove_diag
from plot_formatting import Formatter
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.stats  import norm, lognorm, poisson

###################
# TRADER STORIES  #
###################

class TraderStoryPlotter(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.96,0.91)

    fig_title_y_lower = (0.96,0.89)

    # standard linewidth
    linewidth = 5

    # empty variables for bias terms
    seller_biases = None
    buyer_biases = None

    # empty variables for embeddings
    l = None
    r = None
    u = None

    # size of points in the trader story plots
    s_big = 1000
    s_small = 40

    # coordinate name
    csuffix = defaultdict(lambda: "th")
    csuffix[1] = "st"
    csuffix[2] = "nd"
    csuffix[3] = "rd"

    # legend symbolsize
    legend_symbolsize = 200

    def __init__(self,blockchain="ETH",month="2021-02",mtype="tri",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/TraderStories"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

    
    def load_biases(self):
        self.seller_biases = torch.load("{path}/results/D{dim}/seller_biases".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.buyer_biases = torch.load("{path}/results/D{dim}/buyer_biases".format(path=self.results_path,dim=self.dim)).detach().numpy()

    def load_embeddings(self):
        self.l = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.r = torch.load("{path}/results/D{dim}/seller_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.u = torch.load("{path}/results/D{dim}/buyer_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()

    # the following function is heavily inspired by the matplotlib documentation
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    def scatter_hist(self,x, y, ax, ax_histx, ax_histy,ax_col=None,remove_origin=None, colours=None):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # Set bottom and left spines as x and y axes of coordinate system
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # the scatter plot:
        #ax.scatter(x, y, c=colours, cmap="viridis")
        image = ax.scatter(x, y, c=colours, cmap="plasma")

        # OBS 
        # HER SKAL MAN MANUELT UNDERSØGE HVILKE INDEKSER DER FJERNER 0'erne
        if remove_origin is not None:
            idx,idy = remove_origin
            xticks = ax.xaxis.get_ticklabels()
            xticks[idx].set_visible(False)
            yticks = ax.yaxis.get_ticklabels()
            yticks[idy].set_visible(False)    

        # Create 'x' and 'y' labels placed at the end of the axes
        ax.set_xlabel(r'$\nu$', size=28, labelpad=-24)
        ax.set_ylabel(r'$\tau$', size=28, labelpad=-21,rotation=0)

        ax.xaxis.set_label_coords(1.03, 0.52)
        ax.yaxis.set_label_coords(0.495, 1.01)

        # Draw arrows
        arrow_fmt = dict(markersize=4, color='black', clip_on=False)
        ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
        ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins, density=True)
        mu, std = norm.fit(x)
        xmin, xmax = plt.xlim()
        xt = np.linspace(xmin, xmax, 100)
        p = norm.pdf(xt, mu, std)
        ax_histx.plot(xt, p, 'k', linewidth=self.linewidth)
        ax_histx.set_yticks([0,0.1,0.2,0.3],[0,0.1,0.2,0.3])

        ax_histy.hist(y, bins=bins, density=True,orientation='horizontal')
        mu, std = norm.fit(y)
        xmin, xmax = plt.xlim()
        yt = np.linspace(xmin, xmax, 100)
        p = norm.pdf(yt, mu, std)
        ax_histy.plot(p, yt, 'k', linewidth=self.linewidth)
        ax_histy.set_xticks([0,0.1,0.2,0.3],[0,0.1,0.2,0.3])
        


        ax_col.grid(False)
        ax_col.axis("off")
        
        clb = plt.colorbar(image,ax=ax_col)
        ax_col.text(x=0,y=-12,s='Distance')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    def make_distances_biases_plot(self, save=False, show=False, normalize=False, remove_origin=None):
        if self.seller_biases is None or self.buyer_biases is None:
            self.load_biases()

        self.fig = plt.figure(figsize=self.figsize)

        
        gs = self.fig.add_gridspec(4, 4, width_ratios=(5,1,1,2), height_ratios=(1,2,1,5),
                                   left=0.1, right=0.9, bottom=0.1, top=0.9,
                                   wspace=0.05, hspace=0.05)

        ax = self.fig.add_subplot(gs[3, 0])
        ax.set_xlabel("Buyer biases")
        ax.set_ylabel("Seller biases")
        ax_histx = self.fig.add_subplot(gs[1, 0], sharex=ax)
        ax_histy = self.fig.add_subplot(gs[3, 3], sharey=ax)

        ax_histy.set_xlabel("Probability", weight="bold", rotation=0, rotation_mode='default', y=0.4, x=-1.5)
        ax_histy.xaxis.set_label_coords(0.5, -.1)
        ax_histy.set_ylabel("Buyer bias value", weight="bold", rotation=-90, rotation_mode='default', y=0.4, x=-1.5)
        ax_histy.yaxis.set_label_coords(-.1, 0.3)
        ax_histx.set_xlabel("Seller bias value", weight="bold")
        ax_histx.set_ylabel("Probability", weight="bold", rotation=-270, rotation_mode='default', y=0.4, x=-1.5)
        ax_histx.yaxis.set_label_coords(-.15, 0.2)

        ax_col = self.fig.add_subplot(gs[3, 1], sharex=ax, sharey=ax)

        if self.r is None or self.u is None:
            self.load_embeddings()

        # only include traders that act as both sellers and buyers
        df = pd.read_csv(self.results_path + "/sellerbuyeridtable.csv")
        seller_ids = df["ei_seller"]
        buyer_ids = df['ei_buyer']

        # get embeddings
        se = self.r[seller_ids]
        be = self.u[buyer_ids]


        # get euclidean distances
        distances = [np.linalg.norm(s-b) for (s,b) in zip(se,be)]
        distances=np.array(distances)

        #sorted distances
        sequence = np.argsort(distances)
        distances = distances[sequence]
    
        # make the scatterplot
        sb = self.seller_biases[seller_ids]
        bb = self.buyer_biases[buyer_ids]

        # sort
        sb = sb[sequence]
        bb = bb[sequence]

        if normalize:
            sb = (sb - np.mean(sb)) / np.std(sb)
            bb = (bb - np.mean(bb)) / np.std(bb)


        #self.fig = plt.figure(figsize=self.figsize)
        self.scatter_hist(sb, bb, ax, ax_histx, ax_histy, ax_col, remove_origin=remove_origin, colours=distances)
        title = "Bias and distance distribution"
        #title += " - Normalized" if normalize else ""
        self.set_titles_3D(title=title, subtitle=self.dataname, title_y=self.fig_title_y)

        if save:
            plt.savefig(
                "{path}/distance_bias_scatter_{mtype}_D{dim:d}".format(path=self.store_path, mtype=self.mtype,
                                                                            dim=self.dim))
        if show:
            plt.show()

    def make_only_sellers_bias_distribution_plot(self,save=False,show=False):
        # plot distribution of distances between between seller and buyer positions
        self.fig = plt.figure(figsize=self.figsize)

        if self.seller_biases is None:
            self.load_biases()

        # only include sellers that never buy
        seller_ids = np.loadtxt(self.results_path + "/onlysellers.txt",dtype=int)

        # get biases
        sb = self.seller_biases[seller_ids]

        plt.hist(sb,bins=200,density=True)
        mu, std = norm.fit(sb)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=self.linewidth)

        self.format_plot(title="Distribution of seller biases for non-buyers", subtitle=self.dataname,
                         title_y=self.fig_title_y, xlabel="Seller bias value", ylabel="Probability")
        
        if save:
            plt.savefig("{path}/only_sellers_bias_distribution_plot_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_only_buyers_bias_distribution_plot(self,save=False,show=False):
        # plot distribution of distances between between seller and buyer positions
        self.fig = plt.figure(figsize=self.figsize)

        if self.buyer_biases is None:
            self.load_biases()

        # only include buyers that never sell
        buyer_ids = np.loadtxt(self.results_path + "/onlybuyers.txt",dtype=int)

        # get biases
        bb = self.buyer_biases[buyer_ids]

        plt.hist(bb,bins=200,density=True)

        #PDF
        mu, std = norm.fit(bb)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth = self.linewidth)

        self.format_plot(title="Distribution of buyer biases for non-sellers", subtitle=self.dataname,
                         title_y=self.fig_title_y, xlabel="Buyer bias value", ylabel="Probability")
        
        if save:
            plt.savefig("{path}/only_buyers_bias_distribution_plot_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_distance_distribution_plot(self,log=False,save=False,show=False):
        # plot distribution of distances between between seller and buyer positions
        self.fig = plt.figure(figsize=self.figsize)

        if self.r is None or self.u is None:
            self.load_embeddings()

        # only include traders that act as both sellers and buyers
        df = pd.read_csv(self.results_path + "/sellerbuyeridtable.csv")
        seller_ids = df["ei_seller"]
        buyer_ids = df['ei_buyer']

        # get embeddings
        se = self.r[seller_ids]
        be = self.u[buyer_ids]

        # get euclidean distances
        distances = [np.linalg.norm(s-b) for (s,b) in zip(se,be)]

        # PDF
        if log:
            distances = np.log(distances)
        
        plt.hist(distances, bins=200, density=True)
        shape, loc, scale = lognorm.fit(distances)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = lognorm.pdf(x, shape, loc, scale)
        plt.plot(x, p, 'k', linewidth=self.linewidth)

        title = "Distribution of distances between seller and buyer positions" if not log else "Distribution of log distances between seller and buyer positions"
        xlabel = "Distance" if not log else "Log distance"

        self.format_plot(title=title, subtitle=self.dataname,
                         title_y=self.fig_title_y, xlabel=xlabel, ylabel="Probability")
        

        if save:
            if log:
                plt.savefig("{path}/log_distance_distribution_plot_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
            else:
                plt.savefig("{path}/distance_distribution_plot_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
    def make_trader_story_plot_2D(self,d1=1,d2=2,story="random",min_sales=0,min_purchases=0,save=False,show=False):
        if self.l is None or self.r is None or self.u is None:
            self.load_embeddings()
        
        # only include traders that act as both sellers and buyers
        df = pd.read_csv(self.results_path + "/sellerbuyeridtable.csv")
        seller_ids = df["ei_seller"]
        buyer_ids = df['ei_buyer']

        # load sparse matrices to represent each trade
        sparse_i = np.loadtxt(self.results_path + "/train/sparse_i.txt",dtype=int)
        sparse_j = np.loadtxt(self.results_path + "/train/sparse_j.txt",dtype=int)
        sparse_k = np.loadtxt(self.results_path + "/train/sparse_k.txt",dtype=int)

        # load categories
        categories = np.loadtxt("./results_final/ETH/2021-03/tri/sparse_c.txt",dtype=str)

        # load collections
        collections = np.loadtxt("./results_final/ETH/2021-03/tri/collections.txt",dtype=str)

        # pick trader based on story
        idx_seller = 0
        idx_buyer = 0
        if story == "random":
            title = "Trader story plot of a random trader"
            idx = np.random.choice(range(len(seller_ids)))
            idx_seller = seller_ids[idx]
            idx_buyer = buyer_ids[idx]
        elif story == "most_frequent_seller":
            title = "Trader story plot of the most frequent seller"
            cj = Counter(sparse_j)
            sales = [cj[si] for si in seller_ids]
            idx_seller = seller_ids[np.argmax(sales)]
            idx_buyer = buyer_ids[np.argmax(sales)]
        elif story == "most_frequent_buyer":
            title = "Trader story plot of the most frequent buyer"
            ck = Counter(sparse_k)
            purchases = [ck[bi] for bi in buyer_ids]
            idx_seller = seller_ids[np.argmax(purchases)]
            idx_buyer = buyer_ids[np.argmax(purchases)]
        elif story == "most_active_trader":
            title = "Trader story plot of the most active trader"
            cj = Counter(sparse_j)
            ck = Counter(sparse_k)
            trades = [cj[si] for si in seller_ids] + [ck[bi] for bi in buyer_ids]
            idx_seller = seller_ids[np.argmax(trades)]
            idx_buyer = buyer_ids[np.argmax(trades)]
        elif story == "most_versatile_seller":
            title = "Trader story plot of the most versatile seller"
            ci = -1
            cmax = 0

            for j,sj in enumerate(seller_ids):
                sold_nfts = np.unique(sparse_i[sparse_j==sj])
                dc = len(np.unique([collections[nft] for nft in sold_nfts]))
                if dc > cmax:
                    ci = j
                    cmax = dc
            idx_seller = seller_ids[ci]
            idx_buyer = buyer_ids[ci]
        elif story == "most_versatile_buyer":
            title = "Trader story plot of the most versatile buyer"
            ci = -1
            cmax = 0

            for k,sk in enumerate(buyer_ids):
                bought_nfts = np.unique(sparse_i[sparse_k==sk])
                dc = len(np.unique([collections[nft] for nft in bought_nfts]))
                if dc > cmax:
                    ci = k
                    cmax = dc
            idx_seller = seller_ids[ci]
            idx_buyer = buyer_ids[ci]
        elif story == "most_distant_trader":
            title = "Trader story plot of the most distant trader"
            # get embeddings
            se = self.r[seller_ids]
            be = self.u[buyer_ids]

            # get euclidean distances
            distances = [np.linalg.norm(s-b) for (s,b) in zip(se,be)]

            idx = -1
            dmax = 0
            min_sales = 5
            min_purchases = 5
            for i, dist in enumerate(distances):
                sn = len(np.unique(sparse_i[sparse_j==seller_ids[i]]))
                bn = len(np.unique(sparse_i[sparse_k==buyer_ids[i]]))
                if dist > dmax and sn >= min_sales and bn >= min_purchases:
                    dmax = dist
                    idx = i
            idx_seller = seller_ids[idx]
            idx_buyer = buyer_ids[idx]
        elif story == "custom_trader":
            title = "Trader story plot of specific trader"
            cj = Counter(sparse_j)
            ck = Counter(sparse_k)
            trader_found = False
            for (si,bi) in zip(seller_ids,buyer_ids):
                if (cj[si] >= min_sales and ck[bi] >= min_purchases):
                    idx_seller = si
                    idx_buyer = bi
                    trader_found = True
                    break
            if not trader_found:
                raise Exception("No trader satifies the given criteria!")
        else:
            raise Exception("Invalid story type!")

        # get information about the chosen trader
        sold_nfts = sparse_i[sparse_j==idx_seller]
        bought_nfts = sparse_i[sparse_k==idx_buyer]

        # initialize figure
        self.fig = plt.figure(figsize=self.figsize)

        # plot sold NFTs
        plt.scatter(self.l[sold_nfts][:,d1-1],self.l[sold_nfts][:,d2-1], marker="v", s =self.s_small, color = "green", label = "Sold NFTs")
        # plot bought NFTs
        plt.scatter(self.l[bought_nfts][:,d1-1],self.l[bought_nfts][:,d2-1], s = self.s_small, color = "red", label = "Bought NFTs")

        # plot the seller
        plt.scatter(self.r[idx_seller,d1-1],self.r[idx_seller,d2-1], marker = "X",edgecolor="black",s = self.s_big, color = "purple", label = "Seller location")
        # plot the buyer
        plt.scatter(self.u[idx_buyer,d1-1],self.u[idx_buyer,d2-1], marker="X",edgecolor="black",s=self.s_big, color = "blue", label = "Buyer location")
        legend = plt.legend(loc="upper right")
        for handle in legend.legendHandles:
            handle.set_sizes([self.legend_symbolsize])

        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])

        # get subtitle
        subtitle = "Sold items: {nsold}, Bought items: {nbought}\n{dataname}".format(nsold=len(sold_nfts),nbought=len(bought_nfts),dataname=self.dataname)

        self.format_plot(title=title,subtitle=subtitle,title_y=self.fig_title_y_lower,xlabel=xlabel,ylabel=ylabel)

        if save:
            plt.savefig("{path}/{story}_story_plot_{mtype}_D{dim:d}".format(path=self.store_path,story=story,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()




# choose data set to investigate
blockchain="ETH"
month="2021-03"
mtype="tri"
dims=[2]

for dim in dims:
    tsp = TraderStoryPlotter(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
    #tsp.degree_heterogeneity_plot(show=True)
    #tsp.make_bias_distribution_plot(remove_origin=(3,3),save=True, show =True)
    #tsp.make_bias_distribution_plot(normalize=True,remove_origin=(3,3),save=True)
    #tsp.make_only_sellers_bias_distribution_plot(save=True)
    #tsp.make_only_buyers_bias_distribution_plot(save=True)
    #tsp.make_distance_distribution_plot(save=True)
    #tsp.make_distance_distribution_plot(log=True,save=True)
    #tsp.make_trader_story_plot_2D(save=True)
    #tsp.make_trader_story_plot_2D(story="most_versatile_seller",save=True)
    #tsp.make_trader_story_plot_2D(story="most_versatile_buyer",save=True)
    tsp.make_trader_story_plot_2D(story="most_distant_trader",save=True)
    #tsp.make_trader_story_plot_2D(story="most_frequent_buyer",save=True)
    #tsp.make_trader_story_plot_2D(story="most_active_trader",save=True)
    #tsp.make_trader_story_plot_2D(story="custom_trader",min_sales=100,min_purchases=100,save=True)
    #tsp.make_distances_biases_plot(save=True, normalize = False)
    #tsp.make_distances_biases_plot(save=True, normalize = True)
    