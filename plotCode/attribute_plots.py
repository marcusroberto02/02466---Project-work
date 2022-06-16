import gc

from plot_formatting import Formatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import platform
from dateutil.relativedelta import relativedelta

###################
# ATTRIBUTE PLOTS #
###################

class AttributePlotter(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.95,0.90) 

    # standard line width
    linewidth = 5

    # color for embeddings
    colors = {'Art':'green','Collectible':'blue','Games':'red','Metaverse':'orange','Other':'purple','Utility':'brown'}
    
    # name for the dataset
    namedict = {"API":"Full dataset","ETH":"Ethereum blockchain","WAX":"WAX blockchain"}

    def __init__(self,dname="API",month="2021-02"):
        self.initialize_fontsizes_big()
        super().__init__(month=month)
        self.dname = dname
        self.load_data()
        self.store_path = self.figurebase + "/AttributePlots"
        self.fig = plt.figure(figsize=self.figsize)
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)    

    def load_data(self):
        if platform.system() == "Linux" or platform.system() == "Darwin":
            ndots = ".."
        else:
            ndots = "."
        path = "{ndots}/data/Data_{dname}.csv".format(ndots=ndots,dname=self.dname)
        self.df = pd.read_csv(path, low_memory=True)

    def plot_trades_per_category(self,start_date=datetime.datetime(2017,11,1),end_date=datetime.datetime(2021,4,30),padding=False,save=False,show=False):
        self.df['Datetime_updated'] = pd.to_datetime(self.df['Datetime_updated'], format='%Y-%m-%d')
        lower_limit = self.df['Datetime_updated'] >= start_date
        upper_limit = self.df['Datetime_updated'] <= end_date
        self.df = self.df.loc[lower_limit]
        self.df = self.df.loc[upper_limit]
        self.df = self.df.reset_index(drop=True)
        self.df['Datetime_updated'] = self.df['Datetime_updated'].dt.to_period('M')
        data = self.df.groupby(['Datetime_updated','Category']).size().unstack(fill_value=0).stack()
        x = data.index.get_level_values('Datetime_updated').unique()
        x = x.astype(str)

        # add months for the WAX dataset
        if self.dname == "WAX" and padding:
            remaining_months = [(start_date+relativedelta(months =+ i)).strftime("%Y-%m") for i in range(0,31)]
            x = np.concatenate((remaining_months,x),axis=None)
        self.fig = plt.figure(figsize=self.figsize)

        for category, color in self.colors.items():
            y = data.loc[(data.index.get_level_values('Category') == category)].values
            # remove zeros from plot
            y = y.astype(float)
            y[y<=0] = np.nan
            if self.dname == "WAX" and padding:
                remaining_months = [np.nan for _ in range(0,31)]
                y = np.concatenate((remaining_months,y),axis=None)
            plt.plot(x, y, color=color,label = category,linewidth=self.linewidth)
        
        if self.dname == "WAX":
            plt.legend(loc="upper left")
        else:
            plt.legend(loc="lower right")
        subtitle = self.namedict[self.dname]
        self.format_plot(title="Number of trades pr. month",subtitle=subtitle,title_y=self.fig_title_y,xlabel="Month",ylabel="Number of trades")

        ax = plt.gca()
        n = 5 if not (self.dname == "WAX" and not padding) else 2
        print(n)
        if (self.dname == "WAX" and padding):
            plt.xticks(range(len(x)),x)
        tickboolean = lambda i : ((i-1) % n != 0 and i != 0) or (i == 1) if not (self.dname == "WAX" and not padding) else i % n != 0
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if tickboolean(i)]
        

        plt.xticks(rotation="vertical")
        plt.yscale('log')
        plt.ylim(1e0)
        
        if save:
            plt.savefig("{path}/trades_per_category_plot_{dname}".format(path=self.store_path,dname=self.dname))
        if show:
            plt.show()
            


    def category_analysis(self,start_date,end_date):
        self.df['Datetime_updated'] = pd.to_datetime(self.df['Datetime_updated'], format='%Y-%m-%d')
        lower_limit = self.df['Datetime_updated'] >= start_date
        upper_limit = self.df['Datetime_updated'] <= end_date
        self.df = self.df.loc[lower_limit]
        self.df = self.df.loc[upper_limit]
        self.df = self.df.reset_index(drop=True)
        df_buyers = self.df.copy()
        df_buyers["Seller_address"] = df_buyers["Buyer_address"]
        self.df = pd.concat([self.df, df_buyers])
        self.df = self.df.rename(columns = {'Seller_address':'Trader_address'})
        data = self.df.groupby(['Trader_address', 'Category']).size()
        data = data.groupby(['Trader_address']).size()

        total = len(data)
        percentages = [sum(data==i)/total for i in range(1,7)]  
        percentages = [round(num,3)*100 for num in percentages]
        counts = [sum(data==i) for i in range(1,7)]

        print(percentages)
        print(counts)

    def load_datasets(self, dataset="API"):
        if dataset == self.dname:
            data = self.df
        else:
            data =  pd.read_csv("../data/Data_{dataset}.csv".format(dataset=dataset), low_memory=True)

        seller_count = data.groupby('Seller_address').size().values
        buyer_count = data.groupby('Buyer_address').size().values
        nft_count = data.groupby('Unique_id_collection').size().values
        traders = np.append(data['Seller_address'], data['Buyer_address'])
        trad_df = pd.DataFrame(traders)
        trad_count = trad_df.groupby(0).size().values

        del (data)
        gc.collect()

        return seller_count, buyer_count, trad_count, nft_count


    def degree_heterogeneity_plot(self, save=False, show=False, dataset="All Data"):
        # Collect all counts from all three datasets
        seller_count, buyer_count, trader_count, nft_count = self.load_datasets(dataset="API")
        eth_seller_count, eth_buyer_count, eth_trader_count, eth_nft_count = self.load_datasets(dataset="ETH")
        wax_seller_count, wax_buyer_count, wax_trader_count, wax_nft_count = self.load_datasets(dataset="WAX")

        # Log-Log seller plot
        min_bin, max_bin = min(seller_count), max(seller_count)
        min_eth_bin, max_eth_bin = min(eth_seller_count), max(eth_seller_count)
        min_wax_bin, max_wax_bin = min(wax_seller_count), max(wax_seller_count)

        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)
        eth_bins = np.logspace(np.log10(min_eth_bin), np.log10(max_eth_bin), 20)
        wax_bins = np.logspace(np.log10(min_wax_bin), np.log10(max_wax_bin), 20)

        hist, edges = np.histogram(seller_count, bins=bins, density=False)
        eth_hist, eth_edges = np.histogram(eth_seller_count, bins=eth_bins, density=False)
        wax_hist, wax_edges = np.histogram(wax_seller_count, bins=wax_bins, density=False)

        x = (edges[1:] + edges[:-1]) / 2
        eth_x = (eth_edges[1:] + eth_edges[:-1])/2
        wax_x = (wax_edges[1:] + wax_edges[:-1])/2

        fig, axs = plt.subplots(2, 2, figsize=(20,20))

        axs[0, 0].plot(x, hist, marker='.', label="All data")
        axs[0, 0].plot(eth_x, eth_hist, marker='.', label = "Ethereum")
        axs[0, 0].plot(wax_x, wax_hist, marker='.', label="WAX")
        axs[0, 0].set_xlabel('Sellers')
        axs[0, 0].set_ylabel('Counts')
        axs[0, 0].set_xscale('log')
        axs[0, 0].set_yscale('log')
        axs[0, 0].grid(True)
        axs[0,0].legend()


        # Log-Log Buyer plot

        min_bin, max_bin = min(buyer_count), max(buyer_count)
        min_eth_bin, max_eth_bin = min(eth_buyer_count), max(eth_buyer_count)
        min_wax_bin, max_wax_bin = min(wax_buyer_count), max(wax_buyer_count)

        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)
        eth_bins = np.logspace(np.log10(min_eth_bin), np.log10(max_eth_bin), 20)
        wax_bins = np.logspace(np.log10(min_wax_bin), np.log10(max_wax_bin), 20)

        hist, edges = np.histogram(seller_count, bins=bins, density=False)
        eth_hist, eth_edges = np.histogram(eth_buyer_count, bins=eth_bins, density=False)
        wax_hist, wax_edges = np.histogram(wax_buyer_count, bins=wax_bins, density=False)

        x_b = (edges[1:] + edges[:-1]) / 2
        eth_x_b = (eth_edges[1:] + eth_edges[:-1]) / 2
        wax_x_b = (wax_edges[1:] + wax_edges[:-1]) / 2

        axs[0, 1].plot(x_b, hist, marker='.')
        axs[0, 1].plot(eth_x_b, eth_hist, marker='.', label="Ethereum")
        axs[0, 1].plot(wax_x_b, wax_hist, marker='.', label="WAX")

        axs[0, 1].set_xlabel('Buyers')
        axs[0, 1].set_ylabel('Counts')
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_yscale('log')
        axs[0, 1].grid(True)


        # Log-Log NFT plot
        min_bin, max_bin = min(nft_count), max(nft_count)
        min_eth_bin, max_eth_bin = min(eth_nft_count), max(eth_nft_count)
        min_wax_bin, max_wax_bin = min(wax_nft_count), max(wax_nft_count)

        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)
        eth_bins = np.logspace(np.log10(min_eth_bin), np.log10(max_eth_bin), 20)
        wax_bins = np.logspace(np.log10(min_wax_bin), np.log10(max_wax_bin), 8)

        hist, edges = np.histogram(nft_count, bins=bins, density=False)
        eth_hist, eth_edges = np.histogram(eth_nft_count, bins=eth_bins, density=False)
        wax_hist, wax_edges = np.histogram(wax_nft_count, bins=wax_bins, density=False)

        x_n = (edges[1:] + edges[:-1]) / 2
        eth_x_n = (eth_edges[1:] + eth_edges[:-1]) / 2
        wax_x_n = (wax_edges[1:] + wax_edges[:-1]) / 2

        axs[1, 1].plot(x_n, hist, marker='.', label = "All data")
        axs[1, 1].plot(eth_x_n, eth_hist, marker='.', label = "Ethereum")
        axs[1, 1].plot(wax_x_n, wax_hist, marker='.', label = "WAX")
        axs[1, 1].set_xlabel('NFTs')
        axs[1, 1].set_ylabel('Counts')
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_yscale('log')
        axs[1, 1].grid(True)

        #Log-log Traders plot
        min_bin, max_bin = min(trader_count), max(trader_count)
        min_eth_bin, max_eth_bin = min(eth_trader_count), max(eth_trader_count)
        min_wax_bin, max_wax_bin = min(wax_trader_count), max(wax_trader_count)

        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)
        eth_bins = np.logspace(np.log10(min_eth_bin), np.log10(max_eth_bin), 20)
        wax_bins = np.logspace(np.log10(min_wax_bin), np.log10(max_wax_bin), 20)

        hist, edges = np.histogram(seller_count, bins=bins, density=False)
        eth_hist, eth_edges = np.histogram(eth_trader_count, bins=eth_bins, density=False)
        wax_hist, wax_edges = np.histogram(wax_trader_count, bins=wax_bins, density=False)

        x_t = (edges[1:] + edges[:-1]) / 2
        eth_x_t = (eth_edges[1:] + eth_edges[:-1]) / 2
        wax_x_t = (wax_edges[1:] + wax_edges[:-1]) / 2

        axs[1, 0].plot(x_t, hist, marker='.', label="All data")
        axs[1, 0].plot(eth_x_t, eth_hist, marker='.', label="Ethereum")
        axs[1, 0].plot(wax_x_t, wax_hist, marker='.', label="WAX")
        axs[1, 0].set_xlabel('Traders')
        axs[1, 0].set_ylabel('Counts')
        axs[1, 0].set_xscale('log')
        axs[1, 0].set_yscale('log')
        axs[1, 0].grid(True)

        plt.tight_layout()
        plt.title("Degree distribution plot")
        plt.legend()
        #plt.rcParams['figure.figsize'] = [30, 20]

        #plt.rcParams['figure.figsize'] = [12, 8]
        #self.format_plot(title="Degree Heterogeneity", subtitle="Dataset: {blockchain}".format(blockchain=dataset), xlabel='Entity', ylabel='Occurrences')
        if save:
            plt.savefig()
        if show:
            plt.show()


# choose dataset you are interested to investigate
data = "API"

ap = AttributePlotter(dname=data)
ap.plot_trades_per_category(save=True)
#start_date=datetime.datetime(2021,2,1)
#end_date=datetime.datetime(2021,3,1)
#ap.category_analysis(start_date,end_date)
#ap.plot_trades_per_category(save=True)
ap.degree_heterogeneity_plot(show=True)

#category_analysis(df,start_date,end_date)
