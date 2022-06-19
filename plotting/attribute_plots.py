from plot_formatting import Formatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import platform
from dateutil.relativedelta import relativedelta
import gc

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
    markersize = 15

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
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)    

    def load_data(self):
        ndots = "." if platform.system() != "Darwin" else ".."
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
        percentages = [num*100 for num in percentages]
        counts = [sum(data==i) for i in range(1,7)]

        print(percentages)
        print(counts)

    def load_datasets(self, dataset="API"):
        if dataset == self.dname:
            data = self.df
        else:
            data =  pd.read_csv("./data/Data_{dataset}.csv".format(dataset=dataset), low_memory=True)

        seller_count = data.groupby('Seller_address').size().values
        buyer_count = data.groupby('Buyer_address').size().values
        nft_count = data.groupby('Unique_id_collection').size().values
        traders = np.append(data['Seller_address'], data['Buyer_address'])
        trad_df = pd.DataFrame(traders)
        trad_count = trad_df.groupby(0).size().values

        del (data)
        gc.collect()

        return seller_count, buyer_count, trad_count, nft_count


    def make_degree_heterogeneity_plot(self,save=False, show=False):
        # Collect all counts from all three datasets
        full_seller_count, full_buyer_count, full_trader_count, full_nft_count = self.load_datasets(dataset="API")
        eth_seller_count, eth_buyer_count, eth_trader_count, eth_nft_count = self.load_datasets(dataset="ETH")
        wax_seller_count, wax_buyer_count, wax_trader_count, wax_nft_count = self.load_datasets(dataset="WAX")

        counts = [[full_seller_count,eth_seller_count,wax_seller_count],
                  [full_buyer_count,eth_buyer_count,wax_buyer_count],
                  [full_trader_count,eth_trader_count,wax_trader_count],
                  [full_nft_count,eth_nft_count,wax_nft_count]]
        
        names = ["Sellers","Buyers","Traders","NFTs"]

        for (count, name) in zip(counts,names):
            self.fig = plt.figure(figsize=self.figsize)
            full_count, eth_count, wax_count = count
            min_bin, max_bin = min(full_count), max(full_count)
            min_eth_bin, max_eth_bin = min(eth_count), max(eth_count)
            min_wax_bin, max_wax_bin = min(wax_count), max(wax_count)

            wb = 20 if name != "NFTs" else 8

            bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)
            eth_bins = np.logspace(np.log10(min_eth_bin), np.log10(max_eth_bin), 20)
            wax_bins = np.logspace(np.log10(min_wax_bin), np.log10(max_wax_bin), wb)

            hist, edges = np.histogram(full_count, bins=bins, density=False)
            eth_hist, eth_edges = np.histogram(eth_count, bins=eth_bins, density=False)
            wax_hist, wax_edges = np.histogram(wax_count, bins=wax_bins, density=False)

            x = (edges[1:] + edges[:-1]) / 2
            eth_x = (eth_edges[1:] + eth_edges[:-1])/2
            wax_x = (wax_edges[1:] + wax_edges[:-1])/2

            plt.plot(x, hist, marker='o', mfc='black', markersize=self.markersize,lw=self.linewidth,label="All data")
            plt.plot(eth_x, eth_hist, marker='o', mfc='black', markersize=self.markersize,lw=self.linewidth,label = "Ethereum")
            plt.plot(wax_x, wax_hist, marker='o',mfc='black', markersize=self.markersize,lw=self.linewidth, label="WAX")
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.legend(loc="upper right")

            self.format_plot(title="Degree plot",subtitle=name,title_y=self.fig_title_y,xlabel=name,ylabel="Number of trades")
            if save:
                plt.savefig("{path}/degree_heterogeneity_plot_{name}".format(path=self.store_path,name=name))
            if show:
                plt.show()
    

    def make_month_node_count_plot(self,save=False,show=False):

        self.fig = plt.figure(figsize=self.figsize)

        start_date = datetime.datetime(2017,11,1) if self.dname != "WAX" else datetime.datetime(2020,6,1)
        end_date = start_date + relativedelta(months =+ 1)

        months = []
        
        nft_count = []
        trader_count = []
        seller_count = []
        buyer_count = []

        self.df['Datetime_updated'] = pd.to_datetime(self.df['Datetime_updated'], format='%Y-%m-%d')

        while start_date < datetime.datetime(2021,5,1):
            month = start_date.strftime('%Y-%m')
            months.append(month)

            print(month)

            lower_limit = self.df['Datetime_updated'] >= start_date
            upper_limit = self.df['Datetime_updated'] <  end_date
            df_month = self.df.loc[lower_limit]
            df_month = df_month.loc[upper_limit]
            df_month = df_month.reset_index(drop=True)
            
            # count number of unique number of nfts, sellers and buyers 
            nft_count.append(len(np.unique(df_month["Unique_id_collection"])))
            seller_count.append(len(np.unique(df_month["Seller_address"])))
            buyer_count.append(len(np.unique(df_month["Buyer_address"])))
            
            # count number of traders
            df_buyers = df_month.copy()
            df_buyers["Seller_address"] = df_buyers["Buyer_address"]
            df_month = pd.concat([df_month, df_buyers])
            df_month = df_month.rename(columns = {'Seller_address':'Trader_address'})
            trader_count.append(len(np.unique(df_month["Trader_address"])))

            # update date
            start_date = end_date
            end_date = start_date + relativedelta(months =+ 1)
        
        plt.plot(months, nft_count,lw=self.linewidth,label="NFTs")
        plt.plot(months, trader_count,lw=self.linewidth,label="Traders")
        plt.plot(months, seller_count,lw=self.linewidth,label="Sellers")
        plt.plot(months, buyer_count,lw=self.linewidth,label="Buyers")

        plt.legend(loc="lower right")
        subtitle = self.namedict[self.dname]
        self.format_plot(title="Number of unique elements pr. month",subtitle=subtitle,title_y=self.fig_title_y,xlabel="Month",ylabel="Count")

        ax = plt.gca()
        n = 5 if not self.dname == "WAX" else 2
        print(n)
        if self.dname == "WAX":
            plt.xticks(range(len(months)),months)
        tickboolean = lambda i : ((i-1) % n != 0 and i != 0) or (i == 1) if not self.dname == "WAX" else i % n != 0
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if tickboolean(i)]
        
        plt.xticks(rotation="vertical")
        plt.yscale('log')
        plt.ylim(1e0)
        
        if save:
            plt.savefig("{path}/month_count_plot_{dname}".format(path=self.store_path,dname=self.dname))
        if show:
            plt.show()

            

        



# choose dataset you are interested to investigate
data = ["ETH","WAX","API"]

for dname in data:
    ap = AttributePlotter(dname=dname)
    #ap.make_month_node_count_plot(save=True)
#ap.plot_trades_per_category(save=True)
#start_date=datetime.datetime(2017,11,1)
#end_date=datetime.datetime(2021,4,30)
#ap.category_analysis(start_date,end_date)
#ap.plot_trades_per_category(save=True)
#ap.make_degree_heterogeneity_plot(save=True)
#category_analysis(df,start_date,end_date)
