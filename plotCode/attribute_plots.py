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
        percentages = [round(num,3)*100 for num in percentages]
        counts = [sum(data==i) for i in range(1,7)]

        print(percentages)
        print(counts)


# choose dataset you are interested to investigate
data = "API"

ap = AttributePlotter(dname=data)
ap.plot_trades_per_category(save=True)
#start_date=datetime.datetime(2021,2,1)
#end_date=datetime.datetime(2021,3,1)
#ap.category_analysis(start_date,end_date)
