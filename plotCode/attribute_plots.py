import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import datetime

def plot_trade_per_category(df, start_date, end_date):
    df['Datetime_updated'] = pd.to_datetime(df['Datetime_updated'], format='%Y-%m-%d')
    lower_limit = df['Datetime_updated'] >= start_date
    upper_limit = df['Datetime_updated'] <= end_date
    df = df.loc[lower_limit]
    df = df.loc[upper_limit]
    df = df.reset_index(drop=True)
    df['Datetime_updated'] = df['Datetime_updated'].dt.to_period('M')
    data = df.groupby(['Datetime_updated','Category']).size().unstack(fill_value=0).stack()
    x = data.index.get_level_values('Datetime_updated').unique()
    x = x.astype(str)
    y1 = data.loc[(data.index.get_level_values('Category') == 'Art')].values
    y2 = data.loc[(data.index.get_level_values('Category') == 'Collectible')].values
    y3 = data.loc[(data.index.get_level_values('Category') == 'Games')].values
    y4 = data.loc[(data.index.get_level_values('Category') == 'Metaverse')].values
    y5 = data.loc[(data.index.get_level_values('Category') == 'Other')].values
    y6 = data.loc[(data.index.get_level_values('Category') == 'Utility')].values
    plt.plot(x, y1, label = 'Art')
    plt.plot(x, y2, label = 'Collectible')
    plt.plot(x, y3, label = 'Games')
    plt.plot(x, y4, label = 'Metaverse')
    plt.plot(x, y5, label = 'Other')
    plt.plot(x, y6, label = 'Utility')
    plt.legend()
    plt.title('trades per category for each month')
    plt.xlabel('Month')
    plt.ylabel('trades')
    plt.xticks(rotation='vertical')
    plt.yscale('log')
    plt.ylim(1e0)
    plt.show()


def category_analysis(df, start_date, end_date):
    df['Datetime_updated'] = pd.to_datetime(df['Datetime_updated'], format='%Y-%m-%d')
    lower_limit = df['Datetime_updated'] >= start_date
    upper_limit = df['Datetime_updated'] <= end_date
    df = df.loc[lower_limit]
    df = df.loc[upper_limit]
    df = df.reset_index(drop=True)
    df_buyers = df.copy()
    df_buyers["Seller_address"] = df_buyers["Buyer_address"]
    df = pd.concat([df, df_buyers])
    df = df.rename(columns = {'Seller_address':'Trader_address'})
    data = df.groupby(['Trader_address', 'Category']).size()
    data = data.groupby(['Trader_address']).size()

    total = len(data)
    percentages = [sum(data==i)/total for i in range(1,7)]  
    percentages = [round(num,3)*100 for num in percentages]
    counts = [sum(data==i) for i in range(1,7)]

    print(percentages)
    print(counts)


path = "C:/Users/khelp/OneDrive/Desktop/DTU/4. semester/Fagprojekt/"
name = "Data_API.csv"
df = pd.read_csv(path + name, low_memory=True)

start_date = datetime.datetime(2017,11,1)
end_date = datetime.datetime(2021,4,30)

plot_trade_per_category(df, start_date, end_date)
category_analysis(df,start_date,end_date)
