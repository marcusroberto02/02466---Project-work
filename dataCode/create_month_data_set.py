import os
import pandas
import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

# link til data 
# https://osf.io/wsnzr/?view_only=319a53cf1bf542bbbe538aba37916537


def sparse_tri_data(data_path, dataset):

    df = pandas.read_csv(data_path + dataset)[["Unique_id_collection","Seller_address","Buyer_address","Category"]]
    #Removes dublicate trades and increments counter
    df = df.groupby(["Unique_id_collection","Seller_address","Buyer_address","Category"]).size().reset_index()
    # rename count column
    df.columns = ["Unique_id_collection","Seller_address","Buyer_address","Category","Count"]
    store_path = data_path + "tri/"
    # save files
    df["Unique_id_collection"].to_csv(store_path +'sparse_i.txt',header=None,index=None)
    df["Seller_address"].to_csv(store_path + 'sparse_j.txt',header=None,index=None)
    df["Buyer_address"].to_csv(store_path + 'sparse_k.txt',header=None,index=None)
    df["Category"].to_csv(store_path + 'sparse_c.txt',header=None,index=None)
    df["Count"].to_csv(store_path + 'sparse_w.txt',header=None,index=None)
    #write to a text file
    with open(store_path + "Info.txt", "w") as f:
        f.write("Number of unique Sellers: " + str(len(set(df["Seller_address"]))) + "\n")
        f.write("Number of unique Buyers: " + str(len(set(df["Buyer_address"]))) + "\n")
        f.write("Number of unique NFT: " + str(len(set(df["Unique_id_collection"]))) + "\n")
        f.write("Number of unique categories: " + str(len(set(df["Category"]))))


def sparse_bi_data(df,path, end):
    # Hacks a way to get buyers as an extention of sellers
    df_buyers = df.copy()
    df_buyers["Seller_address"] = df_buyers["Buyer_address"]
    df = pd.concat([df, df_buyers])
    df = df.rename(columns = {'Seller_address':'Trader_address'})

    # factorizes data and splits into test and training data
    facts = ["Unique_id_collection", "Trader_address", "Category"]
    df[facts] = df[facts].apply(lambda x: pandas.factorize(x)[0])
    df_test = df[df["Datetime_updated"] >= end]
    df_train = df[df["Datetime_updated"] < end]

    #Removes dublicate trades and increments counter
    df_test = df_test.groupby(["Unique_id_collection","Trader_address","Category"]).size().reset_index()
    df_train = df_train.groupby(["Unique_id_collection", "Trader_address", "Category"]).size().reset_index()
    # rename count column
    df_test.columns = ["Unique_id_collection","Trader_address","Category","Count"]
    df_train.columns = ["Unique_id_collection","Trader_address","Category","Count"]

    # save files
    store_path = path + "/train/" + "bi/"
    df_train["Unique_id_collection"].to_csv(store_path + 'sparse_i.txt',header=None,index=None)
    df_train["Trader_address"].to_csv(store_path + 'sparse_j.txt',header=None,index=None)
    df_train["Category"].to_csv(store_path + 'sparse_c.txt',header=None,index=None)
    df_train["Count"].to_csv(store_path + 'sparse_w.txt',header=None,index=None)

    with open(store_path + "Info.txt", "w") as f:
        f.write("Number of unique Traders: " + str(len(set(df_train["Trader_address"]))) + "\n")
        f.write("Number of unique NFT: " + str(len(set(df_train["Unique_id_collection"]))) + "\n")
        f.write("Number of unique categories: " + str(len(set(df_train["Category"]))))

    store_path = path + "/test/" + "bi/"
    df_test["Unique_id_collection"].to_csv(store_path + 'sparse_i.txt', header=None, index=None)
    df_test["Trader_address"].to_csv(store_path + 'sparse_j.txt', header=None, index=None)
    df_test["Category"].to_csv(store_path + 'sparse_c.txt', header=None, index=None)
    df_test["Count"].to_csv(store_path + 'sparse_w.txt', header=None, index=None)

    with open(store_path + "Info.txt", "w") as f:
        f.write("Number of unique Traders: " + str(len(set(df_test["Trader_address"]))) + "\n")
        f.write("Number of unique NFT: " + str(len(set(df_test["Unique_id_collection"]))) + "\n")
        f.write("Number of unique categories: " + str(len(set(df_test["Category"]))))


path = "../data/"
start = datetime.datetime(2019, 1, 1)
end = start + relativedelta(months =+ 1)

while start < datetime.datetime(2019, 4, 1):
    dataset = pandas.DataFrame()
    date = start.strftime("%Y-%m")
    # check if a directory exists for the current month and
    if not os.path.exists(path + date):
        for chunk in pandas.read_csv("../data/Data_API.csv", chunksize=10000, parse_dates=[18]):
            temp = chunk[chunk["Datetime_updated"] >= start]
            dataset = pandas.concat([dataset, temp[temp["Datetime_updated"] < end + relativedelta(weeks =+ 1)]])

        # A copy of the dataset used to get the sparse text files for the bi partite case
        df = dataset.copy()
        facts = ["Unique_id_collection","Seller_address","Buyer_address","Category"]
        # factorize before the data is split into train and test
        # this is done to avoid the same categories being used in both train and test
        dataset[facts] = dataset[facts].apply(lambda x: pandas.factorize(x)[0])
        # split into train and test based on the date
        test = dataset[dataset["Datetime_updated"] >= end]
        dataset = dataset[dataset["Datetime_updated"] < end]

        #Create directories for the current month
        os.makedirs(path + date)
        os.makedirs(path + date + "/train")
        os.makedirs(path + date + "/train/bi")
        os.makedirs(path + date + "/train/tri")
        os.makedirs(path + date + "/test")
        os.makedirs(path + date + "/test/bi")
        os.makedirs(path + date + "/test/tri")
        #Save test and train datasets
        dataset.to_csv(path + date + "/train/data_train.csv")
        test.to_csv(path + date + "/test/data_test.csv")

        #Save the sparse text files
        sparse_tri_data(path + date + "/train/","data_train.csv")
        sparse_tri_data(path + date + "/test/", "data_test.csv")
        sparse_bi_data(df,path + date,end)

    #Update the start and end dates
    start = end
    end = end + relativedelta(months=+1)





