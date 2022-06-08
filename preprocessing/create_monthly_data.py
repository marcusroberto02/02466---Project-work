import os
import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# link til data 
# https://osf.io/wsnzr/?view_only=319a53cf1bf542bbbe538aba37916537

# main function
def main():
    # define path for storing data
    path = "./data/ETH/"
    dataset_name = "data_ETH.csv"
    
    # define start and end
    start = datetime.datetime(2020, 10, 1)
    end = start + relativedelta(months =+ 1)

    # mark last month for storing data
    last_month = datetime.datetime(2020, 11, 1)
    
    while start < last_month:
        dataset = pd.DataFrame()
        date = start.strftime("%Y-%m")
        i = 0
        # create path for specific month
        store_path = path + date

        min_date = datetime.datetime(2022, 10, 1)
        max_date = datetime.datetime(2014, 10, 1)
        if not os.path.exists(store_path):
            for chunk in pd.read_csv(path + dataset_name, chunksize=10000, parse_dates=[18]):
                # if max(chunk["Datetime_updated"]) > max_date:
                #     max_date = max(chunk["Datetime_updated"])
                # if min(chunk["Datetime_updated"]) < min_date:
                #     min_date = min(chunk["Datetime_updated"])
                temp = chunk[chunk["Datetime_updated"] >= start]
                dataset = pd.concat([dataset, temp[temp["Datetime_updated"] < end + relativedelta(weeks =+ 1)]])
                if i % 100 == 0:
                    print(i)
                i += 1
            print(min_date,max_date)
            # create directories for storing data
            create_directories(store_path)

            # create dataset for the bipartite model
            create_sparse_data_bi(dataset,end,store_path)

            # create dataset for the bipartite model
            create_sparse_data_tri(dataset,end,store_path)

            # save entire dataframe
            save_data(dataset,end,store_path)

        #Update the start and end dates
        start = end
        end = end + relativedelta(months=+1)

# Create directories for the current month
def create_directories(path):
    os.makedirs(path)
    os.makedirs(path + "/bi")
    os.makedirs(path + "/bi/train")
    os.makedirs(path + "/bi/test")
    os.makedirs(path + "/bi/results")
    os.makedirs(path + "/tri")
    os.makedirs(path + "/tri/train")
    os.makedirs(path + "/tri/test")
    os.makedirs(path + "/tri/results")

def save_sparse_data_bi(df,df_cat,path):
    store_path = path
    df["Unique_id_collection"].to_csv(store_path + '/sparse_i.txt',header=None,index=None)
    df["Trader_address"].to_csv(store_path + '/sparse_j.txt',header=None,index=None)
    df_cat["Category"].to_csv(store_path + '/sparse_c.txt',header=None,index=None)
    df["Count"].to_csv(store_path + '/sparse_w.txt',header=None,index=None)

    with open(store_path + "/info.txt", "w") as f:
        f.write("Number of unique traders: " + str(len(set(df["Trader_address"]))) + "\n")
        f.write("Number of unique NFTs: " + str(len(set(df["Unique_id_collection"]))) + "\n")
        f.write("Number of unique categories: " + str(len(set(df["Category"]))))

def create_sparse_data_bi(df,end,path):
    # Hacks a way to get buyers as an extention of sellers
    df_buyers = df.copy()
    df_buyers["Seller_address"] = df_buyers["Buyer_address"]
    df = pd.concat([df, df_buyers])
    df = df.rename(columns = {'Seller_address':'Trader_address'})

    # factorizes data and splits into test and training 
    df_train = df[df["Datetime_updated"] < end]
    df_test = df[df["Datetime_updated"] >= end]

    # remove all rows in test set that contains unseen nfts or traders
    df_test = df_test[df_test["Unique_id_collection"].isin(df_train["Unique_id_collection"])]
    df_test = df_test[df_test["Trader_address"].isin(df_train["Trader_address"])]

    # concatenate train and test to get identical ids
    df = pd.concat([df_train,df_test])
    facts = ["Unique_id_collection", "Trader_address"]
    df[facts] = df[facts].apply(lambda x: pd.factorize(x)[0])
    df_test = df[df["Datetime_updated"] >= end]
    df_train = df[df["Datetime_updated"] < end]

    #Removes dublicate trades and increments counter
    df_train = df_train.groupby(["Unique_id_collection", "Trader_address", "Category"]).size().reset_index()
    df_test = df_test.groupby(["Unique_id_collection","Trader_address","Category"]).size().reset_index()
    
    # for getting categories
    df_cat = df_train.groupby(["Unique_id_collection","Category"]).size().reset_index()

    # rename count column
    df_train.columns = ["Unique_id_collection","Trader_address","Category","Count"]
    df_test.columns = ["Unique_id_collection","Trader_address","Category","Count"]

    # save files
    save_sparse_data_bi(df_train,df_cat,path+"/bi/train")
    save_sparse_data_bi(df_test,df_cat,path+"/bi/test")
    
def save_sparse_data_tri(df,df_cat,path):
    store_path = path
    # save files
    df["Unique_id_collection"].to_csv(store_path +'/sparse_i.txt',header=None,index=None)
    df["Seller_address"].to_csv(store_path + '/sparse_j.txt',header=None,index=None)
    df["Buyer_address"].to_csv(store_path + '/sparse_k.txt',header=None,index=None)
    df_cat["Category"].to_csv(store_path + '/sparse_c.txt',header=None,index=None)
    df["Count"].to_csv(store_path + '/sparse_w.txt',header=None,index=None)
    #write to a text file
    with open(store_path + "/info.txt", "w") as f:
        f.write("Number of unique sellers: " + str(len(set(df["Seller_address"]))) + "\n")
        f.write("Number of unique buyers: " + str(len(set(df["Buyer_address"]))) + "\n")
        f.write("Number of unique NFTs: " + str(len(set(df["Unique_id_collection"]))) + "\n")
        f.write("Number of unique categories: " + str(len(set(df["Category"]))))

def create_sparse_data_tri(df,end,path):
    # used for factorization
    facts = ["Unique_id_collection","Seller_address","Buyer_address"]
    
    # temporary split into test and train
    df_train = df[df["Datetime_updated"] < end]
    df_test = df[df["Datetime_updated"] >= end]
    
    # remove entries from the test set that dont appear in the training set
    df_test = df_test[df_test["Unique_id_collection"].isin(df_train["Unique_id_collection"])]
    df_test = df_test[df_test["Seller_address"].isin(df_train["Seller_address"])]
    df_test = df_test[df_test["Buyer_address"].isin(df_train["Buyer_address"])]
    
    # factorize before the data is split into train and test
    # this is done to avoid the same categories being used in both train and test
    # remove all rows in test set that contains unseen nfts or traders
    # concatenate train and test to get identical ids
    df = pd.concat([df_train,df_test])
    nft = df["Unique_id_collection"].unique()
    seller = df["Seller_address"].unique()
    buyer = df["Buyer_address"].unique()
    
    trader_df=pd.DataFrame(np.append(seller,buyer))
    traders = trader_df[0].unique()

    nft_cat = pd.CategoricalDtype(categories=sorted(nft))
    trader_cat = pd.CategoricalDtype(categories=sorted(traders))
    
    df["Unique_id_collection"] = df["Unique_id_collection"].astype(nft_cat).cat.codes
    df["Seller_address"] = df["Seller_address"].astype(trader_cat).cat.codes
    df["Buyer_address"] = df["Buyer_address"].astype(trader_cat).cat.codes

    
    #df[facts] = df[facts].apply(lambda x: pd.factorize(x)[0])
    
    # split into train and test based on the date
    df_test = df[df["Datetime_updated"] >= end]
    df_train = df[df["Datetime_updated"] < end]

    #Removes dublicate trades and increments counter
    df_train = df_train.groupby(["Unique_id_collection","Seller_address","Buyer_address","Category"]).size().reset_index()
    df_test = df_test.groupby(["Unique_id_collection","Seller_address","Buyer_address","Category"]).size().reset_index()

    # for getting categories
    df_cat = df_train.groupby(["Unique_id_collection","Category"]).size().reset_index()

    # rename count column
    df_train.columns = ["Unique_id_collection","Seller_address","Buyer_address","Category","Count"]
    df_test.columns = ["Unique_id_collection","Seller_address","Buyer_address","Category","Count"]

    save_sparse_data_tri(df_train,df_cat,path+"/tri/train")
    save_sparse_data_tri(df_test,df_cat,path+"/tri/test")

def save_data(df,end,path):
    # split into train and test based on the date
    train = df[df["Datetime_updated"] < end]
    test = df[df["Datetime_updated"] >= end]
    
    #Save test and train datasets
    train.to_csv(path + "/data_train.csv")
    test.to_csv(path + "/data_test.csv")

main()




