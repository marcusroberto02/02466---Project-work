import gzip
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/marcu/Downloads/Data_API.csv.gz')

print(len(np.unique(df["Unique_id_collection"])))     
print(len(np.unique(df["Seller_address"])))
print(len(np.unique(df["Buyer_address"])))     