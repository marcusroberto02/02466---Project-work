import pandas

small_dataset = pandas.DataFrame()
i = 0
for chunk in pandas.read_csv("./Data/Data_API.csv", chunksize=10000):
    small_dataset = pandas.concat([small_dataset, chunk.sample(frac=0.02, random_state = 1)])
    break
    
#sellers_buyers_table = small_dataset[['Seller_address','Buyer_address','Unique_id_collection']]

small_dataset.to_csv('./Data/small_toy_data_set.csv')

