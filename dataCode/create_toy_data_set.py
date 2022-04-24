import pandas

# link til data 
# https://osf.io/wsnzr/?view_only=319a53cf1bf542bbbe538aba37916537

small_dataset = pandas.DataFrame()
i = 0
<<<<<<< HEAD
for chunk in pandas.read_csv("./Data/Data_API.csv", chunksize=10000):
    small_dataset = pandas.concat([small_dataset, chunk.sample(frac=0.0002, random_state = 1)])
=======
for chunk in pandas.read_csv(
        "C:/Users/khelp/OneDrive/Desktop/4. semester/Fagprojekt/02466---Project-work/data/Data_API.csv",
        chunksize=10000):
    small_dataset = pandas.concat([small_dataset, chunk.sample(frac=0.002, random_state=1)])
>>>>>>> 2b2ecd52e840b7b0fdcf13925ea48eaa10de762c
    if i % 100 == 0:
        print(i)
    i += 1

# sellers_buyers_table = small_dataset[['Seller_address','Buyer_address','Unique_id_collection']]

small_dataset.to_csv('C:/Users/khelp/OneDrive/Desktop/4. semester/Fagprojekt/02466---Project-work/data/small_dataset.csv')

