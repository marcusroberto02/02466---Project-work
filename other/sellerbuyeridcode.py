nfts = df["Unique_id_collection"].unique()
sellers = df["Seller_address"].unique()
buyers = df["Buyer_address"].unique()

trader_df=pd.DataFrame(np.append(sellers,buyers))
traders = trader_df[0].unique()

nft_cat = pd.CategoricalDtype(categories=sorted(nfts))
trader_cat = pd.CategoricalDtype(categories=sorted(traders))

df["Unique_id_collection"] = df["Unique_id_collection"].astype(nft_cat).cat.codes
df["Seller_address"] = df["Seller_address"].astype(trader_cat).cat.codes
df["Buyer_address"] = df["Buyer_address"].astype(trader_cat).cat.codes