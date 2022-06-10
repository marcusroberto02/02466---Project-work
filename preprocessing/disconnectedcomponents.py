from cgi import test
from email.policy import default
import pandas as pd
import numpy as np
from collections import defaultdict,deque

#print(test)
path = './data/'
dataset = "Data_API.csv"
df = pd.read_csv(path + dataset)

cryptos = ['0xBTC','1MT','2XDN','ABST','AMPL','ANRX','ARCONA','ART','ASLT','ATRI',
 'AVRT','B0T','BAEPAY','BASED','BAT','BLVD','BON','BONDLY','BONES','BOOB',
 'BORG','BPC','BUDGET','BUDZ','BZN','CAMEL','CGG','CHERRY','CHONK','COIN',
 'COVAL','CPT','CRED','CUBE','CURIO','DAI','DAPPT','DDIM','DENA','DGX',
 'DHC','DOOM','DUST','EBB','ECAT','ECTO','EGGS','ELAND','ELET','EMONT',
 'ENJ','EPIC','ETH','FIRST','FRFY','FTHR','FTM','FUD','GALA','GCASH','GEM',
 'GMX','GOKU','GOOSE','GOU','GPL','GUSD','HIVED','HOUR','HUE','IMP','INI',
 'INK','JBG','JGN','KAP','KEK','KEK-DEP','KING','KIWI','KLTR','KOI','LAR',
 'LESS','LINK','LIT','LOAD','MANA','MATIC','MBC','MCX','MEME','MGDv2',
 'MKR','MM','MNFT','MORK','MX','NDR','NEWS','NUGS','NVT','OLDWAXIE',
 'OLDWZX','PGU','PIXEL','PIXIE','PLAY','PMON','PPDEX','PRIME','PXART',
 'PYRO','RAINBOW','RARE','RARI','RCC','RCDY','REVV','RLY','RODZ','ROPE',
 'ROT','RUGZ','SAI','SAL','SAND','SKULL','SLP','SMTS','SURF','SWAG','TATR',
 'THREE','TRIP','TRISM','TRSH','UNI','USDC','VEGETA','VI','VIDEO','VIDT',
 'VSF','WAIF','WAX','WAXIE','WBTC','WCK','WETH','WGM','WIPC','WMC','XMON',
 'YGT','YUMI','ZIOT','ZURU','ZUT','eFAME']

# keeps track of which crypto currencies each nft, seller and buyer has traded
nft_dict = defaultdict(lambda:set())
seller_dict = defaultdict(lambda:set())
buyer_dict = defaultdict(lambda:set())

variables = zip(df["Unique_id_collection"], df["Seller_address"],df["Buyer_address"],df["Crypto"])

i = 0
for nft, seller, buyer, crypto in variables:
    if i % 1000000 == 0:
        print(i)
    i += 1
    nft_dict[nft].add(crypto)
    seller_dict[seller].add(crypto)
    buyer_dict[buyer].add(crypto)

nfts = np.unique(df["Unique_id_collection"])
sellers = np.unique(df["Seller_address"])
buyers = np.unique(df["Buyer_address"])

# get edges of connected cryptos
crypto_dict = defaultdict(lambda:set())
for nft in nfts:
    for crypto1 in nft_dict[nft]:
        for crypto2 in nft_dict[nft]:
            if crypto1 != crypto2:
                crypto_dict[crypto1].add(crypto2)

print("NFT edges done!")

for seller in sellers:
    for crypto1 in seller_dict[seller]:
        for crypto2 in seller_dict[seller]:
            if crypto1 != crypto2:
                crypto_dict[crypto1].add(crypto2)
            
print("Seller edges done!")

for buyer in buyers:
    for crypto1 in buyer_dict[buyer]:
        for crypto2 in buyer_dict[buyer]:
            if crypto1 != crypto2:
                crypto_dict[crypto1].add(crypto2)

print("Buyer edges done!")

# check if a person has sold using one cryptocurrency and sold using another
for seller in sellers:
    for crypto1 in seller_dict[seller]:
        for crypto2 in buyer_dict[seller]:
            if crypto1 != crypto2:
                crypto_dict[crypto1].add(crypto2)

print("Seller to buyer edges done!")

cryptos = np.unique(df["Crypto"])
visited = defaultdict(lambda:False) 
cluster = defaultdict(lambda:-1)
cnum = 0

for crypto in cryptos:
    if visited[crypto]:
        continue

    queue = deque([crypto])

    while queue:
        vertex = queue.popleft()
        cluster[vertex] = cnum
        for neighbour in crypto_dict[vertex]:
            if not visited[neighbour]:
                visited[neighbour] = True
                queue.append(neighbour)
    
    cnum += 1

clusters = defaultdict(lambda:[])

for crypto in cryptos:
    clusters[cluster[crypto]].append(crypto)

crypto_count = defaultdict(lambda: 0)
cluster_count = defaultdict(lambda: 0)

for crypto in df["Crypto"]:
    crypto_count[crypto] += 1
    cluster_count[cluster[crypto]] += 1

for c in range(cnum):
    print(clusters[c], cluster_count[c], len(clusters[c]))
    for crypto, count in reversed(sorted(crypto_count.items(),key=lambda x: x[1])):
        if crypto in clusters[c]:
            print(crypto + ": " + str(count))
    



