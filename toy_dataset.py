import numpy as np
import pandas as pd

Traders = []
Nfts = np.array(['NFT'+ str(i) for i in range(1,1001)])


class Trader():
    def __init__(self, Trader_id,nfts):
        self.Trader_id = Trader_id
        self.nfts = nfts
        
    
    def Trade(self,T2):
        
        index = np.random.randint(0,len(self.nfts))
        nft_trade = self.nfts[index]
        
        self.nfts = np.delete(self.nfts,index)
        
        T2.nfts = np.append(T2.nfts,nft_trade)
             
        return nft_trade
        


def create_traders(trader_id, nfts):
    
    n = np.random.randint(0,20)
    nft_index = np.random.randint(0,len(nfts),n)
    
    trader_nfts = nfts[nft_index]
    
    T = Trader(trader_id,trader_nfts)
    
    nfts = np.delete(nfts,nft_index)
    
    return T, nfts


for i in range(100):
    trader_id = 'Trader' + str(i+1)
    
    T, nfts = create_traders(trader_id, Nfts)
    
    Traders.append(T)
    Nfts = nfts
    
    
Sellers = []
traded_nfts = []
buyers = []

while len(Sellers) < 100:
    T1 = np.random.randint(0,len(Traders))
    T2 = np.random.randint(0,len(Traders))
    while T2 == T1:
        T2 = np.random.randint(0,len(Traders))
    
    T1 = Traders[T1]
    T2 = Traders[T2]
    
    if len(T1.nfts) > 0:
        NFT = T1.Trade(T2)
    
        Sellers.append(T1.Trader_id)
        traded_nfts.append(NFT)
        buyers.append(T2.Trader_id)
    
df=pd.DataFrame({'Sellers': Sellers, 'NFTs':traded_nfts, 'Buyers':buyers})
    
print(df)

print(len(set(Sellers)))
print(len(set(buyers)))
#print(T2.nfts)