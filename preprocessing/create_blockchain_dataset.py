import pandas as pd

# name of the data set
name = "data_ETH"
# name = "data_WAX"
#cryptos = ['WAX']
# choose which cryptos should be included
cryptos = ['0xBTC', '1MT', 'ABST', 'AMPL', 'ANRX', 'ARCONA', 'ART', 'ASLT',
            'ATRI', 'AVRT', 'B0T', 'BAEPAY', 'BASED', 'BAT', 'BLVD', 'BON',
            'BONDLY', 'BONES', 'BOOB', 'BPC', 'BUDGET', 'BUDZ', 'BZN', 'CAMEL',
            'CGG', 'CHERRY', 'CHONK', 'COIN', 'COVAL', 'CPT', 'CRED', 'CUBE',
            'CURIO', 'DAI', 'DAPPT', 'DDIM', 'DENA', 'DGX', 'DHC', 'DOOM',
            'DUST', 'EBB', 'ECAT', 'ECTO', 'EGGS', 'ELAND', 'ELET', 'EMONT',
            'ENJ', 'EPIC', 'ETH', 'FIRST', 'FRFY', 'FTHR', 'FTM', 'FUD', 'GALA',
            'GCASH', 'GEM', 'GMX', 'GOKU', 'GOOSE', 'GOU', 'GPL', 'GUSD', 'HIVED',
            'HOUR', 'HUE', 'IMP', 'INK', 'JBG', 'JGN', 'KAP', 'KEK', 'KEK-DEP',
            'KING', 'KIWI', 'KLTR', 'KOI', 'LAR', 'LESS', 'LINK', 'LIT', 'LOAD',
            'MANA', 'MATIC', 'MBC', 'MCX', 'MEME', 'MGDv2', 'MKR', 'MM', 'MNFT',
            'MORK', 'MX', 'NDR', 'NEWS', 'NUGS', 'NVT', 'OLDWAXIE', 'OLDWZX',
            'PGU', 'PIXEL', 'PIXIE', 'PLAY', 'PMON', 'PPDEX', 'PRIME', 'PXART',
            'PYRO', 'RAINBOW', 'RARE', 'RARI', 'RCC', 'RCDY', 'REVV', 'RLY',
            'RODZ', 'ROPE', 'ROT', 'RUGZ', 'SAI', 'SAL', 'SAND', 'SKULL', 'SLP',
            'SMTS', 'SURF', 'SWAG', 'TATR', 'THREE', 'TRIP', 'TRISM', 'TRSH',
            'UNI', 'USDC', 'VEGETA', 'VI', 'VIDEO', 'VIDT', 'VSF', 'WAIF', 'WAXIE',
            'WBTC', 'WCK', 'WETH', 'WGM', 'WIPC', 'WMC', 'XMON', 'YUMI', 'ZIOT',
            'ZURU', 'ZUT', 'eFAME']

dataset = pd.DataFrame()
path = '../data/'
dataset_name = "Data_API.csv"
i = 0
for chunk in pd.read_csv(path + dataset_name, chunksize=10000, parse_dates=[18]):
    temp = chunk[chunk['Crypto'].isin(cryptos)]
    
    dataset = pd.concat([dataset,temp],axis = 0)
    if i % 100 == 0:
        print(i)
    i += 1
    
dataset.to_csv(path+name+".csv", index = None)
