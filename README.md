# NFT Project - DTU 02466 Project Work 
### By Kasper Helverskov Petersen (s203294), Mads Vibe Ringsted (s204144), Marcus Roberto Nielsen (s204126) and Roneet Vijay Nagale (s204091)

Follow these steps to reproduce the results of this project: 

1. Acquire the full dataset from: osf.io/xnj3k/?view_only=319a53cf1bf542bbbe538aba37916537
2. Get partition of the data set for a specific blockchain by running the python file: create_blockchain_dataset.py
3. Create monthly data:
  - Find the month(s) of interest and run the code: create_monthly_data.py
5. Set model variables in: main.py:
  - Choose blockchain
  - Choose number of latent dimensions, the learning rate and number of epochs
  - Choose the month(s) of interest
6. Run the code: main.py
