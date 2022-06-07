
# Import all the packages
from dataclasses import replace
from enum import unique
from random import sample
import pandas
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ldm_bi import run_ldm_bi
from ldm_tri import run_ldm_tri

from torch_sparse import spspmm

CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Dataset Name
# for the hpc
#path = '/zhome/45/e/155478/Desktop/02466---Project-work/data'
#blockchain = '/ETH'
#date = '/2020-10'
#dataset = path + blockchain + date

# for testing on local pc
dataset = "./data/ETH/2020-10"

# model parameters
latent_dims = [1,2,4,5]
total_epochs= 1
n_test_batches = 5
lrs=[0.1]
# Total independent runs of the model
total_runs=1


run_ldm_bi(dataset, latent_dims,total_epochs,n_test_batches,lrs,total_runs)
run_ldm_tri(dataset, latent_dims,total_epochs,n_test_batches,lrs,total_runs)

 