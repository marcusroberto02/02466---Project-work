# HUSK CREDITS!!!

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

from torch_sparse import spspmm

CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
 
    
# i corresponds to NFT's
# j corresponds to traders

class LDM_TRI(nn.Module):
    def __init__(self,sparse_i,sparse_j,sparse_k,sparse_w,nft_size,seller_size,buyer_size,latent_dim,nft_sample_size):
        super(LDM_TRI, self).__init__()
        # input sizes
        self.nft_size = nft_size
        self.seller_size = seller_size
        self.buyer_size = buyer_size
        # dimension D of embeddings
        self.latent_dim=latent_dim

        # scaling term
        self.scaling=0
        
        # create indices to index properly the receiver and senders variable
        self.sparse_i_idx = sparse_i
        self.sparse_j_idx = sparse_j
        self.sparse_k_idx = sparse_k
        self.weights = sparse_w

        self.Softmax=nn.Softmax(1)

        # for sampling
        self.sampling_weights=torch.ones(self.nft_size,device=device)
        self.nft_sample_size=nft_sample_size

        # PARAMETERS
        # nft embeddings
        self.latent_l=nn.Parameter(torch.randn(self.nft_size,latent_dim,device=device))
        # seller embeddings
        self.latent_r=nn.Parameter(torch.randn(self.seller_size,latent_dim,device=device))
        # buyer embeddings
        self.latent_u=nn.Parameter(torch.randn(self.buyer_size,latent_dim,device=device))
        # define bias terms
        self.rho=nn.Parameter(torch.randn(self.nft_size,device=device))
        self.nu=nn.Parameter(torch.randn(self.seller_size,device=device))
        self.tau=nn.Parameter(torch.randn(self.buyer_size,device=device))

    #introducing the Poisson log-likelihood  
    def LSM_likelihood_bias(self,epoch):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        
        '''
        self.epoch=epoch

        # mini batch over network
        sample_i, sample_j, sample_k, sample_weights = self.sample_network()
        unique_i = torch.unique(sample_i)
        unique_j = torch.unique(sample_j)
        unique_k = torch.unique(sample_k)

        # distance matrix for seller and NFT embeddings
        # dimension is S x N
        # NB! Important that the nft size is the second dimension
        d_rl = torch.cdist(self.latent_r[unique_j]+1e-06,self.latent_l[unique_i],p=2)+1e-06
        
        # distance matrix for buyer and NFT embeddings
        # dimension is B x N
        d_ul = torch.cdist(self.latent_u[unique_k]+1e-06,self.latent_l[unique_i],p=2)+1e-06
        
        # calculate seller and nft non link part
        # dimension is S x N
        non_link_rl = torch.exp(self.nu[unique_j].unsqueeze(1)-d_rl)

        # calculate seller and nft non link part
        # dimension is B x N
        non_link_ul = torch.exp(self.rho[unique_i]+self.tau[unique_k].unsqueeze(1)-d_ul)

        # total non link matrix
        # dimension is S x B x N
        # S x 1 x N * B x N = S x B x N
        total_non_link = non_link_rl.unsqueeze(1) * non_link_ul

        # sum over values to get z_pdist1
        z_pdist1 = torch.sum(total_non_link)

        # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
        zqdist_lr = -((((self.latent_l[sample_i]-self.latent_r[sample_j]+1e-06)**2).sum(-1))**0.5)
        zqdist_lu = -((((self.latent_l[sample_i]-self.latent_u[sample_k]+1e-06)**2).sum(-1))**0.5)
        sum_bias = self.rho[sample_i]+self.nu[sample_j]+self.tau[sample_k]
        z_pdist2=(sample_weights*(sum_bias+zqdist_lr+zqdist_lu)).sum()

        # Total Log-likelihood
        log_likelihood_sparse=z_pdist2-z_pdist1
        
        return log_likelihood_sparse

    def sample_network(self):
        # sample nfts
        sample_idx = torch.multinomial(self.sampling_weights,self.nft_sample_size,replacement=False)

        # extract edges
        edge_translator = torch.isin(self.sparse_i_idx,sample_idx)

        # get edges
        sample_i_idx = self.sparse_i_idx[edge_translator]
        sample_j_idx = self.sparse_j_idx[edge_translator]
        sample_k_idx = self.sparse_k_idx[edge_translator]
        sample_weights = self.weights[edge_translator]

        return sample_i_idx, sample_j_idx, sample_k_idx, sample_weights


    
 
#################################################################
'''
MAIN: Training LDM

'''
#################################################################       
plt.style.use('ggplot')
torch.autograd.set_detect_anomaly(True)
# Number of latent communities
latent_dims=[2]
# Total model iterations
total_epochs=1
# Initial iterations for scaling the random effects
scaling_it=2000
# Dataset Name
dataset=r'C:\Users\marcu\Google Drev\DTU\02466(fagprojekt)\02466---Project-work\data\2017_11\train\sparse_tri'
# Learning rates
lrs=[0.1]
# Total independent runs of the model
total_runs=1
# path to results folder
results_path = r"C:\Users\marcu\Google Drev\DTU\02466(fagprojekt)\02466---Project-work\data\2017_11\results\tri"

for run in range(1,total_runs+1):
    for latent_dim in latent_dims:
       
        print('Latent Communities: ',latent_dim)

        for lr in lrs:
            print('Learning rate: ',lr)
            # EDGELIST
            # nft input
            sparse_i=torch.from_numpy(np.loadtxt(dataset+'/sparse_i.txt')).long().to(device)
            # seller input
            sparse_j=torch.from_numpy(np.loadtxt(dataset+'/sparse_j.txt')).long().to(device)
            # buyer input
            sparse_k=torch.from_numpy(np.loadtxt(dataset+'/sparse_k.txt')).long().to(device)
            # weight input
            sparse_w=torch.from_numpy(np.loadtxt(dataset+'/sparse_w.txt')).long().to(device)
            # network size
            N=int(sparse_i.max()+1)
            S=int(sparse_j.max()+1)
            B=int(sparse_k.max()+1)
            
            # initialize model
            model = LDM_TRI(sparse_i=sparse_i,sparse_j=sparse_j,sparse_k=sparse_k,sparse_w=sparse_w,
                        nft_size=N,seller_size=S,buyer_size=B,latent_dim=latent_dim,nft_sample_size=1000).to(device)         

            optimizer = optim.Adam(model.parameters(), lr=lr)
    
            print('Dataset: ',dataset)
            print('##################################################################')

            losses=[]
            ROC=[]
            PR=[]
            # model.scaling=1
    
    
            for epoch in range(total_epochs):
             
                if epoch==scaling_it:
                    model.scaling=0
                
                loss=-model.LSM_likelihood_bias(epoch=epoch)
                losses.append(loss.item())
                
                optimizer.zero_grad() # clear the gradients.   
                loss.backward() # backpropagate
                optimizer.step() # update the weights

            # save bias terms
            torch.save(model.rho.detach().cpu(),results_path + "/nft_biases")
            torch.save(model.nu.detach().cpu(),results_path + "/seller_biases")
            torch.save(model.tau.detach().cpu(),results_path + "/buyer_biases")

            # save embeddings
            torch.save(model.latent_l.detach().cpu(),results_path + "/nft_embeddings")
            torch.save(model.latent_r.detach().cpu(),results_path + "/seller_embeddings")
            torch.save(model.latent_u.detach().cpu(),results_path + "/buyer_embeddings")



            # plot in latent space
            # nft
            """
            l = model.latent_l.detach().numpy()
            lx = [el[0] for el in l]
            ly = [el[1] for el in l]
            plt.scatter(lx,ly,s=10)
            # seller
            r = model.latent_r.detach().numpy()
            rx = [el[0] for el in r]
            ry = [el[1] for el in r]
            plt.scatter(rx,ry,s=10)
            # buyer
            u = model.latent_u.detach().numpy()
            ux = [el[0] for el in u]
            uy = [el[1] for el in u]
            plt.scatter(ux,uy,s=10)
            plt.show()
            """

