# HUSK CREDITS!!!

# Import all the packages
import pandas
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics

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

class LDM(nn.Module):
    def __init__(self,sparse_i,sparse_j,sparse_k,sparse_w,nft_size,seller_size,buyer_size,latent_dim):
        super(LDM, self).__init__()
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

        if self.scaling:
            
            # Optimize only the random effects initially so proper scaling i.e. the rate now is only l_ij=exp(a_i+b_j)
            # gamma matrix
            mat_gamma=torch.exp(torch.zeros(self.nft_size,self.nft_size)+1e-06)
            mat_delta=torch.exp(torch.zeros(self.nft_size,self.nft_size)+1e-06)

            #exp(gamma)*exp(delta)=exp(gamma+delta)

            # Non-link N^2 likelihood term, i.e. \sum_ij lambda_ij
            z_pdist1=torch.mm(torch.exp(self.gamma.unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma).unsqueeze(-1))))
            # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
            zqdist = -((((self.latent_z[self.sparse_i_idx]-self.latent_q[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5)
            z_pdist2=(self.weights*(self.gamma[self.sparse_i_idx]+self.delta[self.sparse_j_idx]-zqdist)).sum()
    
            log_likelihood_sparse=z_pdist2-z_pdist1
                            
        else:
            # exp(||l_i - r_j||)
            mat_lr=torch.exp(-(torch.cdist(self.latent_l+1e-06,self.latent_r,p=2)+1e-06))
            # exp(||l_i - u_k||)
            mat_lu=torch.exp(-(torch.cdist(self.latent_l+1e-06,self.latent_u,p=2)+1e-06))
            
            # get term sum(exp(rho_i+nu_j+tau_k-||l_i - r_j||-||l_i - u_k||))
            # rho_i: size N
            # nu_j: size S
            # tau_k: size B
            # mat_lr: size N x S
            # mat_lu: size N x B
            N = self.nft_size
            S = self.seller_size
            B = self.buyer_size
            z_pdist1 = 0
            for i in range(N):
                if i % 100 == 0:
                    print(i,N,S,B)
                for j in range(S):
                    for k in range(B):
                        # exp(rho_i+nu_j+tau_k-||l_i-r_k||-||l_i-u_k||)
                        z_pdist1 += torch.exp(self.rho[i]+self.nu[j]+self.tau[k]-mat_lr[i][j]-mat_lu[i][k])
            
            # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
            zqdist_lr = -((((self.latent_l[self.sparse_i_idx]-self.latent_r[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5)
            zqdist_lu = -((((self.latent_l[self.sparse_i_idx]-self.latent_u[self.sparse_k_idx]+1e-06)**2).sum(-1))**0.5)
            sum_bias = self.rho[self.sparse_i_idx]+self.nu[self.sparse_j_idx]+self.tau[self.sparse_k_idx]
            z_pdist2=(self.weights*(sum_bias-zqdist_lr-zqdist_lu)).sum()

            # Total Log-likelihood
            log_likelihood_sparse=z_pdist2-z_pdist1
    
    
        return log_likelihood_sparse
    
    
    
 
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
total_epochs=2
# Initial iterations for scaling the random effects
scaling_it=2000
# Dataset Name
dataset='data/sparse_matrix_tri_toy.csv'
# Learning rates
lrs=[0.1]
# Total independent runs of the model
total_runs=1

for run in range(1,total_runs+1):
    for latent_dim in latent_dims:
       
        print('Latent Communities: ',latent_dim)

        for lr in lrs:
            print('Learning rate: ',lr)

            sparse_data = pandas.read_csv(dataset)
            # EDGELIST
            # nft input
            sparse_i=torch.from_numpy(sparse_data["NFT_idx"].to_numpy()).long().to(device)
            # seller input
            sparse_j=torch.from_numpy(sparse_data["Seller_idx"].to_numpy()).long().to(device)
            # buyer input
            sparse_k=torch.from_numpy(sparse_data["Buyer_idx"].to_numpy()).long().to(device)
            # weight input
            sparse_w=torch.from_numpy(sparse_data["count"].to_numpy()).long().to(device)
            # network size
            N=int(sparse_i.max()+1)
            S=int(sparse_j.max()+1)
            B=int(sparse_k.max()+1)
            
            # initialize model
            model = LDM(sparse_i=sparse_i,sparse_j=sparse_j,sparse_k=sparse_k,sparse_w=sparse_w,
                        nft_size=N,seller_size=S,buyer_size=B,latent_dim=latent_dim).to(device)         

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
            
            # plot in latent space
            # nft
            l = model.latent_l.detach().numpy()
            lx = [el[0] for el in l]
            ly = [el[1] for el in l]
            plt.scatter(lx,ly)
            # seller
            r = model.latent_r.detach().numpy()
            rx = [el[0] for el in r]
            ry = [el[1] for el in r]
            plt.scatter(rx,ry)
            # buyer
            u = model.latent_u.detach().numpy()
            ux = [el[0] for el in u]
            uy = [el[1] for el in u]
            plt.scatter(ux,uy)
            plt.show()
        


