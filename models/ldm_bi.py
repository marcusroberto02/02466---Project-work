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

class LDM_BI(nn.Module):
    def __init__(self,sparse_i,sparse_j,sparse_w,nft_size,trader_size,latent_dim,nft_sample_size,trader_sample_size):
        super(LDM_BI, self).__init__()
        # input sizes
        self.nft_size = nft_size
        self.trader_size = trader_size
        # dimension D of embeddings
        self.latent_dim=latent_dim
        
        # scaling term
        self.scaling=0
        
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx = sparse_i
        self.sparse_j_idx = sparse_j
        self.weights = sparse_w

        self.Softmax=nn.Softmax(1)
        
        #
        self.sampling_weights_nfts = torch.ones(self.nft_size,device = device)
        self.sampling_weights_traders=torch.ones(self.trader_size,device = device)
        self.nft_sample_size=nft_sample_size
        self.trader_sample_size =trader_sample_size


        # PARAMETERS
        # nft embeddings
        self.latent_z=nn.Parameter(torch.randn(self.nft_size,latent_dim,device=device))
        # trader embeddings
        self.latent_q=nn.Parameter(torch.randn(self.trader_size,latent_dim,device=device))
        # define bias terms
        self.gamma=nn.Parameter(torch.randn(self.nft_size,device=device))
        self.delta=nn.Parameter(torch.randn(self.trader_size,device=device))

    #introducing the Poisson log-likelihood  
    def LSM_likelihood_bias(self,epoch):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        
        '''
        self.epoch=epoch   

        self.sparse_i_idx, self.sparse_j_idx = self.sample_network()
                            
        # exp(||z_i - q_j||)
        mat=torch.exp(-(torch.cdist(self.latent_z+1e-06,self.latent_q,p=2)+1e-06))
        # Non-link N^2 likelihood term, i.e. \sum_ij lambda_ij
        # for the bipartite case the diagonal part should be removed
        # as well as the 1/2 term
        # exp(gamma)*exp(delta)*exp(mat)

        # spar

        z_pdist1=torch.mm(torch.mm(torch.exp(self.gamma.unsqueeze(0)),mat),torch.exp(self.delta.unsqueeze(-1)))
        # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
        zqdist = -((((self.latent_z[self.sparse_i_idx]-self.latent_q[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5)
        z_pdist2=(self.weights*(self.gamma[self.sparse_i_idx]+self.delta[self.sparse_j_idx]+zqdist)).sum()

        # Total Log-likelihood
        log_likelihood_sparse=z_pdist2-z_pdist1
    
    
        return log_likelihood_sparse

    def sample_network(self):
        '''
        Network Sampling procecdure used for large scale networks
        '''
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_idx_nfts=torch.multinomial(self.sampling_weights_nfts, self.nft_sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx_nfts.unsqueeze(0),sample_idx_nfts.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        indexC, valueC = spspmm(edges,torch.ones(edges.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.nft_size,self.nft_size,self.nft_size,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC = spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.nft_size,self.nft_size,self.nft_size,coalesced=True)
        
        # edge row position
        sparse_nfts_sample=indexC[0,:]
        # edge column position
        sparse_traders_sample=indexC[1,:]
     
        
        return sparse_nfts_sample,sparse_traders_sample
        
    #
    
    
    
 
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
dataset='data/sparse_bi'
# Learning rates
lrs=[0.1]
# Total independent runs of the model
total_runs=1

for run in range(1,total_runs+1):
    for latent_dim in latent_dims:
       
        print('Latent Communities: ',latent_dim)

        for lr in lrs:
            print('Learning rate: ',lr)

            # nft items
            sparse_i=torch.from_numpy(np.loadtxt(dataset+'/sparse_i.txt')).long().to(device)
            # traders
            sparse_j=torch.from_numpy(np.loadtxt(dataset+'/sparse_j.txt')).long().to(device)
            # weight items
            sparse_w=torch.from_numpy(np.loadtxt(dataset+'/sparse_w.txt')).long().to(device)
            # network size
            N=int(sparse_i.max()+1)
            T=int(sparse_j.max()+1)
            
            # initialize model
            model = LDM_BI(sparse_i=sparse_i,sparse_j=sparse_j,sparse_w=sparse_w,nft_size=N,trader_size=T,latent_dim=latent_dim,nft_sample_size=2,trader_sample_size=2).to(device)         

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
            z = model.latent_z.detach().numpy()
            zx = [el[0] for el in z]
            zy = [el[1] for el in z]
            plt.scatter(zx,zy,s=1)
            # trader
            q = model.latent_q.detach().numpy()
            qx = [el[0] for el in q]
            qy = [el[1] for el in q]
            plt.scatter(qx,qy,s=1)
            plt.show()
        