
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:05:57 2021

@author: nnak
"""

# Import all the packages
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
 
    

class LDM(nn.Module):
    def __init__(self,sparse_i,sparse_j, input_size,latent_dim,sample_size,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,scaling=None,norm_values=False):
        super(LDM, self).__init__()
        self.input_size=input_size
       
        self.latent_dim=latent_dim
        
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j
        
        self.sampling_weights=torch.ones(self.input_size,device=device)
        self.sample_size=sample_size

        self.norm_values=norm_values
       
        self.non_sparse_i_idx_removed=non_sparse_i
     
        self.non_sparse_j_idx_removed=non_sparse_j
           
        self.sparse_i_idx_removed=sparse_i_rem
        self.sparse_j_idx_removed=sparse_j_rem
        
        # total sample of missing dyads with i<j
        self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
        self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))

        self.Softmax=nn.Softmax(1)
        
        # PARAMETERS
        self.latent_z=nn.Parameter(torch.randn(self.input_size,latent_dim,device=device))
        # define w for product embeddings
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device))
        self.delta=nn.Parameter(torch.randn(self.input_size))
        # define delta for product biases


    

    
    def sample_network(self):
        '''
        Network Sampling procecdure used for large scale networks
        '''
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx.unsqueeze(0),sample_idx.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges,torch.ones(edges.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
        
        # edge row position
        sparse_i_sample=indexC[0,:]
        # edge column position
        sparse_j_sample=indexC[1,:]
     
        
        return sample_idx,sparse_i_sample,sparse_j_sample
        
    #
    
    #introducing the Poisson log-likelihood  
    def LSM_likelihood_bias(self,epoch):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        
        '''
        self.epoch=epoch
        if self.norm_values:
            raise Exception("Not implented yet") 
                
        else:

            if self.scaling:
                
                # Optimize only the random effects initially so proper scaling i.e. the rate now is only l_ij=exp(a_i+b_j)
                
                mat=torch.exp(torch.zeros(self.input_size,self.input_size)+1e-06)
                # Non-link N^2 likelihood term, i.e. \sum_ij lambda_ij
                z_pdist1=0.5*torch.mm(torch.exp(self.gamma.unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma).unsqueeze(-1))))
                # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
                z_pdist2=(self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]).sum()
        
                log_likelihood_sparse=z_pdist2-z_pdist1
                              
            else:
                
                mat=torch.exp(-(torch.cdist(self.latent_z+1e-06,self.latent_z,p=2)+1e-06))
                # Non-link N^2 likelihood term, i.e. \sum_ij lambda_ij
                # for the bipartite case the diagonal part should be removed
                # as well as the 1/2 term
                z_pdist1=0.5*torch.mm(torch.exp(self.gamma.unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma).unsqueeze(-1))))
                # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
                z_pdist2=(-((((self.latent_z[self.sparse_i_idx]-self.latent_z[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5)+self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]).sum()

                # Total Log-likelihood
                log_likelihood_sparse=z_pdist2-z_pdist1
        
        
        return log_likelihood_sparse
    
    
    
    
   
    
    def link_prediction(self):

        with torch.no_grad():
            z_pdist_miss=((((self.latent_z[self.removed_i]-self.latent_z[self.removed_j]+1e-06)**2).sum(-1))**0.5)
            if self.scaling:
                logit_u_miss=self.gamma[self.removed_i]+self.gamma[self.removed_j]

            else:
                logit_u_miss=-z_pdist_miss+self.gamma[self.removed_i]+self.gamma[self.removed_j]
            rates=torch.exp(logit_u_miss)
            self.rates=rates

            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    
    
    
    
 
#################################################################
'''
MAIN: Training LDM

'''
#################################################################       
plt.style.use('ggplot')
torch.autograd.set_detect_anomaly(True)
# Number of latent communities
latent_dims=[2,3,4,8,16]
# Total model iterations
total_epochs=10000
# Initial iterations for scaling the random effects
scaling_it=2000
# Dataset Name
dataset='grqc'
# Learning rates
lrs=[0.1]
# Total independent runs of the model
total_runs=5

for run in range(1,total_runs+1):
    for latent_dim in latent_dims:
       
        print('Latent Communities: ',latent_dim)

        for lr in lrs:
            print('Learning rate: ',lr)

            # POSITONS of missing LINK dyads and Negative sample dyads used for link prediction
            # i.e. pairs of i and j where we hide a link, meaning turning y_ij=1 to y_ij=0 and try to predict it back again
            # negative samples are actual pairs of y_ij=0 that we also try to predict to see how well we can order the rates of a connection
            
            # file denoting rows i of missing links, with i<j 
            sparse_i_rem=None
            # file denoting columns j of missing links, with i<j
            sparse_j_rem=None
            # file denoting negative sample rows i, with i<j
            non_sparse_i=None
            # file denoting negative sample columns, with i<j
            non_sparse_j=None

            sparse_data =
            # EDGELIST
            # input data, link rows i positions with i<j
            sparse_i=torch.from_numpy(np.loadtxt(dataset+'/sparse_i.txt')).long().to(device)
            # input data, link column positions with i<j
            sparse_j=torch.from_numpy(np.loadtxt(dataset+'/sparse_j.txt')).long().to(device)
            # network size
            N=int(sparse_j.max()+1)
            # sample size of blocks-> sample_size*(sample_size-1)/2 pairs
            sample_size=int(N)

            model = LDM(sparse_i=sparse_i,sparse_j=sparse_j,input_size=N,latent_dim=latent_dim,sample_size=sample_size,non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,norm_values=False).to(device)         

            optimizer = optim.Adam(model.parameters(), lr=lr)  
    
            print('Dataset: ',dataset)
            print('##################################################################')

            losses=[]
            ROC=[]
            PR=[]
            model.scaling=1
    
    
            for epoch in range(total_epochs):
             
                if epoch==scaling_it:
                    model.scaling=0
                
                loss=-model.LSM_likelihood_bias(epoch=epoch)/sample_size
                losses.append(loss.item())
                
         
             
                optimizer.zero_grad() # clear the gradients.   
                loss.backward() # backpropagate
                optimizer.step() # update the weights
                if epoch%100==0:
                    # AUC-ROC and PR-AUC
                    # Receiver operating characteristic-area under curve   AND precision recal-area under curve
                    roc,pr=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr
                    print('Epoch: ',epoch)
                    print('ROC:',roc)
                    print('PR:',pr)
                    ROC.append(roc)
                    PR.append(pr)
            # save bias/random-effect      
            torch.save(model.gamma.detach(),dataset+f'/{dataset}_K_{latent_dim}_lr_{lr}_gamma_{run}')
            # save latent embedding position
            torch.save(model.latent_z.detach(),dataset+f'/{dataset}_K_{latent_dim}_lr_{lr}_zeta_{run}')
            roc,pr=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr
            print('dim',latent_dim)
            print('Epoch: ',epoch)
            print('ROC:',roc)
            print('PR:',pr)
            ROC.append(roc)
            PR.append(pr)
    
            filename_roc=dataset+f"/ROC_K_{latent_dim}_lr_{lr}_{run}"+".txt"
            filename_pr=dataset+f"/PR_K_{latent_dim}_lr_{lr}_{run}"+".txt"
            # save performance statistics
            np.savetxt(filename_roc,(ROC),delimiter=' ')
            np.savetxt(filename_pr,(PR),delimiter=' ')
        


