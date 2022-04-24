# HUSK CREDITS!!!

# Import all the packages
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
    def __init__(self,sparse_i,sparse_j,sparse_k,sparse_w,nft_size,seller_size,buyer_size,latent_dim):
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
            z_pdist2=(self.weights*(self.gamma[self.sparse_i_idx]+self.delta[self.sparse_j_idx]+zqdist)).sum()
    
            log_likelihood_sparse=z_pdist2-z_pdist1
                            
        else:
            # distance matrix for seller and NFT embeddings
            # dimension is S x N
            # NB! Important that the nft size is the second dimension
            d_rl = torch.cdist(self.latent_r+1e-06,self.latent_l,p=2)+1e-06
            
            # distance matrix for buyer and NFT embeddings
            # dimension is B x N
            d_ul = torch.cdist(self.latent_u+1e-06,self.latent_l,p=2)+1e-06
            
            # calculate seller and nft non link part
            # dimension is S x N
            non_link_rl = torch.exp(self.nu.unsqueeze(1)-d_rl)

            # calculate seller and nft non link part
            # dimension is B x N
            non_link_ul = torch.exp(self.rho+self.tau.unsqueeze(1)-d_ul)

            # total non link matrix
            # dimension is S x B x N
            # S x 1 x N * B x N = S x B x N
            total_non_link = non_link_rl.unsqueeze(1) * non_link_ul

            # sum over values to get z_pdist1
            z_pdist1 = torch.sum(total_non_link)

            # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
            zqdist_lr = -((((self.latent_l[self.sparse_i_idx]-self.latent_r[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5)
            zqdist_lu = -((((self.latent_l[self.sparse_i_idx]-self.latent_u[self.sparse_k_idx]+1e-06)**2).sum(-1))**0.5)
            sum_bias = self.rho[self.sparse_i_idx]+self.nu[self.sparse_j_idx]+self.tau[self.sparse_k_idx]
            z_pdist2=(self.weights*(sum_bias+zqdist_lr+zqdist_lu)).sum()

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
dataset='/zhome/45/e/155478/Desktop/02466---Project-work/data/sparse_tri'
# Learning rates
lrs=[0.1]
# Total independent runs of the model
total_runs=1
# path to results folder
results_path = "/zhome/45/e/155478/Desktop/02466---Project-work/results"

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
<<<<<<< HEAD
            """
        
=======

#################################################################
'''
Node classification

'''
#################################################################
df = pd.DataFrame[model.latent_l.detach().numpy()]
df['Category'] = np.loadtxt(dataset+'/sparse_v.txt')

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(df, df['Category'], test_size=0.2, random_state=42)

#Multinomial logistic regression
logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
logreg.fit(X_train, y_train)

#Metrics
print('Number of miss-classifications for Multinomial regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test), len(y_test)))
print('Accuracy for Multinomial regression:\n\t {0}'.format(logreg.score(X_test, y_test)))
print('Confusion matrix for Multinomial regression:\n\t {0}'.format(confusion_matrix(y_test, logreg.predict(X_test))))
print('ROC AUC for Multinomial regression:\n\t {0}'.format(roc_auc_score(y_test, logreg.predict(X_test))))
print('Precision for Multinomial regression:\n\t {0}'.format(precision_score(y_test, logreg.predict(X_test))))
print('Recall for Multinomial regression:\n\t {0}'.format(recall_score(y_test, logreg.predict(X_test))))



#KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#Metrics
print('Number of miss-classifications for KNN:\n\t {0} out of {1}'.format(np.sum(knn.predict(X_test)!=y_test), len(y_test)))
print('Accuracy for KNN:\n\t {0}'.format(knn.score(X_test, y_test)))
print('Confusion matrix for KNN:\n\t {0}'.format(confusion_matrix(y_test, knn.predict(X_test))))
print('ROC AUC for KNN:\n\t {0}'.format(roc_auc_score(y_test, knn.predict(X_test))))
print('Precision for KNN:\n\t {0}'.format(precision_score(y_test, knn.predict(X_test))))
print('Recall for KNN:\n\t {0}'.format(recall_score(y_test, knn.predict(X_test))))


#################################################################
'''
Clustering

'''
#################################################################

kmeans = KMeans(n_clusters=3, random_state=0).fit(df)

label = kmeans.labels_

u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], s=10, c=np.random.rand(3,), label='cluster %d' % i)
plt.legend()
plt.show()








>>>>>>> 2b2ecd52e840b7b0fdcf13925ea48eaa10de762c


