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
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#from torch_sparse import spspmm

CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
 
    
# i corresponds to NFT's
# j corresponds to traders
def run_ldm_tri(dataset=None, latent_dims = [2],total_epochs = 10000,n_test_batches = 5,lrs=[0.001],total_runs = 1,device=torch.device("cpu")):
    class LDM_TRI(nn.Module):
        def __init__(self,sparse_i,sparse_j,sparse_k,sparse_w,sparse_c,nft_size,seller_size,buyer_size,latent_dim,nft_sample_size,seller_matrix,buyer_matrix,test_batch_size=1000,sparse_i_test=None,sparse_j_test=None,sparse_k_test=None,sparse_w_test=None):
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

            # categories 
            self.sparse_c = sparse_c

            # matrices that stores the seller/buyer distribution across categories
            self.seller_matrix = seller_matrix
            self.buyer_matrix = buyer_matrix
            
            # used for the test set
            self.sparse_i_test = sparse_i_test
            self.sparse_j_test = sparse_j_test
            self.sparse_k_test = sparse_k_test
            self.sparse_w_test = sparse_w_test
            

            self.Softmax=nn.Softmax(1)

            # for sampling
            self.sampling_weights=torch.ones(self.nft_size,device=device)
            self.nft_sample_size=nft_sample_size

            # size of each test_batch
            self.test_size = self.sparse_i_test.size(0)
            self.sampling_weights_test = torch.ones(self.test_size,device=device)

            # size of each test batch
            self.test_batch_size = int(1/5 * len(self.sparse_i_test))

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
        
        def create_test_batch(self):
            sample_idx = torch.multinomial(self.sampling_weights_test,self.test_batch_size,replacement=False)

            # positive class
            test_i_pos = self.sparse_i_test[sample_idx]
            test_j_pos = self.sparse_j_test[sample_idx]
            test_k_pos = self.sparse_k_test[sample_idx]
            
            # negative class
            test_i_neg = torch.randint(0,self.nft_size,size=(self.test_batch_size,))
            test_j_neg = torch.randint(0,self.seller_size,size=(self.test_batch_size,))
            test_k_neg = torch.randint(0,self.buyer_size,size=(self.test_batch_size,))
            
            # print number of positive class equal to negative class
            print(len(torch.eq(test_i_pos,test_i_neg).masked_select(torch.eq(test_i_pos,test_i_neg)==True)))
            print(len(torch.eq(test_j_pos,test_j_neg).masked_select(torch.eq(test_j_pos,test_j_neg)==True)))
            print(len(torch.eq(test_k_pos,test_k_neg).masked_select(torch.eq(test_k_pos,test_k_neg)==True)))

            return test_i_pos, test_j_pos, test_k_pos, test_i_neg, test_j_neg, test_k_neg


        def link_prediction(self):
            # apply link prediction to the test set
            with torch.no_grad():
                test_i_pos,test_j_pos,test_k_pos,test_i_neg,test_j_neg,test_k_neg = self.create_test_batch()

                # get lambda values for negative edges
                rates_neg = torch.zeros(self.test_batch_size)
                for t in range(self.test_batch_size):
                    i,j,k = test_i_neg[t],test_j_neg[t],test_k_neg[t]
                    rho_i = self.rho[i]
                    nu_j = self.nu[j]
                    tau_k = self.tau[k]
                    lr_dist = ((self.latent_l[i]-self.latent_r[j]+1e-06)**2).sum(-1)**0.5
                    lu_dist = ((self.latent_l[i]-self.latent_u[k]+1e-06)**2).sum(-1)**0.5
                    rates_neg[t] = torch.exp(rho_i+nu_j+tau_k-lr_dist-lu_dist)

                # get lambda values for positive edges
                rates_pos = torch.zeros(self.test_batch_size)
                for t in range(self.test_batch_size):
                    i,j,k = test_i_pos[t],test_j_pos[t],test_k_pos[t]
                    rho_i = self.rho[i]
                    nu_j = self.nu[j]
                    tau_k = self.tau[k]
                    lr_dist = ((self.latent_l[i]-self.latent_r[j]+1e-06)**2).sum(-1)**0.5
                    lu_dist = ((self.latent_l[i]-self.latent_u[k]+1e-06)**2).sum(-1)**0.5
                    rates_pos[t] = torch.exp(rho_i+nu_j+tau_k-lr_dist-lu_dist)

                self.rates = torch.cat((rates_neg,rates_pos))
                self.target=torch.cat((torch.zeros(self.test_batch_size),torch.ones(self.test_batch_size)))
                precision, tpr, thresholds = metrics.precision_recall_curve(self.target.cpu().data.numpy(), self.rates.cpu().data.numpy())

                predictions = [self.rates.cpu().data.numpy() > t for t in thresholds]

                max_accuracy = max([metrics.accuracy_score(self.target.cpu().data.numpy(),p) for p in predictions])
                
            return metrics.roc_auc_score(self.target.cpu().data.numpy(),self.rates.cpu().data.numpy()),metrics.auc(tpr,precision),max_accuracy
        
        def baseline_model(self):
            test_i_pos,test_j_pos,test_k_pos,test_i_neg,test_j_neg,test_k_neg = self.create_test_batch()

            test_c_pos = [self.sparse_c[nft] for nft in test_i_pos]
            test_c_neg = [self.sparse_c[nft] for nft in test_i_neg]

            predictions = []
            target = [1] * self.test_batch_size + [0] * self.test_batch_size
            for t in range(self.test_batch_size):
                j,k,c = test_j_pos[t].item(),test_k_pos[t].item(),test_c_pos[t]
                

                if seller_matrix[c][j] > 0 and buyer_matrix[c][k] > 0:
                    predictions.append(1)
                else:
                    predictions.append(0)

            for t in range(self.test_batch_size):
                j,k,c = test_j_neg[t].item(),test_k_neg[t].item(),test_c_neg[t]


                if seller_matrix[c][j] > 0 and buyer_matrix[c][k] >0:
                    predictions.append(1)
                else:
                    predictions.append(0)
                
                
            accuracy = accuracy_score(target, predictions)

            return accuracy

            



        
    
    #################################################################
    '''
    MAIN: Training LDM

    '''
    #################################################################       
    plt.style.use('ggplot')
    torch.autograd.set_detect_anomaly(True)
    # Number of latent communities
    latent_dims=latent_dims
    # Total model iterations
    total_epochs= total_epochs
    # number of test batches for error bars
    n_test_batches = n_test_batches

    dataset = dataset

    # Learning rates
    lrs=lrs
    # Total independent runs of the model
    total_runs= total_runs
    
    for run in range(1,total_runs+1):
        for latent_dim in latent_dims:
            # results path
            results_path = dataset + "/tri/results/D" + str(latent_dim)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            print('Latent Communities: ',latent_dim)

            for lr in lrs:
                print('Learning rate: ',lr)
                # EDGELIST
                # TRAIN SET
                trainset = dataset + "/tri/train"
                # nft input
                sparse_i=torch.from_numpy(np.loadtxt(trainset+'/sparse_i.txt')).long().to(device)
                # seller input
                sparse_j=torch.from_numpy(np.loadtxt(trainset+'/sparse_j.txt')).long().to(device)
                # buyer input
                sparse_k=torch.from_numpy(np.loadtxt(trainset+'/sparse_k.txt')).long().to(device)
                # weight input
                sparse_w=torch.from_numpy(np.loadtxt(trainset+'/sparse_w.txt')).long().to(device)

                # TEST SET
                testset = dataset + "/tri/test"
                # nft items
                sparse_i_test=torch.from_numpy(np.loadtxt(testset+'/sparse_i.txt')).long().to(device)
                # sellers
                sparse_j_test=torch.from_numpy(np.loadtxt(testset+'/sparse_j.txt')).long().to(device)
                # buyers
                sparse_k_test=torch.from_numpy(np.loadtxt(testset+'/sparse_k.txt')).long().to(device)
                # weight items
                sparse_w_test=torch.from_numpy(np.loadtxt(testset+'/sparse_w.txt')).long().to(device)

                # network size
                N=int(len(sparse_i.unique()))
                S=int(len(sparse_j.unique()))
                B=int(len(sparse_k.unique()))

                # categories for each nft
                sparse_c = np.loadtxt(dataset + "/tri/sparse_c.txt",dtype='str')

                # seller buyer category matrices
                seller_matrix = pd.read_csv(dataset+"/tri/train/sellercategories.csv")
                buyer_matrix = pd.read_csv(dataset+"/tri/train/buyercategories.csv")

                # initialize model
                model = LDM_TRI(sparse_i=sparse_i,sparse_j=sparse_j,sparse_k=sparse_k,sparse_w=sparse_w,sparse_c=sparse_c,
                                nft_size=N,seller_size=S,buyer_size=B,latent_dim=latent_dim,nft_sample_size=1000,
                                sparse_i_test=sparse_i_test,sparse_j_test=sparse_j_test,
                                sparse_k_test=sparse_k_test,sparse_w_test=sparse_w_test,
                                seller_matrix = seller_matrix, buyer_matrix = buyer_matrix).to(device)         

                optimizer = optim.Adam(model.parameters(), lr=lr)
        
                print('Dataset: ',dataset)
                print('##################################################################')

                losses=[]
                ROC_train=[]
                PR_train=[]
                max_accuracy_train=[]
                baseline_accuracy_train=[]
                # model.scaling=1
        
                for epoch in range(total_epochs):
                    
                    loss=-model.LSM_likelihood_bias(epoch=epoch)
                    losses.append(loss.item())
                    
                    optimizer.zero_grad() # clear the gradients.   
                    loss.backward() # backpropagate
                    optimizer.step() # update the weights

                    if epoch%100==0:
                        # AUC-ROC and PR-AUC
                        # Receiver operating characteristic-area under curve   AND precision recal-area under curve
                        # and max accuracy for the best threshold
                        roc,pr,max_accuracy=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr, max accuracy
                        #roc, pr = 0,0
                        #print('Epoch: ',epoch)
                        #print('ROC:',roc)
                        #print('PR:',pr)
                        ROC_train.append([epoch,roc])
                        PR_train.append([epoch,pr])
                        max_accuracy_train.append([epoch,max_accuracy])
                        baseline_accuracy = model.baseline_model()
                        baseline_accuracy_train.append([epoch,baseline_accuracy])

                # save bias terms
                torch.save(model.rho.detach().cpu(),results_path + "/nft_biases")
                torch.save(model.nu.detach().cpu(),results_path + "/seller_biases")
                torch.save(model.tau.detach().cpu(),results_path + "/buyer_biases")

                # save embeddings
                torch.save(model.latent_l.detach().cpu(),results_path + "/nft_embeddings")
                torch.save(model.latent_r.detach().cpu(),results_path + "/seller_embeddings")
                torch.save(model.latent_u.detach().cpu(),results_path + "/buyer_embeddings")

                # run predictions on multiple test batches to obtain error bars
                ROC_final = []
                PR_final = []
                max_accuracy_final = [] 
                baseline_accuracy_final = [] 
                for i in range(n_test_batches):
                    roc,pr,max_accuracy=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr and max accuracy
                    ROC_final.append(roc)
                    PR_final.append(pr)
                    max_accuracy_final.append(max_accuracy)
                    baseline_accuracy = model.baseline_model()
                    baseline_accuracy_final.append(baseline_accuracy)
                ROC_avg = np.mean(ROC_final)
                ROC_std = np.std(ROC_final)
                PR_avg = np.mean(PR_final)
                PR_std = np.std(PR_final)
                max_accuracy_avg = np.mean(max_accuracy_final)
                max_accuracy_std = np.std(max_accuracy_final)
                baseline_accuracy_avg = np.mean(baseline_accuracy_final)
                baseline_accuracy_std = np.std(baseline_accuracy_final)

                with open(results_path + "/ROC-PR-MA-BA.txt", "w") as f:
                    f.write("ROC average: " + str(ROC_avg) + "\n")
                    f.write("ROC std: " + str(ROC_std) + "\n")
                    f.write("PR average: " + str(PR_avg) + "\n")
                    f.write("PR std: " + str(PR_std) + "\n")
                    f.write("Max accuracy average: " + str(max_accuracy_avg) + "\n")
                    f.write("Max accuracy std: " + str(max_accuracy_std) + "\n")
                    f.write("Baseline accuracy average: " + str(baseline_accuracy_avg) + "\n")
                    f.write("Baseline accuracy std: " + str(baseline_accuracy_std))

                #roc,pr = 0,0
                #print('dim',latent_dim)
                #print('Epoch: ',epoch)
                #print('ROC:',roc)
                #print('PR:',pr)
                ROC_train.append([total_epochs,roc])
                PR_train.append([total_epochs,pr])
                max_accuracy_train.append([total_epochs,max_accuracy])
                baseline_accuracy_train.append([total_epochs,baseline_accuracy])
                #print(ROC)
                #print(PR)
                filename_roc_train=results_path+"/roc_train.txt"
                filename_pr_train=results_path+"/pr_train.txt"
                filename_max_accuracy_train=results_path+"/max_accuracy_train.txt"
                filename_ba=results_path+"/baseline_accuracy_train.txt"
                # save performance statistics
                np.savetxt(filename_roc_train,(ROC_train),delimiter=' ')
                np.savetxt(filename_pr_train,(PR_train),delimiter=' ')
                np.savetxt(filename_max_accuracy_train,(max_accuracy_train),delimiter=' ')
                np.savetxt(filename_ba,(baseline_accuracy_train),delimiter=' ')

                

if __name__ == "__main__":
    run_ldm_tri()         