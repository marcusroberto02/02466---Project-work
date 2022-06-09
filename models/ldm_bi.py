# HUSK CREDITS!!!

# Import all the packages
import pandas
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
import os

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
def run_ldm_bi(dataset=None, latent_dims = [2],total_epochs = 10000,n_test_batches = 5,lrs=[0.001],total_runs = 1):
    class LDM_BI(nn.Module):
        def __init__(self,sparse_i,sparse_j,sparse_w,nft_size,trader_size,latent_dim,nft_sample_size,test_batch_size=1000,sparse_i_test=None,sparse_j_test=None,sparse_w_test=None):
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

            # used for the test set
            self.sparse_i_test = sparse_i_test
            self.sparse_j_test = sparse_j_test
            self.sparse_w_test = sparse_w_test

            # used for sampling
            self.sampling_weights_nfts = torch.ones(self.nft_size,device = device)
            self.sampling_weights_traders=torch.ones(self.trader_size,device = device)
            self.nft_sample_size=nft_sample_size

            # size of each test_batch
            self.test_size = self.sparse_i_test.size(0)
            self.sampling_weights_test = torch.ones(self.test_size,device=device)

            # size of each test batch
            self.test_batch_size = int(1/5 * len(self.sparse_j_test))

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

            # mini batch over network
            sample_i, sample_j, sample_weights = self.sample_network()
            unique_i = torch.unique(sample_i)
            unique_j = torch.unique(sample_j)

            # exp(||z_i - q_j||)
            mat=torch.exp(-(torch.cdist(self.latent_z[unique_i]+1e-06,self.latent_q[unique_j],p=2)+1e-06))
            # Non-link N^2 likelihood term, i.e. \sum_ij lambda_ij
            # for the bipartite case the diagonal part should be removed
            # as well as the 1/2 term
            # exp(gamma)*exp(delta)*exp(mat)
            z_pdist1=torch.mm(torch.mm(torch.exp(self.gamma[unique_i].unsqueeze(0)),mat),torch.exp(self.delta[unique_j].unsqueeze(-1)))
            # log-Likehood link term i.e. \sum_ij y_ij*log(lambda_ij)
            zqdist = -((((self.latent_z[sample_i]-self.latent_q[sample_j]+1e-06)**2).sum(-1))**0.5)
            z_pdist2=(sample_weights*(self.gamma[sample_i]+self.delta[sample_j]+zqdist)).sum()

            # Total Log-likelihood
            log_likelihood_sparse=z_pdist2-z_pdist1
        
        
            return log_likelihood_sparse
        
        def sample_network(self):
            # sample nfts
            sample_idx = torch.multinomial(self.sampling_weights_nfts,self.nft_sample_size,replacement=False)

            # extract edges
            edge_translator = torch.isin(self.sparse_i_idx,sample_idx)

            # get edges
            sample_i_idx = self.sparse_i_idx[edge_translator]
            sample_j_idx = self.sparse_j_idx[edge_translator]
            sample_weights = self.weights[edge_translator]

            return sample_i_idx, sample_j_idx, sample_weights

        def create_test_batch(self):
            sample_idx = torch.multinomial(self.sampling_weights_test,self.test_batch_size,replacement=False)

            # positive class
            test_i_pos = self.sparse_i_test[sample_idx]
            test_j_pos = self.sparse_j_test[sample_idx]

            # negative class
            test_i_neg = torch.randint(0,self.nft_size,size=(self.test_batch_size,))
            test_j_neg = torch.randint(0,self.trader_size,size=(self.test_batch_size,))
            
            return test_i_pos, test_j_pos, test_i_neg, test_j_neg

        
        def link_prediction(self):
            # apply link prediction to the test set
            with torch.no_grad():
                test_i_pos,test_j_pos,test_i_neg,test_j_neg = self.create_test_batch()

                # get lambda values for negative edges
                rates_neg = torch.zeros(self.test_batch_size)
                for t in range(self.test_batch_size):
                    i,j = test_i_neg[t],test_j_neg[t]
                    gamma_i = self.gamma[i]
                    delta_j = self.delta[j]
                    zq_dist = ((self.latent_z[i]-self.latent_q[j]+1e-06)**2).sum(-1)**0.5
                    rates_neg[t] = torch.exp(gamma_i+delta_j-zq_dist)

                # get lambda values for positive edges
                rates_pos = torch.zeros(self.test_batch_size)
                for t in range(self.test_batch_size):
                    i,j = test_i_pos[t],test_j_pos[t]
                    gamma_i = self.gamma[i]
                    delta_j = self.delta[j]
                    zq_dist = ((self.latent_z[i]-self.latent_q[j]+1e-06)**2).sum(-1)**0.5
                    rates_pos[t] = torch.exp(gamma_i+delta_j-zq_dist)

                self.rates = torch.cat((rates_neg,rates_pos))
                self.target=torch.cat((torch.zeros(self.test_batch_size),torch.ones(self.test_batch_size)))
                precision, tpr, thresholds = metrics.precision_recall_curve(self.target.cpu().data.numpy(), self.rates.cpu().data.numpy())


            return metrics.roc_auc_score(self.target.cpu().data.numpy(),self.rates.cpu().data.numpy()),metrics.auc(tpr,precision)
        
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
            results_path = dataset + "/bi/results/D" + str(latent_dim)
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            print('Latent Communities: ',latent_dim)

            for lr in lrs:
                print('Learning rate: ',lr)

                # TRAIN SET
                trainset = dataset + "/bi/train"
                # nft items
                sparse_i=torch.from_numpy(np.loadtxt(trainset+'/sparse_i.txt')).long().to(device)
                # traders
                sparse_j=torch.from_numpy(np.loadtxt(trainset+'/sparse_j.txt')).long().to(device)
                # weight items
                sparse_w=torch.from_numpy(np.loadtxt(trainset+'/sparse_w.txt')).long().to(device)

                # TEST SET
                testset = dataset + "/bi/test"
                # nft items
                sparse_i_test=torch.from_numpy(np.loadtxt(testset+'/sparse_i.txt')).long().to(device)
                # traders
                sparse_j_test=torch.from_numpy(np.loadtxt(testset+'/sparse_j.txt')).long().to(device)
                # weight items
                sparse_w_test=torch.from_numpy(np.loadtxt(testset+'/sparse_w.txt')).long().to(device)

                # network size
                N=int(sparse_i.max()+1)
                T=int(sparse_j.max()+1)
                
                # initialize model
                model = LDM_BI(sparse_i=sparse_i,sparse_j=sparse_j,sparse_w=sparse_w,nft_size=N,trader_size=T,
                            latent_dim=latent_dim,nft_sample_size=1000,
                            sparse_i_test=sparse_i_test,sparse_j_test=sparse_j_test,sparse_w_test=sparse_w_test).to(device)         

                optimizer = optim.Adam(model.parameters(), lr=lr)  
        
                print('Dataset: ',dataset)
                print('##################################################################')

                losses=[]
                ROC_train=[]
                PR_train=[]
                # model.scaling=1

                for epoch in range(total_epochs):
                    
                    # get log likelihood
                    loss=-model.LSM_likelihood_bias(epoch=epoch)
                    losses.append(loss.item())
                    
                    optimizer.zero_grad() # clear the gradients.   
                    loss.backward() # backpropagate
                    optimizer.step() # update the weights
                    if epoch%100==0:
                        # AUC-ROC and PR-AUC
                        # Receiver operating characteristic-area under curve   AND precision recal-area under curve
                        roc,pr=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr
                        #roc,pr = 0,0
                        #print('Epoch: ',epoch)
                        #print('ROC:',roc)
                        #print('PR:',pr)
                        ROC_train.append([epoch,roc])
                        PR_train.append([epoch,pr])
                
                # save bias/random-effect      
                torch.save(model.gamma.detach().cpu(),results_path+"/nft_biases")
                torch.save(model.delta.detach().cpu(),results_path+"/trader_biases")
                # save latent embedding position
                torch.save(model.latent_z.detach().cpu(),results_path+"/nft_embeddings")
                torch.save(model.latent_q.detach().cpu(),results_path+"/trader_embeddings")
                # run predictions on multiple test batches to obtain error bars
                ROC_final = []
                PR_final = [] 
                for i in range(n_test_batches):
                    roc,pr=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr
                    ROC_final.append(roc)
                    PR_final.append(pr)
                ROC_avg = np.mean(ROC_final)
                ROC_std = np.std(ROC_final)
                PR_avg = np.mean(PR_final)
                PR_std = np.std(PR_final)
                #roc, pr = 0,0
                #print('dim',latent_dim)
                #print('Epoch: ',epoch)
                #print('ROC:',roc)
                #print('PR:',pr)
                ROC_train.append([total_epochs,roc])
                PR_train.append([total_epochs,pr])
                #print(ROC)
                #print(PR)
                filename_roc_train=results_path+"/roc_train.txt"
                filename_pr_train=results_path+"/pr_train.txt"
                    
                # save performance statistics
                np.savetxt(filename_roc_train,(ROC_train),delimiter=' ')
                np.savetxt(filename_pr_train,(PR_train),delimiter=' ')

                with open(results_path + "/ROC-PR.txt", "w") as f:
                    f.write("ROC average: " + str(ROC_avg) + "\n")
                    f.write("ROC std: " + str(ROC_std) + "\n")
                    f.write("PR average: " + str(PR_avg) + "\n")
                    f.write("PR std: " + str(PR_std))

if __name__ == "__main__":
    run_ldm_bi()             
                