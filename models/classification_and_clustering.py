import pandas as pd
from sympy import block_collapse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score,plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
from collections import Counter
import os
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})

class dataFrame:
    # base path for loading embeddings
    resultsbase = "./results_final"

    # base path for storing figures
    figurebase = "C:/Users/marcu/Google Drev/DTU/02466(fagprojekt)/Figurer"

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.blockchain = blockchain
        self.month = month
        self.mtype = mtype
        self.dim = dim
        self.setup_paths()
        self.initialize_fontsizes()
        self.preprocess_data()
    

    def setup_paths(self):
        # path for loading results
        self.results_path = self.resultsbase + "/" + self.blockchain + "/" + self.month + "/" + self.mtype

        # path for storing plots
        self.store_path = self.figurebase + "/" + self.blockchain + "/" + self.month + "/Classification"

    def initialize_fontsizes(self):
        # fontsizes for plots
        self.fontsize_title = 45
        self.fontsize_subtitle = 35
        self.fontsize_labels = 40
        self.fontsize_ticks = 35
        self.fontsize_values = 30
        # used to make values bigger in plots
        matplotlib.rcParams["font.size"] = self.fontsize_values

    def preprocess_data(self):
        #load nft embeddings as array in X and categories in y
        self.X = torch.load(self.results_path + "/results/D" + str(self.dim) + "/nft_embeddings").detach().numpy()
        self.y = np.loadtxt(self.results_path + "/sparse_c.txt",dtype="str").reshape(-1,1)

        # split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,stratify=self.y,random_state=42)

        # encode y
        self.encoder = LabelEncoder()
        self.y_train = self.encoder.fit_transform(self.y_train.ravel())
        self.y_test = self.encoder.fit_transform(self.y_test.ravel())
    
    def print_class_distribution(self):
        class_counts = Counter(self.y.ravel())
        print("\nDistribution of classes in {blockchain}-{month} data set:\n".format(blockchain=self.blockchain,month=self.month))
        for name,count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.y) * 100
            print("{name}: {count} appearances --> {p:0.2f}%".format(name=name,count=count,p=percentage))

    def print_encoding_labels(self):
        print("\nEncoding for classes in {blockchain}-{month} data set:\n".format(blockchain=self.blockchain,month=self.month))
        for i, cname in enumerate(self.encoder.classes_):
            print("{name} --> {eid}".format(name=cname,eid=i))
    
class classicationPlotter(dataFrame):
    # size of bar plots
    barplot_figsize = (20,20)

    # y position of title and subtitle barplot
    barplot_title_y = (0.95,0.90) 

    # size of confusion matrix plots
    cm_figsize = (16,16)

    # y position of title and subtitle confusion matrix
    cm_title_y = (0.94,0.89) 

    # empty model variables
    logreg = None
    knn = None

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        # make plotname for specific model type and 
        self.bmname = "{blockchain}-{month}".format(blockchain=blockchain,month=month)
        self.dataname = "{blockchain}-{month}: {mtype}partite {dim:d}D".format(blockchain=blockchain,month=month,mtype=self.mtype.capitalize(),dim=self.dim)

    def set_titles(self,title,subtitle,title_y=(0.94,0.88)):
        # code for introducing subtitle
        plt.text(x=0.53, y=title_y[0], s=title, fontsize=self.fontsize_title, weight="bold",ha="center", transform=self.fig.transFigure)
        plt.text(x=0.53, y=title_y[1], s=subtitle, fontsize=self.fontsize_subtitle, ha="center", transform=self.fig.transFigure)

    def set_axislabels(self,xlabel,ylabel):
        plt.ylabel(xlabel, fontsize=self.fontsize_labels, weight='bold')
        plt.xlabel(ylabel, fontsize=self.fontsize_labels, weight='bold')

    def format_plot(self,title="Basic title",subtitle="Basic subtitle",title_y=(0.94,0.88),xlabel="Basic x-label",ylabel="Basic y-label"):
        self.set_titles(title=title,subtitle=subtitle,title_y=title_y)
        self.set_axislabels(xlabel=xlabel,ylabel=ylabel)

    def make_barplot(self,data,title="Barplot"):
        self.fig = plt.figure(figsize=self.barplot_figsize)
        plt.bar(np.unique(list(data)), height=[sum(data==c) for c in np.unique(list(data))])
        # set axis labels
        plt.xticks([0,1,2,3,4,5],self.encoder.classes_,rotation=45, fontsize=self.fontsize_ticks)
        plt.yticks(fontsize=self.fontsize_ticks)
        self.format_plot(title=title,subtitle=self.bmname,title_y=self.barplot_title_y,xlabel="Category",ylabel="Count")

    def make_barplot_train(self,save=False,show=False):
        self.make_barplot(self.y_train,title="Barplot of categories in the training set")
        if save:
            plt.savefig("{path}/barplot_train".format(path=self.store_path))
        if show:
            plt.show()

    def make_barplot_test(self,save=False,show=False):
        self.make_barplot(self.y_test,title="Barplot of categories in the test set")
        if save:
            plt.savefig("{path}/barplot_test".format(path=self.store_path))
        if show:
            plt.show()

    def print_model_results(self,model,name):
        print('Number of miss-classifications for {0}:\n\t {1} out of {2}'.format(name,np.sum(model.predict(self.X_test)!=self.y_test), len(self.y_test)))
        print('Accuracy for {0}:\n\t {1}'.format(name,model.score(self.X_test, self.y_test)))
        print('Confusion matrix for {0}:\n\t {1}'.format(name,confusion_matrix(self.y_test, model.predict(self.X_test))))
    
    def train_multinomial_logistic_regression(self,solver='lbfgs'):
        self.logreg = lm.LogisticRegression(solver=solver, multi_class='multinomial', max_iter=1000, random_state=42)
        self.logreg.fit(self.X_train,self.y_train)
    
    def get_multinomial_results(self):
        if self.logreg is None:
            # train multinomial logistic regression
            self.train_multinomial_logistic_regression()
        
        print("\nMultinomial logistic regression results for the {blockchain}-{month} data set:\n".format(blockchain=self.blockchain,month=self.month))
        self.print_model_results(self.logreg,"Multinomial Logistic Regression")

    def train_k_nearest_neighbors(self,k=5):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.X_train, self.y_train)

    def train_optimal_k_nearest_neighbors(self,save=False,show=False):
        # optimal knn is defined as the one with the highest accuracy for k=1:30
        n_neighbors = range(1,31)
        knn_scores = []

        for k in n_neighbors:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            knn_scores.append(knn.score(self.X_test, self.y_test))
        
        if save or show:
            self.fig = plt.figure(figsize=self.barplot_figsize)
            plt.plot(n_neighbors,knn_scores,linewidth=5)
            self.format_plot(title="K-nearest neighbors performance plot",subtitle=self.dataname,title_y=self.barplot_title_y,xlabel="Number of neighbors",ylabel="Accuracy")
        
        if save:
            plt.savefig("{path}/knn_performance_plot_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

        optimal_k = np.argmax(knn_scores) + 1
        self.train_k_nearest_neighbors(k=optimal_k)

    def get_k_nearest_neighbors_results(self):
        if self.knn is None:
            # train multinomial logistic regression
            self.train_k_nearest_neighbors()
        
        print("\nK-nearest neighbors results with K={k} for the {blockchain}-{month} data set:\n".format(k=self.knn.n_neighbors,blockchain=self.blockchain,month=self.month))
        self.print_model_results(self.knn,"K-nearest neighbors")

    def make_confusion_matrix(self,modeltype="multinomial",k=5,save=False,show=False):
        # define model
        if modeltype == "multinomial":
            if self.logreg is None:
                self.train_multinomial_logistic_regression()
            model = self.logreg
            title = "Multinomial Logistic Regression"
            fname = "multinomial"
        elif modeltype == "KNN":
            if self.knn is None:
                self.train_k_nearest_neighbors(k=k)
            model = self.knn
            title = "K-nearest neighbors with K={k}".format(k=k)
            fname = "knn{k}".format(k=k)
        elif modeltype == "Optimal KNN":
            self.train_optimal_k_nearest_neighbors()
            model = self.knn
            title = "K-nearest neighbors with K={k}".format(k=self.knn.n_neighbors)
            fname = "knn{k}optimal".format(k=self.knn.n_neighbors)
        
        # plotting
        self.fig, ax = plt.subplots(figsize=self.cm_figsize)
        y_pred = model.predict(self.X_test)
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred,ax=ax)
        # set axis labels
        plt.xticks([0,1,2,3,4,5],self.encoder.classes_,rotation=45,fontsize=self.fontsize_ticks)
        plt.yticks([0,1,2,3,4,5],self.encoder.classes_,rotation=45,fontsize=self.fontsize_ticks)
        self.format_plot(title=title,subtitle=self.dataname,title_y=self.cm_title_y,xlabel="Predicted label",ylabel="True label")

        if save:
            plt.savefig("{path}/confusion_matrix_{fname}_{mtype}_D{dim:d}".format(path=self.store_path,fname=fname,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
    def print_baseline_model_performance(self):
        # baseline performance - predict only the majority class
        majority_class = np.argmax([sum(self.y_train==c) for c in np.unique(list(self.y_train))])
        y_pred = [majority_class] * len(self.y_test)

        accuracy = np.sum(y_pred == self.y_test) / len(self.y_test) * 100
        misclassifications = np.sum(y_pred!=self.y_test)
        
        print("\nBaseline model results for the {blockchain}-{month} data set:\n".format(blockchain=self.blockchain,month=self.month))

        print("Majority class voting accuracy: {accuracy:0.2f}%".format(accuracy=accuracy))
        print("Number of misclassifications for majority voting: {nwrong} out of {ntotal}".format(nwrong=misclassifications,ntotal=len(self.y_test)))
        
# choose data set to investigate
blockchain="ETH"
month="2021-02"
mtype="tri"
dim=3

cp = classicationPlotter(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
cp.print_class_distribution()
cp.print_encoding_labels()
cp.make_barplot_train(save=True)
cp.make_barplot_test(save=True)
cp.make_confusion_matrix("multinomial",save=True)
cp.make_confusion_matrix("KNN",save=True)
cp.make_confusion_matrix("Optimal KNN",save=True)
cp.train_optimal_k_nearest_neighbors(save=True)
cp.print_baseline_model_performance()





#print('ROC AUC for Multinomial regression:\n\t {0}'.format(roc_auc_score(y_test, logreg.predict(X_test))))
#print('Precision for Multinomial regression:\n\t {0}'.format(precision_score(y_test, logreg.predict(X_test))))
#print('Recall for Multinomial regression:\n\t {0}'.format(recall_score(y_test, logreg.predict(X_test))))