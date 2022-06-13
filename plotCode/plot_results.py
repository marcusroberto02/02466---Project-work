from logging import raiseExceptions
from cvxopt import normal
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
from sklearn.decomposition import PCA
from collections import Counter,defaultdict
import os
import matplotlib
import platform
import seaborn as sns

matplotlib.rcParams.update({'figure.autolayout': True})

ndots = "."

if platform.system() == "Darwin":
    ndots = ".."

class DataFrame:
    # base path for loading embeddings
    resultsbase = "{dots}/results_final".format(dots=ndots)

    # base path for storing figures
    figurebase = "C:/Users/marcu/Google Drev/DTU/02466(fagprojekt)/Figurer"

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.blockchain = blockchain
        self.month = month
        self.mtype = mtype
        self.dim = dim
        self.setup_paths()
        self.preprocess_data()
    
    def setup_paths(self):
        # path for loading results
        self.results_path = self.resultsbase + "/" + self.blockchain + "/" + self.month + "/" + self.mtype

        # path for storing plots
        self.store_path = self.figurebase + "/" + self.blockchain + "/" + self.month

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
    
class Formatter(DataFrame):
    def __init__(self, blockchain="ETH", month="2021-02", mtype="bi", dim=2):
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        # make plotname for specific blockchain and month
        self.bmname = "{blockchain}-{month}".format(blockchain=blockchain,month=month)
        # make plotname for specific blockchain, month and model type
        self.bmmname = "{blockchain}-{month}: {mtype}partite model".format(blockchain=blockchain,month=month,mtype=self.mtype.capitalize(),dim=self.dim)
        # make plotname for specific blockchain, month, model type and dimension
        self.dataname = "{blockchain}-{month}: {mtype}partite {dim:d}D".format(blockchain=blockchain,month=month,mtype=self.mtype.capitalize(),dim=self.dim)

    def initialize_fontsizes_big(self):
        # fontsizes for plots
        self.fontsize_title = 45
        self.fontsize_subtitle = 35
        self.fontsize_labels = 40
        self.fontsize_ticks = 35
        self.fontsize_values = 30
        self.fontsize_legend = 20
        self.markerscale = 15
        # used to make values bigger in 2D plots
        matplotlib.rcParams["font.size"] = self.fontsize_values

    def initialize_fontsizes_small(self):
        # fontsizes for plots
        self.fontsize_title = 25
        self.fontsize_subtitle = 15
        self.fontsize_labels = 20
        self.fontsize_ticks = 15
        self.fontsize_values = 10
        self.fontsize_legend = 15
        self.markerscale = 15
        matplotlib.rcParams["font.size"] = self.fontsize_values

    def set_titles(self,title,subtitle,title_y=(0.94,0.88)):
        # code for introducing subtitle
        plt.text(x=0.53, y=title_y[0], s=title, fontsize=self.fontsize_title, weight="bold",ha="center", transform=self.fig.transFigure)
        plt.text(x=0.53, y=title_y[1], s=subtitle, fontsize=self.fontsize_subtitle, ha="center", transform=self.fig.transFigure)

    def set_titles_3D(self,title,subtitle,title_y=(0.94,0.88)):
        # placeholder is necessary to make space for the actual title and subtitle
        self.fig.suptitle("Placeholder\nPlaceholder\nPlaceholder",color="white")
        self.fig.text(x=0.53, y=title_y[0], s=title, fontsize=self.fontsize_title, weight="bold",ha="center", transform=self.fig.transFigure)
        self.fig.text(x=0.53, y=title_y[1], s=subtitle, fontsize=self.fontsize_subtitle, ha="center", transform=self.fig.transFigure)

    def set_axislabels(self,xlabel,ylabel):
        plt.xlabel(xlabel, fontsize=self.fontsize_labels, weight='bold')
        plt.ylabel(ylabel, fontsize=self.fontsize_labels, weight='bold')

    def format_plot(self,title="Basic title",subtitle="Basic subtitle",title_y=(0.94,0.88),xlabel="Basic x-label",ylabel="Basic y-label"):
        self.set_titles(title=title,subtitle=subtitle,title_y=title_y)
        self.set_axislabels(xlabel=xlabel,ylabel=ylabel)

    def format_plot(self,title="Basic title",subtitle="Basic subtitle",title_y=(0.94,0.88),xlabel="Basic x-label",ylabel="Basic y-label"):
        self.set_titles(title=title,subtitle=subtitle,title_y=title_y)
        self.set_axislabels(xlabel=xlabel,ylabel=ylabel)

    def format_plot_3D(self):
        pass


#######################
# NODE CLASSIFICATION #
#######################

class ClassicationPlotter(Formatter):
    # size of bar plots
    barplot_figsize = (20,20)

    # y position of title and subtitle barplot
    barplot_title_y = (0.95,0.90) 

    # size of confusion matrix plots
    cm_figsize = (16,16)

    # y position of title and subtitle confusion matrix
    cm_title_y = (0.94,0.89)

    # standard line width
    linewidth = 5

    # empty model variables
    logreg = None
    knn = None

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/Classification"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)     

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
            plt.plot(n_neighbors,knn_scores,linewidth=self.lw)
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
dim=5

#cp = ClassicationPlotter(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
#cp.print_class_distribution()
#cp.print_encoding_labels()
#cp.make_barplot_train(save=True)
#cp.make_barplot_test(save=True)
#cp.make_confusion_matrix("multinomial",save=True)
#cp.make_confusion_matrix("KNN",save=True)
#cp.make_confusion_matrix("Optimal KNN",save=True)
#cp.train_optimal_k_nearest_neighbors(save=True)
#cp.print_baseline_model_performance()


###################
# EMBEDDING PLOTS #
###################

# 2D

class EmbeddingPlotter2D(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.95,0.90) 

    # color for embeddings
    colors = {'Games':'red','Art':'green','Collectible':'blue','Metaverse':'orange','Other':'purple','Utility':'brown'}
    
    # empty embedding variables for bi
    z = None
    q = None

    # empty embedding variables for tri
    l = None
    r = None
    u = None

    # coordinate name
    csuffix = defaultdict(lambda: "th")
    csuffix[1] = "st"
    csuffix[2] = "nd"
    csuffix[3] = "rd"

    # size of each data point in plot
    s_big = 4
    s_small = 1

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/EmbeddingPlots/2D"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)    
    
    def load_embeddings_bi(self):
        self.z = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.q = torch.load("{path}/results/D{dim}/trader_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
    
    def load_embeddings_tri(self):
        self.l = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.r = torch.load("{path}/results/D{dim}/seller_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.u = torch.load("{path}/results/D{dim}/buyer_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()

    def make_scatter_plot_bi(self,d1=1,d2=2):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.z is None or self.q is None:
            self.load_embeddings_bi()
        
        self.fig = plt.figure(figsize=self.figsize)
        plt.scatter(self.z[:,d1-1],self.z[:,d2-1],s=self.s_big,label="NFTs")
        plt.scatter(self.q[:,d1-1],self.q[:,d2-1],s=self.s_big,label="Traders")
        plt.legend(loc="upper right",markerscale=self.markerscale)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Scatter plot of embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

    def make_scatter_plot_tri(self,d1=1,d2=2,d3=3,plot3d=False,n_rot=4):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.l is None or self.r is None or self.u is None:
            self.load_embeddings_tri()

        self.fig = plt.figure(figsize=self.figsize)
        plt.scatter(self.l[:,d1-1],self.l[:,d2-1],s=self.s_big,label="NFTs")
        plt.scatter(self.r[:,d1-1],self.r[:,d2-1],s=self.s_big,label="Sellers")
        plt.scatter(self.u[:,d1-1],self.u[:,d2-1],s=self.s_big,label="Buyers")
        plt.legend(loc="upper right",markerscale=self.markerscale)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Scatter plot of embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

    def make_scatter_plot(self,d1=1,d2=2,save=False,show=False):        
        if self.mtype == "bi":
            self.make_scatter_plot_bi(d1=d1,d2=d2)
        elif self.mtype == "tri":
            self.make_scatter_plot_tri(d1=d1,d2=d2)

        if save:
            plt.savefig("{path}/scatter_plot_2D_{d1}_{d2}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_scatter_plot_all_bi(self,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()
        if self.z is None or self.q is None:
            self.load_embeddings_bi()
        
        self.fig, axes = plt.subplots(nrows=self.dim, ncols=self.dim, sharex=True, sharey=True,figsize=self.figsize)
        
        for d1 in range(1,self.dim+1):
            for d2 in range(1,self.dim+1):
                axes[d2-1,d1-1].scatter(self.z[:,d1-1],self.z[:,d2-1],s=self.s_small,label="NFTs")
                axes[d2-1,d1-1].scatter(self.q[:,d1-1],self.q[:,d2-1],s=self.s_small,label="Traders")
                if d1 == 1:
                    ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
                    axes[d2-1,d1-1].set_ylabel(ylabel)
                if d2 == self.dim:
                    xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
                    axes[d2-1,d1-1].set_xlabel(xlabel)

        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale*2,fontsize=self.fontsize_legend)


        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles(title="Scatter plot of embeddings for all pairs",subtitle=self.dataname,title_y=self.fig_title_y)

    def make_scatter_plot_all_tri(self,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()
        if self.l is None or self.r is None and self.u is None:
            self.load_embeddings_tri()
        
        self.fig, axes = plt.subplots(nrows=self.dim, ncols=self.dim, sharex=True, sharey=True,figsize=self.figsize)
        
        for d1 in range(1,self.dim+1):
            for d2 in range(1,self.dim+1):
                axes[d2-1,d1-1].scatter(self.l[:,d1-1],self.l[:,d2-1],s=self.s_small,label="NFTs")
                axes[d2-1,d1-1].scatter(self.r[:,d1-1],self.r[:,d2-1],s=self.s_small,label="Sellers")
                axes[d2-1,d1-1].scatter(self.u[:,d1-1],self.u[:,d2-1],s=self.s_small,label="Buyers")
                if d1 == 1:
                    ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
                    axes[d2-1,d1-1].set_ylabel(ylabel)
                if d2 == self.dim:
                    xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
                    axes[d2-1,d1-1].set_xlabel(xlabel)

        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale*2,fontsize=self.fontsize_legend)

        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles(title="Scatter plot of embeddings for all pairs",subtitle=self.dataname,title_y=self.fig_title_y)

    def make_scatter_plot_all(self,save=False,show=False):
        if self.mtype == "bi":
            self.make_scatter_plot_all_bi(save=save,show=show)
        elif self.mtype == "tri":
            self.make_scatter_plot_all_tri(save=save,show=show)

        if save:
            plt.savefig("{path}/scatter_plot_all_2D_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_category_plot(self,d1=1,d2=2,save=False,show=False):
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.mtype == "bi":
            if self.z is None:
                self.load_embeddings_bi()
        elif self.mtype == "tri":
            if self.l is None:
                self.load_embeddings_tri()

        # load correct nft embedding
        nft_embeddings = self.z if self.mtype == "bi" else self.l

        self.fig = plt.figure(figsize=self.figsize)

        categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
        for category, color in self.colors.items():
            plt.scatter(nft_embeddings[categories==category,d1-1],nft_embeddings[categories==category,d2-1],s=self.s_big,c=color,label=category)
        plt.legend(loc="upper right",markerscale=15)
        xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
        ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
        self.format_plot(title="Category plot for embeddings in 2D",subtitle=self.dataname,title_y=self.fig_title_y,xlabel=xlabel,ylabel=ylabel)

        if save:
            plt.savefig("{path}/category_plot_2D_{d1}_{d2}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_category_plot_all(self,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()
        
        if self.mtype == "bi":
            if self.z is None:
                self.load_embeddings_bi()
        elif self.mtype == "tri":
            if self.l is None:
                self.load_embeddings_tri()

        # load correct nft embedding
        nft_embeddings = self.z if self.mtype == "bi" else self.l
        
        self.fig, axes = plt.subplots(nrows=self.dim, ncols=self.dim, sharex=True, sharey=True,figsize=self.figsize)

        categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
        for d1 in range(1,self.dim+1):
            for d2 in range(1,self.dim+1):
                for category, color in self.colors.items():
                    axes[d2-1,d1-1].scatter(nft_embeddings[categories==category,d1-1],nft_embeddings[categories==category,d2-1],s=self.s_small,c=color,label=category)
                if d1 == 1:
                    ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
                    axes[d2-1,d1-1].set_ylabel(ylabel)
                if d2 == self.dim:
                    xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
                    axes[d2-1,d1-1].set_xlabel(xlabel)

        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale*2,fontsize=self.fontsize_legend)
        
        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles(title="Category plot for embeddings for all pairs",subtitle=self.dataname,title_y=self.fig_title_y)

        if save:
            plt.savefig("{path}/category_plot_all_2D_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
# choose data set to investigate
blockchain="ETH"
month="2021-02"
mtypes=["bi","tri"]
dims=[2]
"""
for mtype in mtypes:
    for dim in dims:
        ep = EmbeddingPlotter2D(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        ep.make_scatter_plot(save=True)
        ep.make_category_plot(save=True)
        ep.make_scatter_plot_all(save=True)
        ep.make_category_plot_all(save=True)
"""
# 3D

class EmbeddingPlotter3D(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.95,0.90) 

    # color for embeddings
    colors = {'Games':'red','Art':'green','Collectible':'blue','Metaverse':'orange','Other':'purple','Utility':'brown'}
    
    # empty embedding variables for bi
    z = None
    q = None

    # empty embedding variables for tri
    l = None
    r = None
    u = None

    # coordinate name
    csuffix = defaultdict(lambda: "th")
    csuffix[1] = "st"
    csuffix[2] = "nd"
    csuffix[3] = "rd"

    # size of each data point in plot
    s_big = 0.1
    s_small = 0.1

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/EmbeddingPlots/3D"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)    
    
    def load_embeddings_bi(self):
        self.z = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.q = torch.load("{path}/results/D{dim}/trader_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
    
    def load_embeddings_tri(self):
        self.l = torch.load("{path}/results/D{dim}/nft_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.r = torch.load("{path}/results/D{dim}/seller_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.u = torch.load("{path}/results/D{dim}/buyer_embeddings".format(path=self.results_path,dim=self.dim)).detach().numpy()


    def make_scatter_plot_bi(self,d1=1,d2=2,d3=3,n_rot=4,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()

        if self.dim < 3:
            raise Exception("Cannot plot {dim}D embeddings in 3D".format(dim=self.dim))
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1) and d3 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.z is None or self.q is None:
            self.load_embeddings_bi()
        
        self.fig = plt.figure(figsize=self.figsize)
        
        for i in range(n_rot*n_rot):
            ax = self.fig.add_subplot(n_rot,n_rot,i+1,projection='3d')
            ax.scatter(self.z[:,d1-1],self.z[:,d2-1],self.z[:,d3-1],s=self.s_big,label="NFTs")
            ax.scatter(self.q[:,d1-1],self.q[:,d2-1],self.q[:,d3-1],s=self.s_big,label="Traders")
            ax.set_title("Rotation: " + str(i*360/(n_rot*n_rot)),weight="bold")
            ax.view_init(azim=i*360/(n_rot*n_rot))
            xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
            ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
            zlabel = "{c3} coordinate".format(c3=str(d3)+self.csuffix[d3])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)

    def make_scatter_plot_tri(self,d1=1,d2=2,d3=3,n_rot=4,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()

        if self.dim < 3:
            raise Exception("Cannot plot {dim}D embeddings in 3D".format(dim=self.dim))
        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1) and d3 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.l is None or self.r is None or self.u is None:
            self.load_embeddings_tri()
        
        self.fig = plt.figure(figsize=self.figsize)
        
        for i in range(n_rot*n_rot):
            ax = self.fig.add_subplot(n_rot,n_rot,i+1,projection='3d')
            ax.scatter(self.l[:,d1-1],self.l[:,d2-1],self.l[:,d3-1],s=self.s_big,label="NFTs")
            ax.scatter(self.r[:,d1-1],self.r[:,d2-1],self.r[:,d3-1],s=self.s_big,label="Sellers")
            ax.scatter(self.u[:,d1-1],self.u[:,d2-1],self.u[:,d3-1],s=self.s_big,label="Buyers")
            ax.set_title("Rotation: " + str(i*360/(n_rot*n_rot)),weight="bold")
            ax.view_init(azim=i*360/(n_rot*n_rot))
            xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
            ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
            zlabel = "{c3} coordinate".format(c3=str(d3)+self.csuffix[d3])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        
    def make_scatter_plot(self,d1=1,d2=2,d3=3,n_rot=4,save=False,show=False):        
        if self.mtype == "bi":
            self.make_scatter_plot_bi(d1=d1,d2=d2,d3=d3,n_rot=n_rot)
        elif self.mtype == "tri":
            self.make_scatter_plot_tri(d1=d1,d2=d2,d3=d3,n_rot=n_rot)

        # prepare legend
        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale,borderpad=2,fontsize=self.fontsize_legend)

        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles_3D(title="Scatter plot of embeddings in 3D",subtitle=self.dataname,title_y=self.fig_title_y)

        if save:
            plt.savefig("{path}/scatter_plot_3D_{d1}_{d2}_{d3}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,d3=d3,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
    def make_category_plot(self,d1=1,d2=2,d3=3,n_rot=4,save=False,show=False):
        # because we are plotting many plots use the small fontsizes
        self.initialize_fontsizes_small()

        if not (d1 in range(1,self.dim+1) and d2 in range(1,self.dim+1) and d3 in range(1,self.dim+1)):
            raise Exception("Invalid choice of coordinate dimensions")

        if self.mtype == "bi":
            if self.z is None:
                self.load_embeddings_bi()
        elif self.mtype == "tri":
            if self.l is None:
                self.load_embeddings_tri()

        # load correct nft embedding
        nft_embeddings = self.z if self.mtype == "bi" else self.l

        self.fig = plt.figure(figsize=self.figsize)

        for i in range(n_rot*n_rot):
            ax = self.fig.add_subplot(n_rot,n_rot,i+1,projection='3d')
            categories = np.loadtxt("{path}/sparse_c.txt".format(path=self.results_path),dtype='str')
            for category, color in self.colors.items():
                ax.scatter(nft_embeddings[categories==category,d1-1],nft_embeddings[categories==category,d2-1],s=self.s_big,c=color,label=category)
            ax.set_title("Rotation: " + str(i*360/(n_rot*n_rot)),weight="bold")
            ax.view_init(azim=i*360/(n_rot*n_rot))
            xlabel = "{c1} coordinate".format(c1=str(d1)+self.csuffix[d1])
            ylabel = "{c2} coordinate".format(c2=str(d2)+self.csuffix[d2])
            zlabel = "{c3} coordinate".format(c3=str(d3)+self.csuffix[d3])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)

        # get legend
        lines, labels = self.fig.axes[-1].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc = 'upper right',markerscale=self.markerscale,fontsize=self.fontsize_legend)

        # go back to big fontsize
        self.initialize_fontsizes_big()
        self.set_titles_3D(title="Category plot for embeddings in 3D",subtitle=self.dataname,title_y=self.fig_title_y)
        
        if save:
            plt.savefig("{path}/category_plot_3D_{d1}_{d2}_{d3}_{mtype}_D{dim:d}".format(path=self.store_path,d1=d1,d2=d2,d3=d3,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
    
    

    
    

# choose data set to investigate
blockchain="ETH"
month="2021-02"
mtypes=["bi","tri"]
dims=[2]

"""
for mtype in mtypes:
    for dim in dims:
        ep = EmbeddingPlotter3D(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        ep.make_scatter_plot(save=True)
        ep.make_category_plot(save=True)
"""

###################
# LINK PREDICTION #
###################

class LinkPredictionPlotter(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.95,0.90)

    # standard linewidth
    linewidth = 5
    
    # standard markersize
    s_big = 200
    markersize = 15

    # size of test batch
    n_test_batch = 1000

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/LinkPrediction"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)    

    # get error bar size
    def get_error_barsize(self,std):
        return 1.96*std/np.sqrt(self.n_test_batch)

    def make_score_epoch_plot(self,stype = "ROC",save = False, show = False):
        self.fig = plt.figure(figsize = self.figsize)

        path = f"/results/D{self.dim}/{stype.lower()}_train.txt"

        scores = np.loadtxt(self.results_path+path)
        plt.plot(*scores.T, color = "blue", lw = self.linewidth)

        title = f"{stype}-AUC score as a function of epochs" if stype != "max_accuracy" else f"Maximum accuracy as a function of epochs"
        ylabel = f"{stype}-AUC score" if stype != "max_accuracy" else "Accuracy"

        self.format_plot(title=title, subtitle=self.dataname,
                         title_y=self.fig_title_y, xlabel="Nr of epochs", ylabel=ylabel)
        if save:
            plt.savefig("{path}/{stype}_epoch_plot_{mtype}_D{dim:d}".format(path=self.store_path,stype=stype,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_score_epoch_all_plot(self,stype = "ROC",save = False, show = False):
        self.fig = plt.figure(figsize = self.figsize)

        dims = [1,2,5,8,10]
        colors = ["blue","red","green","orange","purple"]
        for dim, color in zip(dims,colors):
            path = f"/results/D{dim}/{stype.lower()}_train.txt"

            scores = np.loadtxt(self.results_path+path)
            plt.plot(*scores.T, color = color, lw = self.linewidth,label=f"Dim: {dim}")

        title = f"{stype}-AUC score as a function of epochs" if stype != "max_accuracy" else f"Maximum accuracy as a function of epochs"
        ylabel = f"{stype}-AUC score" if stype != "max_accuracy" else "Accuracy"

        plt.legend(loc="lower right")
        self.format_plot(title=title, subtitle=self.bmmname,
                         title_y=self.fig_title_y, xlabel="Nr of epochs", ylabel=ylabel)
        if save:
            plt.savefig("{path}/{stype}_epoch_plot_all_{mtype}".format(path=self.store_path,stype=stype,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_score_dim_plot(self, stype = "ROC", save = False, show = False):
        self.fig = plt.figure(figsize=self.figsize)
        ROC_scores = []
        ROC_errorbars = []
        PR_scores = []
        PR_errorbars = []
        MA_scores = []
        MA_errorbars = []

        dims = range(1,11)
        for i in dims:
            with open(self.results_path + f"/results/D{i}/ROC-PR-MA-BA.txt", 'r') as f:
                vals = [float(l.strip().split()[-1]) for l in f.readlines()]
                ROC_scores.append(vals[0])
                ROC_errorbars.append(self.get_error_barsize(vals[1]))
                PR_scores.append(vals[2])
                PR_errorbars.append(self.get_error_barsize(vals[3]))
                MA_scores.append(vals[4])
                MA_errorbars.append(self.get_error_barsize(vals[5]))

        scores = ROC_scores
        errorbars = ROC_errorbars

        if stype == "PR":
            scores = PR_scores
            errorbars = PR_errorbars
        elif stype == "max_accuracy":
            scores = MA_scores
            errorbars = MA_errorbars

        plt.errorbar(dims,scores,errorbars,lw = self.linewidth,capsize=30,markeredgewidth=5,zorder=0)
        plt.scatter(dims,scores,marker='o',color='red',s=self.s_big,zorder=1)
        plt.xticks(range(1,11))

        title = f"{stype}-AUC score as a function of epochs" if stype != "max_accuracy" else f"Maximum accuracy as a function of epochs"
        ylabel = f"{stype}-AUC score" if stype != "max_accuracy" else "Accuracy"

        self.format_plot(title=title.format(stype=stype), subtitle=self.bmmname,
                         title_y=self.fig_title_y,xlabel="Nr. of latent dimensions",ylabel=ylabel)
        if save:
            plt.savefig("{path}/{stype}_dim_plot_{mtype}".format(path=self.store_path,stype=stype,mtype=self.mtype))
        if show:
            plt.show()

    def make_score_month_plot(self, stype = "ROC", save = False, show = False):
        self.fig = plt.figure(figsize=self.figsize)
        ROC_scores = []
        PR_scores = []
        MA_scores = []

        months = ["2020-01","2020-02","2020-03","2020-04","2020-05",
                  "2020-06","2020-07","2020-08","2020-09","2020-10",
                  "2020-11","2020-12","2021-01","2021-02","2021-03"]

        for month in months:
            with open(self.resultsbase + f"/{self.blockchain}/{month}/{self.mtype}/results/D{self.dim}/ROC-PR-MA-BA.txt", 'r') as f:
                vals = [float(l.strip().split()[-1]) for l in f.readlines()]
                ROC_scores.append(vals[0])
                PR_scores.append(vals[2])
                MA_scores.append(vals[4])
        
        scores = ROC_scores

        if stype == "PR":
            scores = PR_scores
        elif stype == "max_accuracy":
            scores = MA_scores

        
        plt.plot(range(len(months)),scores,marker='o',mfc='red',markersize=self.markersize,lw = self.linewidth)
        plt.xticks(range(len(months)),months,rotation=45,fontsize=self.fontsize_ticks)

        title = f"{stype}-AUC score as a function of months" if stype != "max_accuracy" else f"Maximum accuracy as a function of months"
        ylabel = f"{stype}-AUC score" if stype != "max_accuracy" else "Accuracy"

        self.format_plot(title=title, subtitle=self.bmmname,
                         title_y=self.fig_title_y,xlabel="Month",ylabel=ylabel)
        if save:
            plt.savefig("{path}/{stype}_month_plot_{mtype}_D{dim:d}".format(path=self.store_path,stype=stype,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

    def make_baseline_comparison_plot(self, stype = "max_accuracy", save = False,show = False):
        self.fig = plt.figure(figsize=self.figsize)

        path = f"/results/D{self.dim}/"

        scores = np.loadtxt(self.results_path + path+f"{stype.lower()}_train.txt")
        baseline_scores = np.loadtxt(self.results_path + path + "baseline_accuracy_train.txt")
        plt.plot(*scores.T, color="blue", label = "Link prediction", lw = self.linewidth)
        plt.plot(*baseline_scores.T, color = "green", label = "Baseline model", lw = self.linewidth)
        plt.legend(loc="lower right")

        self.format_plot(title=f"Accuracy as a function of epochs", subtitle=self.dataname,
                         title_y=self.fig_title_y, xlabel="Nr of epochs", ylabel="Accuracy")
        if save:
            plt.savefig("{path}/{stype}_baseline_plot_{mtype}_D{dim:d}".format(path=self.store_path, stype=stype,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()



    
# choose data set to investigate
blockchain="ETH"
month="2021-02"
mtypes=["bi","tri"]
dims=[2,3]

for mtype in mtypes:
    for dim in dims:
        lpp = LinkPredictionPlotter(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        #lpp.make_score_dim_plot(save=True)
        #lpp.make_score_epoch_plot(stype="max_accuracy",save=True)
        #lpp.make_score_epoch_all_plot(stype="ROC",save=True)
        #lpp.make_score_epoch_all_plot(stype="max_accuracy",save=True)
        #lpp.make_baseline_comparison_plot(save=True)
        lpp.make_score_month_plot(save=True)


###################
# TRADER STORIES  #
###################

class TraderStoryPlotter(Formatter):
    # standard size
    figsize = (20,20)

    # y position of title and subtitle
    fig_title_y = (0.96,0.91)

    # standard linewidth
    linewidth = 5

    # empty variables for bias terms
    seller_biases = None
    buyer_biases = None

    def __init__(self,blockchain="ETH",month="2021-02",mtype="tri",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.store_path += "/TraderStories"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

    
    def load_biases(self):
        self.seller_biases = torch.load("{path}/results/D{dim}/seller_biases".format(path=self.results_path,dim=self.dim)).detach().numpy()
        self.buyer_biases = torch.load("{path}/results/D{dim}/buyer_biases".format(path=self.results_path,dim=self.dim)).detach().numpy()

    # code from the documentation of matplotlib
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    def scatter_hist(self,x, y, ax, ax_histx, ax_histy,normalize=False):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # Set bottom and left spines as x and y axes of coordinate system
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # the scatter plot:
        ax.scatter(x, y)


        # OBS 
        # HER SKAL MAN MANUELT UNDERSØGE HVILKE INDEKSER DER FJERNER 0'erne
        if normalize:
            ax.set_xticks(range(-4,5),range(-4,5))
            ax.set_yticks(range(-4,5),range(-4,5))
        idx,idy = (4,4) if normalize else (3,3)
        xticks = ax.xaxis.get_ticklabels()
        xticks[idx].set_visible(False)
        yticks = ax.yaxis.get_ticklabels()
        yticks[idy].set_visible(False)

        # Create 'x' and 'y' labels placed at the end of the axes
        ax.set_xlabel(r'$\nu$', size=28, labelpad=-24)
        ax.set_ylabel(r'$\tau$', size=28, labelpad=-21,rotation=0)

        ax.xaxis.set_label_coords(0.99, 0.55)
        ax.yaxis.set_label_coords(0.52, 0.97)

        # Draw arrows
        arrow_fmt = dict(markersize=4, color='black', clip_on=False)
        ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
        ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

        

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

    def make_bias_distribution_plot(self,normalize=False,save=False,show=False):
        if self.seller_biases is None or self.buyer_biases is None:
            self.load_biases()
        
        # only include traders that act as both sellers and buyers
        df = pd.read_csv(self.results_path + "/sellerbuyeridtable.csv")
        seller_ids = df["ei_seller"]
        buyer_ids = df['ei_buyer']

        self.fig = plt.figure(figsize=self.figsize)

        gs = self.fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)


        ax = self.fig.add_subplot(gs[1, 0])
        ax_histx = self.fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = self.fig.add_subplot(gs[1, 1], sharey=ax)

        # make the scatterplot
        sb = self.seller_biases[seller_ids]
        bb = self.buyer_biases[buyer_ids]
        if normalize:
            sb = (sb - np.mean(sb)) / np.std(sb)
            bb = (bb - np.mean(bb)) / np.std(bb)
        self.scatter_hist(sb,bb, ax, ax_histx, ax_histy,normalize=normalize)
        title = "Seller and buyer biases distribution"
        title += " - Normalized" if normalize else ""
        self.set_titles_3D(title=title,subtitle=self.dataname,title_y=self.fig_title_y)
        
        if save:
            if normalize:
                plt.savefig("{path}/bias_distribution_plot_normalized_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
            else:
                plt.savefig("{path}/bias_distribution_plot_{mtype}_D{dim:d}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()
        
        


# choose data set to investigate
blockchain="ETH"
month="2021-02"
mtype="tri"
dims=[2,3]

#for dim in dims:
#    tsp = TraderStoryPlotter(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
#    tsp.make_bias_distribution_plot(save=True)
#    tsp.make_bias_distribution_plot(normalize=True,save=True)