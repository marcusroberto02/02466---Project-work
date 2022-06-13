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

class DataFrame:
    # base path for loading embeddings
    ndots = "." if platform.system() != "Darwin" else ".."
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

        print("\nRemember to set the correct paths in the dataframe class in the plot_formatting.py file!\n")

        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        # make plotname for specific blockchain and month
        self.bmname = "{blockchain}-{month}".format(blockchain=blockchain,month=month)
        # make plotname for specific blockchain, month and model type
        self.bmmname = "{blockchain}-{month}: {mtype}partite model".format(blockchain=blockchain,month=month,mtype=self.mtype.capitalize(),dim=self.dim)
        # make plotname for specific blockchain, month, model type and dimension
        self.dataname = "{blockchain}-{month}: {mtype}partite {dim:d}D".format(blockchain=blockchain,month=month,mtype=self.mtype.capitalize(),dim=self.dim)
        
        # make plots auto fill out
        matplotlib.rcParams.update({'figure.autolayout': True})

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



