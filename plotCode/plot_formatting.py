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
    if platform.system() == "Linux" or platform.system() == "Darwin":
        ndots = ".."
    else:
        ndots = "."
    resultsbase = "{dots}/results_final".format(dots=ndots)

    # base path for storing figures
    figurebase = "../Figurer"

    # base path for loading data
    datapath = "../data"

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.blockchain = blockchain
        self.month = month
        self.mtype = mtype
        self.dim = dim
        self.setup_paths()
    
    def setup_paths(self):
        # path for loading results
        self.results_path = self.resultsbase + "/" + self.blockchain + "/" + self.month + "/" + self.mtype

        # path for storing plots
        self.store_path = self.figurebase + "/" + self.blockchain + "/" + self.month

    
    
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



