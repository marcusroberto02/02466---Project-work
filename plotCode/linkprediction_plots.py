from plot_formatting import Formatter
import os
import numpy as np
import matplotlib.pyplot as plt

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
dims=[9]

for mtype in mtypes:
    for dim in dims:
        lpp = LinkPredictionPlotter(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        lpp.make_score_dim_plot(save=True)
        lpp.make_score_epoch_plot(stype="max_accuracy",save=True)
        lpp.make_score_epoch_all_plot(stype="ROC",save=True)
        lpp.make_score_epoch_all_plot(stype="max_accuracy",save=True)
        lpp.make_baseline_comparison_plot(save=True)
        #lpp.make_score_month_plot(save=True)
