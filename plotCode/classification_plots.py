from plot_formatting import Formatter
import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn.linear_model as lm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import scipy.stats
import scipy.stats as st

#######################
# NODE CLASSIFICATION #
#######################

class ClassicationPlotter(Formatter):
    # size of bar plots
    barplot_figsize = (20,20)

    # very big size
    bigplot_figsize = (23,23)

    # y position of title and subtitle barplot
    barplot_title_y = (0.95,0.90) 

    # size of confusion matrix plots
    cm_figsize = (16,16)

    # y position of title and subtitle confusion matrix
    cm_title_y = (0.94,0.89)

    # standard line width
    linewidth = 5

    # standard markersize
    markersize = 15

    # empty model variables
    logreg = None
    knn = None

    def __init__(self,blockchain="ETH",month="2021-02",mtype="bi",dim=2):
        self.initialize_fontsizes_big()
        super().__init__(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        self.preprocess_data()
        self.store_path += "/Classification"
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

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
    
    def make_model_dim_plot(self,modeltype="multinomial",solver='lbfgs',k=5,save=False,show=False):
        self.fig = plt.figure(figsize=self.barplot_figsize)

        dims = range(1,11)
        scores = []

        for dim in dims:
            #load nft embeddings as array in X and categories in y
            X = torch.load(self.results_path + "/results/D" + str(dim) + "/nft_embeddings").detach().numpy()
            y = np.loadtxt(self.results_path + "/sparse_c.txt",dtype="str").reshape(-1,1)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

            # encode y
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train.ravel())
            y_test = encoder.fit_transform(y_test.ravel())

            model = lm.LogisticRegression(solver=solver, multi_class='multinomial', max_iter=1000, random_state=42)
            if modeltype == "KNN":
                model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train,y_train)

            scores.append(model.score(X_test, y_test))

        plt.plot(dims,scores,marker='o',mfc='red',markersize=self.markersize,lw = self.linewidth)
        plt.xticks(dims,dims)
        plt.ylim([0,1])
        
        title = "Multinomial logistic regression results"
        if modeltype == "KNN":
            title = f"K-nearest neighbors results with K={k}"

        self.format_plot(title=title, subtitle=self.bmmname,
                         title_y=self.barplot_title_y,xlabel="Nr. of latent dimensions",ylabel="Accuracy")
        if save:
            plt.savefig("{path}/{modeltype}_dim_plot_{mtype}".format(path=self.store_path,modeltype=modeltype,mtype=self.mtype))
        if show:
            plt.show()

    def make_dim_plot_all(self,solver='lbfgs',k=5,save=False, show = False):
        self.fig = plt.figure(figsize=self.barplot_figsize)
        dims = range(1,11)
        bi_reg_scores = []
        bi_knn_scores = []
        tri_reg_scores = []
        tri_knn_scores = []
        path = "/".join(self.results_path.split("/")[:-1])

        y = np.loadtxt(self.results_path + "/sparse_c.txt",dtype="str").reshape(-1,1)
        # encode y
        encoder = LabelEncoder()
        y = encoder.fit_transform(y.ravel())
        for dim in dims:
            print(dim)
            X_bi = torch.load(path + "/bi" + "/results/D" + str(dim) + "/nft_embeddings").detach().numpy()
            X_tri = torch.load(path + "/tri" + "/results/D" + str(dim) + "/nft_embeddings").detach().numpy()

            # split data into train and test
            X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(X_bi, y, test_size=0.2, stratify=y, random_state=42)
            X_train_tri, X_test_tri, y_train_tri, y_test_tri = train_test_split(X_tri, y, test_size=0.2, stratify=y,
                                                                            random_state=42)

            lr = lm.LogisticRegression(solver=solver, multi_class='multinomial', max_iter=1000, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=k)
            bi_reg_scores.append(lr.fit(X_train_bi,y_train_bi).score(X_test_bi,y_test_bi))
            bi_knn_scores.append(knn.fit(X_train_bi,y_train_bi).score(X_test_bi,y_test_bi))
            tri_reg_scores.append(lr.fit(X_train_tri,y_train_tri).score(X_test_tri,y_test_tri))
            tri_knn_scores.append(knn.fit(X_train_tri,y_train_tri).score(X_test_tri,y_test_tri))

        plt.plot(dims, bi_reg_scores, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth, label = "Bipartite - MLR")
        plt.plot(dims, bi_knn_scores, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth, label = f"Bipartite - KNN (K={k})")
        plt.plot(dims, tri_reg_scores, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth, label = "Tripartite - MLR")
        plt.plot(dims, tri_knn_scores, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth, label = f"Tripartite - KNN (K={k})")
        plt.xticks(dims, dims)
        plt.legend(loc="lower right")
        plt.ylim([0, 1])
        self.format_plot(title = "Model performance",subtitle = self.bmname,title_y=self.barplot_title_y,xlabel="Nr. of latent dimensions", ylabel = "Accuracy")
        if save:
            plt.savefig("{path}/dim_plot_all_models".format(path=self.store_path))
        if show:
            plt.show()

    def make_month_plot_all(self,solver='lbfgs',k=5,save=False,show=False):
        self.fig = plt.figure(figsize=self.barplot_figsize)
        mtypes = ["bi","tri"]
        dims = [2,3]
        months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05",
                  "2020-06", "2020-07", "2020-08", "2020-09", "2020-10",
                  "2020-11", "2020-12", "2021-01", "2021-02", "2021-03"]

        for mtype in mtypes:
            for dim in dims:
                reg_scores = []
                knn_scores = []
                baseline_scores = []
                for month in months:
                    print(month,mtype,dim)
                    path = self.resultsbase + f"/{self.blockchain}/{month}"

                    y = np.loadtxt(path + f"/{mtype}/sparse_c.txt",dtype="str").reshape(-1,1)
                    # encode y
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(y.ravel())
                    
                    X = torch.load(path + f"/{mtype}/results/D{dim}/nft_embeddings").detach().numpy()
                    
                    # split data into train and test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                    
                    lr = lm.LogisticRegression(solver=solver, multi_class='multinomial', max_iter=1000, random_state=42)
                    knn = KNeighborsClassifier(n_neighbors=k)
                    reg_scores.append(lr.fit(X_train,y_train).score(X_test,y_test))
                    knn_scores.append(knn.fit(X_train,y_train).score(X_test,y_test))

                    if mtype == "tri" and dim == 3:
                        cc = Counter(y_train)
                        majority_class = np.argmax([cc[c] for c in range(6)])
                        y_pred = [majority_class] * len(y_test)

                        accuracy = np.sum(y_pred == y_test) / len(y_test)
                        baseline_scores.append(accuracy)
                    
                plt.plot(range(len(months)), reg_scores, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth, label = f"{mtype.capitalize()}partite {dim}D - MLR")
                plt.plot(range(len(months)), knn_scores, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth, label = f"{mtype.capitalize()}partite {dim}D - KNN (K={k})")
                if mtype == "tri" and dim == 3:
                    plt.plot(range(len(months)), baseline_scores, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth, label = f"Baseline model")

        plt.xticks(range(len(months)), months, rotation=45, fontsize=self.fontsize_ticks)
        plt.legend(loc="lower left")
        plt.ylim([0, 1])
        self.format_plot(title = "Model performance as a function of months",subtitle = "Ethereum blockchain",title_y=self.barplot_title_y,xlabel="Month", ylabel = "Accuracy")
        
        if save:
            plt.savefig("{path}/month_plot_all_models".format(path=self.store_path))
        if show:
            plt.show()

    def train_k_nearest_neighbors(self,k=5):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.X_train, self.y_train)

    def train_optimal_k_nearest_neighbors(self,save=False,show=False):
        self.fig = plt.figure(figsize=self.barplot_figsize)
        # optimal knn is defined as the one with the highest accuracy for k=1:30
        n_neighbors = range(1,31)
        dims = [1,2,3,5,8,10]
        for dim in dims:
            knn_scores = []
            #load nft embeddings as array in X and categories in y
            X = torch.load(self.results_path + "/results/D" + str(dim) + "/nft_embeddings").detach().numpy()
            y = np.loadtxt(self.results_path + "/sparse_c.txt",dtype="str").reshape(-1,1)

            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

            # encode y
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train.ravel())
            y_test = encoder.fit_transform(y_test.ravel())

            for k in n_neighbors:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                knn_scores.append(knn.score(X_test, y_test))
                print(k)
            
            if save or show:
                plt.plot(n_neighbors,knn_scores,marker='o',mfc='black',markersize=self.markersize,linewidth=self.linewidth,label=f"Dim: {dim}")
        
        plt.legend(loc="lower right")
        plt.ylim([0,1])
        self.format_plot(title="K-nearest neighbors performance plot",subtitle=self.bmmname,title_y=self.barplot_title_y,xlabel="Number of neighbors",ylabel="Accuracy")
            
        if save:
            plt.savefig("{path}/knn_performance_plot_{mtype}".format(path=self.store_path,mtype=self.mtype,dim=self.dim))
        if show:
            plt.show()

        optimal_k = np.argmax(knn_scores) + 1
        self.train_k_nearest_neighbors(k=optimal_k)

    def get_k_nearest_neighbors_results(self,k=10):
        if self.knn is None:
            # train multinomial logistic regression
            self.train_k_nearest_neighbors(k=k)
        
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
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred,ax=ax,values_format = '')
        # set axis labels
        plt.xticks([0,1,2,3,4,5],self.encoder.classes_,rotation=45,fontsize=self.fontsize_ticks)
        plt.yticks([0,1,2,3,4,5],self.encoder.classes_,rotation=45,fontsize=self.fontsize_ticks)
        self.format_plot(title=title,subtitle=self.dataname,title_y=self.barplot_title_y,xlabel="Predicted label",ylabel="True label")

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
        print(self.encoder.classes_[majority_class])
        print("Majority class voting accuracy: {accuracy:0.2f}%".format(accuracy=accuracy))
        print("Number of misclassifications for majority voting: {nwrong} out of {ntotal}".format(nwrong=misclassifications,ntotal=len(self.y_test)))
    

    def make_baseline_model_performance_month_plot(self,save=False,show=False):
        self.fig = plt.figure(figsize=self.bigplot_figsize)
        months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05",
                  "2020-06", "2020-07", "2020-08", "2020-09", "2020-10",
                  "2020-11", "2020-12", "2021-01", "2021-02", "2021-03"]
        proportions = []
        for month in months:
            print(month)
            path = self.resultsbase + f"/{self.blockchain}/{month}"

            y = np.loadtxt(path + f"/{mtype}/sparse_c.txt",dtype="str")
            cc = Counter(y)
            proportion = max([cc[c] for c in np.unique(y)]) / len(y)
            proportions.append(proportion)


        plt.plot(range(len(months)), proportions, marker='o', mfc='black', markersize=self.markersize, lw=self.linewidth)

        plt.xticks(range(len(months)), months, rotation=45, fontsize=self.fontsize_ticks)
        plt.ylim([0, 1])
        self.format_plot(title = "Majority NFT class proportion as a function of months",subtitle = "Ethereum blockchain",title_y=self.barplot_title_y,xlabel="Month", ylabel = "Proportion")
        
        if save:
            plt.savefig("{path}/month_plot_baseline_model".format(path=self.store_path))
        if show:
            plt.show()



    def get_mcNemar_test(self):
        alpha = 0.05

        y = np.loadtxt(self.results_path + "/sparse_c.txt",dtype="str").reshape(-1,1)
        # encode y
        encoder = LabelEncoder()
        y = encoder.fit_transform(y.ravel())

        path = "/".join(self.results_path.split("/")[:-1])

        X_bi = torch.load(path + "/bi" + "/results/D" + str(3) + "/nft_embeddings").detach().numpy()
        X_tri = torch.load(path + "/tri" + "/results/D" + str(3) + "/nft_embeddings").detach().numpy()

         # split data into train and test
        X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(X_bi, y, test_size=0.2, stratify=y, random_state=42)
        X_train_tri, X_test_tri, y_train_tri, y_test_tri = train_test_split(X_tri, y, test_size=0.2, stratify=y,
                                                                        random_state=42)

        print("check of train and test for bi and tri")
        print(sum(y_train_bi==y_train_tri) == len(y_train_bi))

        #### get KNN BI and TRI
        knn_bi = KNeighborsClassifier(n_neighbors=10)
        knn_bi.fit(X_train_bi, y_train_bi)
        y_pred_knn_bi = knn_bi.predict(X_test_bi)

        knn_tri = KNeighborsClassifier(n_neighbors=10)
        knn_tri.fit(X_train_tri, y_train_tri)
        y_pred_knn_tri = knn_tri.predict(X_test_tri)


        #### get MLR BI and TRI
        mlr_bi = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42)
        mlr_bi.fit(X_train_bi, y_train_bi)
        y_pred_mlr_bi = mlr_bi.predict(X_test_bi)

        mlr_tri = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42)
        mlr_tri.fit(X_train_tri, y_train_tri)
        y_pred_mlr_tri = mlr_tri.predict(X_test_tri)

        #### get baseline
        majority_class = np.argmax([sum(y_train_bi==c) for c in np.unique(list(y_train_bi))])
        y_pred_base = [majority_class] * len(y_test_bi)


        # baseline vs. knn_bi
        print("baseline vs. knn_bi")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_base, y_pred_knn_bi, alpha)

        # baseline vs. knn_tri
        print("baseline vs. knn_tri")
        m1, CI1, p1 = self.mcnemar(y_test_tri, y_pred_base, y_pred_knn_tri, alpha)  
      
        # baseline vs. mlr_bi
        print("baseline vs. mlr_bi")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_base, y_pred_mlr_bi, alpha)

        # baseline vs. mlr_tri
        print("baseline vs. mlr_tri")
        m1, CI1, p1 = self.mcnemar(y_test_tri, y_pred_base, y_pred_mlr_tri, alpha)

        # knn_bi vs. knn_tri
        print("knn_bi vs. knn_tri")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_knn_bi, y_pred_knn_tri, alpha)

        # knn_bi vs. mlr_bi
        print("knn_bi vs. mlr_bi")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_knn_bi, y_pred_mlr_bi, alpha)

        # knn_bi vs. mlr_tri 
        print("knn_bi vs. mlr_tri")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_knn_bi, y_pred_mlr_tri, alpha)

        # knn_tri vs. mlr_bi
        print("knn_tri vs. mlr_bi")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_knn_tri, y_pred_mlr_bi, alpha)

        # knn_tri vs. mlr_tri
        print("knn_tri vs. mlr_tri")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_knn_tri, y_pred_mlr_tri, alpha)

        # mlr_bi vs. mlr_tri
        print("mlr_bi vs. mlr_tri")
        m1, CI1, p1 = self.mcnemar(y_test_bi, y_pred_mlr_bi, y_pred_mlr_tri, alpha)

       
    
    def mcnemar(self, y_true, yhatA, yhatB, alpha=0.05): 
        # code taken from DTU course 02450 "Introduction to machine learning and data mining" code toolbox provided to students
        # perform McNemars test
        nn = np.zeros((2,2))
        c1 = yhatA - y_true == 0
        c2 = yhatB - y_true == 0

        nn[0,0] = sum(c1 & c2)
        nn[0,1] = sum(c1 & ~c2)
        nn[1,0] = sum(~c1 & c2)
        nn[1,1] = sum(~c1 & ~c2)

        n = sum(nn.flat);
        n12 = nn[0,1]
        n21 = nn[1,0]

        thetahat = (n12-n21)/n
        Etheta = thetahat

        Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

        p = (Etheta + 1)*0.5 * (Q-1)
        q = (1-Etheta)*0.5 * (Q-1)

        CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

        p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
        print("Result of McNemars test using alpha=", alpha)
        print("Comparison matrix n")
        print(nn)
        if n12+n21 <= 10:
            print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

        print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
        print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

        return thetahat, CI, p

        
# choose data set to investigate
blockchain="ETH"
month="2021-03"
mtypes=["bi"]
dims=[3]

for mtype in mtypes:
    for dim in dims:
        #print(mtype,dim)
        cp = ClassicationPlotter(blockchain=blockchain,month=month,mtype=mtype,dim=dim)
        #cp.make_baseline_model_performance_month_plot(save=True)
        #cp.get_multinomial_results()
        #cp.get_k_nearest_neighbors_results(k=10)
        #cp.train_optimal_k_nearest_neighbors(save=True)
        #cp.make_dim_plot_all(k=10,save=True)
        #cp.print_class_distribution()
        #cp.print_class_distribution()
        #cp.print_encoding_labels()
        #logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42)
        #logreg.fit(cp.X_train,cp.y_train)
        #print("Multinomial:",logreg.score(cp.X_test,cp.y_test))
        #knn = KNeighborsClassifier(n_neighbors=10)
        #knn.fit(cp.X_train,cp.y_train)
        #print("KNN:",knn.score(cp.X_test,cp.y_test))
        #cp.make_barplot_train(save=True)
        #cp.make_barplot_test(save=True)
        #cp.make_model_dim_plot(modeltype="multinomial",save=True)
        #cp.make_model_dim_plot(modeltype="KNN",k=10,save=True)
        #cp.make_confusion_matrix("multinomial",save=True)
        #cp.make_confusion_matrix("KNN",k=10,save=True)
        #cp.make_confusion_matrix("Optimal KNN",save=True)
        #cp.train_optimal_k_nearest_neighbors(save=True)
        #cp.print_baseline_model_performance()
        #cp.make_month_plot_all(k=10,save=True)
        cp.get_mcNemar_test()
