import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
from imblearn.over_sampling import RandomOverSampler

#nft_embeddings_path = "C:/Users/khelp/OneDrive/Desktop/4. semester/Fagprojekt/02466---Project-work/data/bi/nft_embeddings"
path = "./data/ETH/2020-10"


#################################################################
'''
Node classification

'''
#################################################################

#load nft embeddings as array in X and categories in y
X = torch.load(path + "/bi/results/D2/nft_embeddings").detach().numpy()
y = np.loadtxt(path + "/bi/train/sparse_c.txt",dtype="str").reshape(-1,1)
oversample = RandomOverSampler(sampling_strategy="not majority")
X,y = oversample.fit_resample(X,y)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y,random_state=42)

# print distribution of plots
print([sum(y_test == c) / len(y_test) for c in np.unique(y_test)])

# encode y
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

print(encoder.classes_)

#Multinomial logistic regression
logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
logreg.fit(X_train, y_train)

#Metrics
print('Number of miss-classifications for Multinomial regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test), len(y_test)))
print('Accuracy for Multinomial regression:\n\t {0}'.format(logreg.score(X_test, y_test)))
print('Confusion matrix for Multinomial regression:\n\t {0}'.format(confusion_matrix(y_test, logreg.predict(X_test))))
#print('ROC AUC for Multinomial regression:\n\t {0}'.format(roc_auc_score(y_test, logreg.predict(X_test))))
#print('Precision for Multinomial regression:\n\t {0}'.format(precision_score(y_test, logreg.predict(X_test))))
#print('Recall for Multinomial regression:\n\t {0}'.format(recall_score(y_test, logreg.predict(X_test))))


#KNN
neighbours = range(1,31)
knn_scores = []

for n in neighbours:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    knn_scores.append(knn.score(X_test, y_test))

plt.plot(neighbours,knn_scores)
plt.xlabel("Number of neighbours")
plt.ylabel("Accuracy")
plt.title("KNN performance on the bipartite model")
plt.show()

#Metrics
#print('Number of miss-classifications for KNN:\n\t {0} out of {1}'.format(np.sum(knn.predict(X_test)!=y_test), len(y_test)))
#print('Accuracy for KNN:\n\t {0}'.format(knn.score(X_test, y_test)))
#print('Confusion matrix for KNN:\n\t {0}'.format(confusion_matrix(y_test, knn.predict(X_test))))
#print('ROC AUC for KNN:\n\t {0}'.format(roc_auc_score(y_test, knn.predict(X_test))))
#print('Precision for KNN:\n\t {0}'.format(precision_score(y_test, knn.predict(X_test))))
#print('Recall for KNN:\n\t {0}'.format(recall_score(y_test, knn.predict(X_test))))




#################################################################
'''
Clustering

'''
#################################################################

kmeans = KMeans(n_clusters=6, random_state=0).fit(X)

label = kmeans.labels_

u_labels = np.unique(label)

fig = plt.figure(figsize = (12,12))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
#fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

for i in u_labels:
    ax1.scatter(X[label == i, 0], X[label == i, 1], s=0.1, label='cluster %d' % i)
#axs[0].legend()
ax1.set_title("Kmeans")

colors = {'Art':'green', 'Collectible':'blue', 'Games':'red','Metaverse':'orange','Other':'purple','Utility':'brown'}

for category, color in colors.items():
    ax2.scatter(*zip(*X[y==category][:,:2]),s=0.1,c=color,label=category)
ax2.legend(loc="upper right", markerscale=15)
ax2.set_title("True classification - Bipartite model")

plt.show()




# 3d scatterplot using plotly
"""
Scene = dict(xaxis = dict(title  = 'x -->'),yaxis = dict(title  = 'y--->'),zaxis = dict(title  = 'z-->'))

# model.labels_ is nothing but the predicted clusters i.e y_clusters
labels = encoder.fit_transform(y)
trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()
"""

