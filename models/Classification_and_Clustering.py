
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset_path='C:/Users/khelp/OneDrive/Desktop/4. semester/Fagprojekt/02466---Project-work/data/sparse_tri'

results_path = 'C:/Users/khelp/OneDrive/Desktop/4. semester/Fagprojekt/02466---Project-work/results'

#################################################################
'''
Node classification

'''
#################################################################

#load nft embeddings as array
nft_embeddings=torch.load(results_path + "/nft_embeddings")

#convert nft_embeddings to numpy array
nft_embeddings = nft_embeddings.numpy()
X = nft_embeddings
y = np.loadtxt(dataset_path + "/sparse_v.txt")

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

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
