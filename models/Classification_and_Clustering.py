
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#nft_embeddings_path = "C:/Users/khelp/OneDrive/Desktop/4. semester/Fagprojekt/02466---Project-work/data/bi/nft_embeddings"
path = "./data/ETH/2020-09"






#################################################################
'''
Node classification

'''
#################################################################

#load nft embeddings as array in X and categories in y
X = torch.load(path + "/results/bi/nft_embeddings").detach().numpy()
y = np.loadtxt(path + "/train/bi/sparse_c.txt",dtype="str").reshape(-1,1)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# encode y
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

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
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

#Metrics
print('Number of miss-classifications for KNN:\n\t {0} out of {1}'.format(np.sum(knn.predict(X_test)!=y_test), len(y_test)))
print('Accuracy for KNN:\n\t {0}'.format(knn.score(X_test, y_test)))
print('Confusion matrix for KNN:\n\t {0}'.format(confusion_matrix(y_test, knn.predict(X_test))))
#print('ROC AUC for KNN:\n\t {0}'.format(roc_auc_score(y_test, knn.predict(X_test))))
#print('Precision for KNN:\n\t {0}'.format(precision_score(y_test, knn.predict(X_test))))
#print('Recall for KNN:\n\t {0}'.format(recall_score(y_test, knn.predict(X_test))))

print([sum(y_test == c) / len(y_test) for c in np.unique(y_test)])


#################################################################
'''
Clustering

'''
#################################################################

kmeans = KMeans(n_clusters=6, random_state=0).fit(X)

label = kmeans.labels_

u_labels = np.unique(label)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

for i in u_labels:
    axs[0].scatter(X[label == i, 0], X[label == i, 1], s=1, label='cluster %d' % i)
axs[0].legend()
axs[0].set_title("Kmeans")

colors = {'Games':'red', 'Art':'green', 'Collectible':'blue', 'Metaverse':'orange','Other':'purple','Utility':'brown'}

categories = list(np.loadtxt(path + "/train/bi/sparse_c.txt",dtype='str'))
df = pd.DataFrame(dict(categories = categories))
axs[1].scatter(*zip(*X[:,:2]),s=0.1,c=df["categories"].map(colors))
axs[1].set_title("True classification")

plt.show()
