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
from sklearn.metrics import plot_confusion_matrix


#nft_embeddings_path = "C:/Users/khelp/OneDrive/Desktop/4. semester/Fagprojekt/02466---Project-work/data/bi/nft_embeddings"
path = r"C:\Users\khelp\OneDrive\Documents\GitHub\02466---Project-work\results_final\ETH\2021-02"


#################################################################
'''
Node classification

'''
#################################################################

#load nft embeddings as array in X and categories in y
X = torch.load(path + "/tri/results/D2/nft_embeddings").detach().numpy()
y = np.loadtxt(path + "/tri/sparse_c.txt",dtype="str").reshape(-1,1)


# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y,random_state=42)

# print distribution of plots
print([sum(y_test == c) / len(y_test) for c in np.unique(y_test)])

# encode y
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

print(encoder.classes_)

fontsize = 20
fontsize_title = 22
fontsize_ticks = 18

# plot class distribution train and test
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fig, axes = plt.subplots()
plt.bar(np.unique(list(y_train)), height=[sum(y_train==c) for c in np.unique(list(y_train))])
plt.xticks([0,1,2,3,4,5],encoder.classes_,rotation=45, fontsize=fontsize_ticks)
plt.ylabel('Count', fontsize=fontsize, weight='bold')
plt.xlabel('Category', fontsize =fontsize, weight='bold')
plt.title('Barplot of categories in the training data set', fontsize=fontsize_title, weight='bold')
plt.show()

fig, axes = plt.subplots()
plt.bar(np.unique(list(y_test)), height=[sum(y_test==c) for c in np.unique(list(y_test))])
plt.xticks([0,1,2,3,4,5],encoder.classes_,rotation=45,fontsize=fontsize_ticks)
plt.ylabel('Count', fontsize=fontsize, weight='bold')
plt.xlabel('Category', fontsize =fontsize, weight='bold')
plt.title('Barplot of categories in the test data set', fontsize=fontsize_title, weight='bold')
plt.show()

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

# plot confusion matrix Multinomial logistic regression
plot_confusion_matrix(logreg, X_test, y_test)
plt.title('Confusion matrix - Multinomial logistic regression - tripartite 3D', fontsize=fontsize_title, weight='bold')
plt.xticks([0,1,2,3,4,5],encoder.classes_,rotation=45,fontsize=fontsize_ticks)
plt.yticks([0,1,2,3,4,5],encoder.classes_,rotation=45,fontsize=fontsize_ticks)
plt.xlabel('Predicted label', fontsize=fontsize, weight='bold')
plt.ylabel('True label', fontsize=fontsize,weight='bold')
plt.show()


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
plt.title("KNN accuracy as a function of neighbors - tripartite 3D")
plt.show()

#Metrics
optimal_n = np.argmax(knn_scores) + 1
knn = KNeighborsClassifier(n_neighbors=optimal_n)
knn.fit(X_train, y_train)
print('Number of miss-classifications for KNN:\n\t {0} out of {1}'.format(np.sum(knn.predict(X_test)!=y_test), len(y_test)))
print('Accuracy for KNN:\n\t {0}'.format(knn.score(X_test, y_test)))
print('Confusion matrix for KNN:\n\t {0}'.format(confusion_matrix(y_test, knn.predict(X_test))))
#print('ROC AUC for KNN:\n\t {0}'.format(roc_auc_score(y_test, knn.predict(X_test))))
#print('Precision for KNN:\n\t {0}'.format(precision_score(y_test, knn.predict(X_test))))
#print('Recall for KNN:\n\t {0}'.format(recall_score(y_test, knn.predict(X_test))))

# plot confusion matrix knn
plot_confusion_matrix(knn, X_test, y_test)
plt.title('Confusion matrix - KNN - tripartite 3D', fontsize=fontsize_title, weight='bold')
plt.xticks([0,1,2,3,4,5],encoder.classes_,rotation=45,fontsize=fontsize_ticks)
plt.yticks([0,1,2,3,4,5],encoder.classes_,rotation=45,fontsize=fontsize_ticks)
plt.xlabel('Predicted label', fontsize=fontsize, weight='bold')
plt.ylabel('True label', fontsize=fontsize,weight='bold')
plt.show()


# baseline model 
majority_class = np.argmax([sum(y_train==c) for c in np.unique(list(y_train))])
y_pred = [majority_class] * len(y_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Majority class voting accuracy:",accuracy)
misclassifications = np.sum(y_pred!=y_test)
print("number of misclassifications majorityvoting:",misclassifications)


