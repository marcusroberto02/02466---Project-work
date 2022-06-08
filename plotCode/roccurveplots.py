import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def score_epoch(mtype, score_type):

    epochs = np.arange(0,301)*100
    roc = np.loadtxt("./data/ETH/2020-10/" + mtype + "/results/D9/"+score_type + "_train.txt")
    
    plt.title(score_type + " scores as a function of epochs - " + mtype + "partite model")
    plt.ylabel(score_type + " score at given epoch")
    plt.xlabel("Nr of epochs")
    plt.plot(epochs, roc, '-k', markersize= 2)
    plt.show()

score_epoch("Bi","ROC")
score_epoch("Bi","PR")
score_epoch("Tri","ROC")
score_epoch("Tri","PR")




def roc_plot():
    x = np.array([-10,-9,-8,-7,-6,6,7,8,9,10],dtype=float)
    y = np.array([0,0,0,1,1,0,0,1,1,1],dtype=float)

    precision, recall, _ = metrics.precision_recall_curve(y,x)
    fpr, tpr, _ = metrics.roc_curve(y,x)

    #plt.step(recall, precision, where='post')
    #plt.ylim([0.0, 1.05])
    #plt.xlim([0.0, 1.0])
    #plt.show()

    # plot roc curve
    x = np.linspace(0,1,100)
    plt.plot(x,x,"k--")
    plt.step(fpr, tpr, where='post',color="k")
    plt.ylim([0.0, 1.05])
    plt.xlim([-0.05, 1])
    plt.show()