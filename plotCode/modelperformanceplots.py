import numpy as np
import torch
import matplotlib.pyplot as plt

path = "./results_final/ETH/2021-02"

specific = "/tri/results/D2"

link_prediction_acc = np.loadtxt(path + specific + "/max_accuracy_train.txt")
baseline_acc = np.loadtxt(path + specific + "/baseline_accuracy_train.txt")


def accuracy_plot(link_prediction_acc, baseline_acc):
    epochs_link,acc_link = link_prediction_acc[:,0],link_prediction_acc[:,1]
    epochs_base,acc_base = baseline_acc[:,0],baseline_acc[:,1]

    plt.plot(epochs_link,acc_link, 'g', label="link prediction accuracy")
    plt.plot(epochs_base, acc_base, 'b', label="baseline model accuracy")
    plt.title('Accuracy as a function of epochs for baseline model and link prediction')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


accuracy_plot(link_prediction_acc,baseline_acc)
