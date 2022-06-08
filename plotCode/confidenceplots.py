import matplotlib.pyplot as plt
import numpy as np

# get error bar size
def get_error_barsize(std,n):
    return 1.96*std/np.sqrt(n)

n = 1000

dimensions = range(1,11)
ROC_scores = []
ROC_errorbars = []
PR_scores = []
PR_errorbars = []

for d in dimensions:
    path = "./data/ETH/backup_2020-10/2020-10/bi/results/D" + str(d)
    with open(path + "/ROC-PR.txt", 'r') as f:
        roc_mean,roc_std,pr_mean,pr_std = [float(l.strip().split()[-1]) for l in f.readlines()]
        ROC_scores.append(roc_mean)
        ROC_errorbars.append(get_error_barsize(roc_std,n))
        PR_scores.append(pr_mean)
        PR_errorbars.append(get_error_barsize(pr_std,n))

# plot performance (measured as ROC and PR) as a function of dimensions
plt.errorbar(dimensions,ROC_scores,ROC_errorbars,capsize=10)
plt.xlabel("Embedding dimensions")
plt.ylabel("ROC score")
plt.title("ROC score plot with error bars - Tripartite model")
plt.show()

plt.errorbar(dimensions,PR_scores,PR_errorbars,capsize=10)
plt.xlabel("Embedding dimensions")
plt.ylabel("PR score")
plt.title("PR score plot with error bars - Tripartite model")
plt.show()