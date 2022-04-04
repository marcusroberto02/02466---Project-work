import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

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