import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
import numpy as np

x = np.linspace(1,7,1000)

plt.plot(x,norm.pdf(x,4,0.4),'r-',lw=1)
plt.plot(x,norm.pdf(x,5,0.3),'b-',lw=1)
# plot points
npoints = 30
plt.plot(np.random.normal(4,0.4,npoints),[0 for _ in range(npoints)],'o',color="red")
plt.plot(np.random.normal(5,0.3,npoints),[0 for _ in range(npoints)],'x',color="blue")
plt.xlabel(r"$\lambda_{ij}=\mathregular{exp}(\gamma_i+\delta_j-\|\|\mathbf{z}_i-\mathbf{q}_j\|\|)$")
plt.ylabel(r"$\mathregular{Density}$")
# vertical line
plt.vlines(4.54,0,2,color="black")
# markers
red_circle = Line2D([0], [0], marker='o', color='r', label=r"Negative, $\tilde{y}_{ij}=0$",
                        markerfacecolor='r', markersize=10,lw=0)
blue_cross = Line2D([0], [0], marker='x', color='b', label=r"Positive, $\tilde{y}_{ij}=1$",
                        markerfacecolor='b', markersize=10,lw=0)

# add text
plt.text(5.14,1.47,r"$\theta$",fontsize=12)
plt.annotate("", xy = (4.54, 1.5), xytext = (5.04,1.5), 
             arrowprops = dict(facecolor = 'black',width=1,headwidth=5),
             color = 'black')
plt.text(2.1,0.3,r"Labeled as negative",fontsize=12,rotation=-90)
plt.text(6.6,0.3,r"Labeled as positive",fontsize=12,rotation=-90)
plt.legend(handles=[red_circle,blue_cross],loc="upper left")
plt.show()

