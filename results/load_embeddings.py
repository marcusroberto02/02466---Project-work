import torch
import matplotlib.pyplot as plt


#bipartite
'''

# nft
z = z.detach().numpy()
zx = [el[0] for el in z]
zy = [el[1] for el in z]
plt.scatter(zx,zy,s=1,color="blue")
# trader
q = q.detach().numpy()
qx = [el[0] for el in q]
qy = [el[1] for el in q]
plt.scatter(qx,qy,s=1,color="red")
plt.show()

print(len(z))
print(len(q))

nft_z = torch.load("../results/seller_embeddings")

print(nft_z)
'''

l = torch.load('./data/ETH/2020-10/results/tri/nft_embeddings')
r = torch.load('./data/ETH/2020-10/results/tri/seller_embeddings')
u = torch.load('./data/ETH/2020-10/results/tri/buyer_embeddings')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# nft
l = l.detach().numpy()[:1000]
lx = [el[0] for el in l]
ly = [el[1] for el in l]
lz = [el[2] for el in l]
ax.scatter(lx,ly,lz,s=0.1,color="red")

# seller
r = r.detach().numpy()[:1000]
rx = [el[0] for el in r]
ry = [el[1] for el in r]
rz = [el[2] for el in r]
ax.scatter(rx,ry,rz,s=0.1,color="green")

# buyer
u = u.detach().numpy()[:1000]
ux = [el[0] for el in u]
uy = [el[1] for el in u]
uz = [el[2] for el in u]
ax.scatter(ux,uy,uz,s=0.1,color="blue")

#plt.ion()
plt.show()

print(len(l))
print(len(r))
print(len(u))
"""
# nft
l = l.detach().numpy()
lx = [el[0] for el in l]
ly = [el[1] for el in l]
plt.scatter(lx,ly,s=0.1,color="red")

# seller
r = r.detach().numpy()
rx = [el[0] for el in r]
ry = [el[1] for el in r]
plt.scatter(rx,ry,s=0.1,color="green")

# buyer
u = u.detach().numpy()
ux = [el[0] for el in u]
uy = [el[1] for el in u]
plt.scatter(ux,uy,s=0.1,color="blue")

#plt.ion()
#plt.show()
#plt.ioff()
plt.show()
"""
