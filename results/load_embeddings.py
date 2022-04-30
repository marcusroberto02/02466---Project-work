import torch
import matplotlib.pyplot as plt

<<<<<<< HEAD
z = torch.load(r"C:\Users\marcu\Google Drev\DTU\02466(fagprojekt)\02466---Project-work\data\2017_11\results\bi\nft_embeddings")
q = torch.load(r"C:\Users\marcu\Google Drev\DTU\02466(fagprojekt)\02466---Project-work\data\2017_11\results\bi\trader_embeddings")

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
=======
nft_z = torch.load("../results/seller_embeddings")

print(nft_z)

>>>>>>> e30f33f3926495f13d64fbae40e48abdf98e3e6f
