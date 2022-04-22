import torch

# NFT items
N = 200
# Sellers
S = 1000
# Buyers
B = 500

# Latent dimension
D = 2

# bias placeholders
nft_bias = torch.randn(N)
seller_bias = torch.randn(S)
buyer_bias = torch.randn(B)

# embedding placeholders
nft_z = torch.randn(N,D)
seller_z = torch.randn(S,D)
buyer_z = torch.randn(B,D)

# Distance matrix for sellers and NFT items
# dimension S by N
d_sn = torch.cdist(seller_z+1e-06,nft_z,p=2)+1e-06

# Distance matrix for buyers and NFT items
# dimension B by N
d_bn = torch.cdist(buyer_z+1e-06,nft_z,p=2)+1e-06

# nft item and seller non link part
# exp(seller_bias-||nft_z-seller_z||)
# dimension S by N
non_link_sn = torch.exp(seller_bias.unsqueeze(1)-d_sn)

# nft item and buyer non link part
# exp(nft_bias+buyer_bias-||nft_z-seller_z||)
# dimension N by B
non_link_bn = torch.exp(nft_bias+buyer_bias.unsqueeze(1)-d_bn)

# total non link matrix N by S by B
total_non_link = (non_link_bn.unsqueeze(1) * non_link_sn).T

"""
# sanity check for a random triad
correct = 0

for i in range(N):
    for j in range(S):
        for k in range(B):
            correct += (total_non_link[i,j,k]-torch.exp(nft_bias[i]+seller_bias[j]+buyer_bias[k]-d_sn[j,i]-d_bn[k,i])) < 10e-5
print(correct)
print(N*S*B)
print(correct == N*S*B)
"""

print(torch.sum(total_non_link))


