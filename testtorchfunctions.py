import torch

t1 = torch.randint(1,7,(10,))
t2 = torch.randint(1,7,(10,))

test = torch.zeros((10,10))

for i,j in zip(t1,t2):
    test[i,j] = 1

print(test)