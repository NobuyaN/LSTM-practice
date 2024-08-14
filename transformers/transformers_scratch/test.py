import torch
torch.random.manual_seed(1234)
a = torch.rand((10,3))
print(a)

print(a[:, 0::2])
