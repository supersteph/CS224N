from highway import Highway
import torch
x=Highway(3)
print(x(torch.tensor([[0.1,0.2,0.3]])))