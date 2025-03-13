# Starting
import torch

a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])

# Element-wise addition
print('Element Wise Addition of a & b: \n', a + b)

# Matrix multiplication
print('Matrix Multiplication of a & b: \n', 
      torch.matmul(a.view(2, 1), b.view(1, 2)))
