# Starting
import torch

a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])

# Element-wise addition
print('Element Wise Addition of a & b: \n', a + b)

# Matrix multiplication
print('Matrix Multiplication of a & b: \n', 
      torch.matmul(a.view(2, 1), b.view(1, 2)))


# Tensor 

# Scalar
scalar = torch.tensor(7)
print(scalar, scalar.ndim, scalar.item())

# Vector
vector = torch.tensor([7, 7])
print(vector, vector.ndim, vector.shape)


# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
print(MATRIX, MATRIX.ndim, MATRIX.shape)
# tensor([[ 7,  8],
#         [ 9, 10]]) 2 torch.Size([2, 2])

# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR, TENSOR.ndim, TENSOR.shape)
# shape: 1 dimension of 3 by 3
# tensor([[[1, 2, 3],
#          [3, 6, 9],
#          [2, 4, 5]]]) 3 torch.Size([1, 3, 3])

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)
# tensor([[0.7250, 0.0431, 0.6012, 0.2024],
#        [0.9579, 0.3952, 0.4195, 0.9615],
#        [0.4463, 0.3111, 0.5499, 0.4141]]) torch.float32

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor, random_image_size_tensor.shape, random_image_size_tensor.ndim)
# tensor([[[0.5631, 0.2775, 0.7231],
#          [0.8530, 0.7840, 0.7407],
#          [0.5692, 0.5122, 0.4169],
#          ...,
#          [0.8866, 0.8623, 0.7617],
#          [0.9167, 0.7838, 0.7799],
#          [0.3511, 0.4502, 0.4726]],

#         [[0.9326, 0.5984, 0.6399],
#          [0.3424, 0.0090, 0.3095],
#          [0.9360, 0.6277, 0.6725],
#          ...,
#          [0.3616, 0.7032, 0.2954],
#          [0.3390, 0.5987, 0.0226],
#          [0.1843, 0.8840, 0.0378]],

#         [[0.6259, 0.1477, 0.7393],
#          [0.5227, 0.8768, 0.0727],
#          [0.4782, 0.0722, 0.9134],
#          ...,
#          [0.9560, 0.0399, 0.2963],
#          [0.3540, 0.1936, 0.9913],
#          [0.6040, 0.8647, 0.1949]],

#         ...,

#         [[0.5829, 0.7774, 0.9512],
#          [0.2604, 0.4475, 0.9357],
#          [0.4567, 0.9996, 0.5250],
#          ...,
#          [0.3324, 0.4730, 0.5380],
#          [0.3571, 0.7965, 0.5342],
#          [0.4069, 0.5066, 0.5803]],

#         [[0.8557, 0.7414, 0.9137],
#          [0.3495, 0.0238, 0.9059],
#          [0.3188, 0.3550, 0.8231],
#          ...,
#          [0.5768, 0.3278, 0.8519],
#          [0.2937, 0.8389, 0.4406],
#          [0.1197, 0.1082, 0.8586]],

#         [[0.0155, 0.8476, 0.1680],
#          [0.5467, 0.3139, 0.6657],
#          [0.5364, 0.2897, 0.9836],
#          ...,
#          [0.7502, 0.7939, 0.2093],
#          [0.0244, 0.3555, 0.4466],
#          [0.6987, 0.8041, 0.3130]]]) torch.Size([224, 224, 3]) 3

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print (zeros, zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
ones, ones.dtype

# Use torch.arange(), torch.range() is deprecated 
#zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future
#print(zero_to_ten_deprecated)
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

#torch.matmul(tensor_A, tensor_B) # (this will error)
print(torch.mm(tensor_A, tensor_B.T))

# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")