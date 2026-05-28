import torch
import numpy as np

# Initialise tensor from regular 2D array
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# Initialise tensor from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data) # retains properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# Tensor Attributes
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}") # torch.Size([3,4])
print(f"Datatype of tensor: {tensor.dtype}") # torch.float32
print(f"Device tensor is stored on: {tensor.device}") # cpu

# Move tensor to GPU if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")

# indexing and slicing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# concatenating
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Multiplying
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n") # This computes the element-wise product
print(f"tensor * tensor \n {tensor * tensor}") # Alternative syntax:

# Matrix mul
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations: Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x.
print(tensor, "\n")
tensor.add_(5)
print(tensor)


# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
