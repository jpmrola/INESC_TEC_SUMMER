import torch
import numpy as np

# Tensor Examples

device = "cuda" if torch.cuda.is_available() else "cpu"
mytensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)

print(mytensor)

# Formas de declarar

a = torch.empty((3, 3))
a = torch.zeros((3, 3))
a = torch.rand(3, 3)
a = torch.ones(3, 3)
a = torch.eye(3, 3)
a = torch.arange(start=0, end=3, step=1)
a = torch.linspace(start=0, end=3, steps=10)
a = torch.empty((3, 3)).normal_(mean=0, std=1)
a = torch.empty((3,3)).uniform_(0, 3)   # mesmo que torch.rand
a = torch.diag(torch.ones(3))

# Typecasting

a.short()

# numpy para pytorch

np_array = np.zeros((5, 5))
np_tensor = torch.from_numpy(np_array)
np_array_from_tensor = np_tensor.numpy()

# Operações com tensors

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.empty(3)

torch.add(x, y, out=z)
# ou
z = torch.add(x, y)
# ou
z = x + y

# inplace

t = torch.zeros(3)
t.add_(x)

# potencias

z = x.pow(3)

# multiplicação de matrizes

x1 = torch.rand(3, 5)
x2 = torch.rand(5, 3)
x3 = torch.mm(x1, x2)
x4 = x3.matrix_power(2)

print(z)
print(t)
print(x3)