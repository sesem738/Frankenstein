import numpy as np
import math
import torch
from time import time

if torch.cuda.is_available():
    print("GPU available")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
def inv_freq_standard(n):
    return 1.0 / (10000 ** (torch.arange(0.0, n, 2.0) / n))

def inv_freq_exp(n):
    return torch.exp(torch.arange(0, n, 2) * (-math.log(10000.0) / n))

def inv_freq_exp_torch(n):
    return torch.exp(torch.arange(0, n, 2) * (-torch.log(torch.Tensor([10000.0])) / n))

# timeit function
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time()
        print("Time: {}".format(end - start))
        return res
    return wrapper

timeit(inv_freq_exp)(10000)

print("inv_freq_exp")
for _ in range(10):
    timeit(inv_freq_exp)(10000)

print("inv_freq_exp_torch")
for _ in range(10):
    timeit(inv_freq_exp_torch)(10000)

print("inv_freq_standard")
for _ in range(10):
    timeit(inv_freq_standard)(10000)