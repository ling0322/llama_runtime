import torch
from torch import nn
from lrt_tensor import write_tensor_dict, write_lrt_tensor


D_INPUT = 16
D_OUTPUT = 20

layer = nn.Linear(D_INPUT, D_OUTPUT)
d = {}
d['weight'] = layer.weight
d['bias'] = layer.bias

write_tensor_dict(d, 'linear-model.params.bin')

with open('linear-model.test_tensors.bin', 'wb') as fp:
    a = torch.rand(2, D_INPUT)
    b = layer(a)
    write_lrt_tensor(a, fp)
    write_lrt_tensor(b, fp)

    a = torch.rand(2, 3, D_INPUT)
    b = layer(a)
    write_lrt_tensor(a, fp)
    write_lrt_tensor(b, fp)

    a = torch.rand(2, 3, 4, D_INPUT)
    b = layer(a)
    write_lrt_tensor(a, fp)
    write_lrt_tensor(b, fp)
