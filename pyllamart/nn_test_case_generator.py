import torch
from torch import nn
from lrt_tensor import write_tensor_dict, write_lrt_tensor


D_MODEL = 16

layer = nn.Linear(D_MODEL, D_MODEL)
d = {}
d['weight'] = layer.weight
d['bias'] = layer.bias

write_tensor_dict(d, 'linear-model.params.bin')

with open('linear-model.test_tensors.bin', 'wb') as fp:
    a = torch.rand(D_MODEL, D_MODEL)
    b = layer(a)

    print(a)
    print(b)
    write_lrt_tensor(a, fp)
    write_lrt_tensor(b, fp)
