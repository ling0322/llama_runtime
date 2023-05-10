import torch
import torch.nn.functional as F
from torch import nn
from lrt_tensor import write_tensor_dict, write_lrt_tensor
from torch.nn import MultiheadAttention

torch.manual_seed(0)

D_MODEL0 = 16
D_MODEL1 = 20
SEQ_LEN = 5
BATCH_SIZE = 2
NUM_HEADS = 2

def gen_linear():


    layer = nn.Linear(D_MODEL0, D_MODEL1)
    d = {}
    d['weight'] = layer.weight
    d['bias'] = layer.bias

    write_tensor_dict(d, 'linear-model.params.bin')

    with open('linear-model.test_tensors.bin', 'wb') as fp:
        a = torch.rand(2, D_MODEL0)
        b = layer(a)
        write_lrt_tensor(a, fp)
        write_lrt_tensor(b, fp)

        a = torch.rand(2, 3, D_MODEL0)
        b = layer(a)
        write_lrt_tensor(a, fp)
        write_lrt_tensor(b, fp)

        a = torch.rand(2, 3, 4, D_MODEL0)
        b = layer(a)
        write_lrt_tensor(a, fp)
        write_lrt_tensor(b, fp)

def gen_layer_norm():
    layer = nn.LayerNorm(D_MODEL0)
    layer.weight.data = torch.rand(D_MODEL0)
    layer.bias.data = torch.rand(D_MODEL0)

    d = {}
    d['weight'] = layer.weight
    d['bias'] = layer.bias
    write_tensor_dict(d, 'layer-norm-model.params.bin')

    with open('layer-norm-model.test_tensors.bin', 'wb') as fp:
        a = torch.rand(D_MODEL0)
        b = layer(a)
        write_lrt_tensor(a, fp)
        write_lrt_tensor(b, fp)

        a = torch.rand(2, D_MODEL0)
        b = layer(a)
        write_lrt_tensor(a, fp)
        write_lrt_tensor(b, fp)

        a = torch.rand(2, 3, D_MODEL0)
        b = layer(a)
        write_lrt_tensor(a, fp)
        write_lrt_tensor(b, fp)

def gen_multi_head_attention():
    layer = nn.MultiheadAttention(D_MODEL0, NUM_HEADS, batch_first=True)
    d = {}
    for i, name in enumerate(['q_proj', 'k_proj', 'v_proj']):
        offset = i * D_MODEL0
        d[name + ".weight"] = layer.in_proj_weight[offset : offset + D_MODEL0]
        d[name + ".bias"] = layer.in_proj_bias[offset : offset + D_MODEL0]

    d['out_proj.weight'] = layer.out_proj.weight
    d['out_proj.bias'] = layer.out_proj.bias

    write_tensor_dict(d, 'attn-model.params.bin')

    with open('attn-model.test_tensors.bin', 'wb') as fp:
        q = torch.rand(BATCH_SIZE, SEQ_LEN, D_MODEL0)
        k = torch.rand(BATCH_SIZE, SEQ_LEN, D_MODEL0)
        v = torch.rand(BATCH_SIZE, SEQ_LEN, D_MODEL0)

        write_lrt_tensor(q, fp)
        write_lrt_tensor(k, fp)
        write_lrt_tensor(v, fp)

        o, _ = layer(q, k, v)
        write_lrt_tensor(o, fp)

if __name__ == '__main__':
    gen_multi_head_attention()
    torch.Size([2,3,4])
