import torch
import math
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



class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        print('Q2')
        print(q)
        print('K2')
        print(k)
        print('V2')
        print(v)

        scores = attention(q, k, v, self.d_k, mask)
        print('attention')
        print(scores)
        concat = scores.transpose(1,2)
        print('attention T')
        print(concat)
        concat = concat.contiguous().view(bs, -1, self.d_model)
        print('concat')
        print(concat)
        output = self.out(concat)
        return output

def attention(q, k, v, d_k, mask=None,):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    print('scores')
    print(scores)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    print('softmax')
    print(scores)
    output = torch.matmul(scores, v)
    return output

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
        print('Q')
        print(q)
        print('K')
        print(k)
        print('V')
        print(v)

        write_lrt_tensor(q, fp)
        write_lrt_tensor(k, fp)
        write_lrt_tensor(v, fp)

        o, _ = layer(q, k, v)
        write_lrt_tensor(o, fp)

        # ref
        layer_ = MultiHeadAttention(2, D_MODEL0)
        with torch.no_grad():
            layer_.q_linear.weight.copy_(d['q_proj.weight'])
            layer_.q_linear.bias.copy_(d['q_proj.bias'])
            layer_.k_linear.weight.copy_(d['k_proj.weight'])
            layer_.k_linear.bias.copy_(d['k_proj.bias'])
            layer_.v_linear.weight.copy_(d['v_proj.weight'])
            layer_.v_linear.bias.copy_(d['v_proj.bias'])
            layer_.out.weight.copy_(d['out_proj.weight'])
            layer_.out.bias.copy_(d['out_proj.bias'])
        o_ref = layer_(q, k, v)
        print(o)
        print(o_ref)

if __name__ == '__main__':
    gen_multi_head_attention()
