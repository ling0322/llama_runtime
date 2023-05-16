import torch
import math
import torch.nn.functional as F
from torch import nn
from lrt_tensor import write_tensor_dict, write_lrt_tensor
from torch.nn import MultiheadAttention

torch.manual_seed(0)

D_MODEL0 = 16
D_MODEL1 = 20
SEQ_LEN = 10
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
    layer = nn.MultiheadAttention(D_MODEL0, NUM_HEADS, batch_first=True,)
    d = {}
    for i, name in enumerate(['q_proj', 'k_proj', 'v_proj']):
        offset = i * D_MODEL0
        d[name + ".weight"] = layer.in_proj_weight[offset : offset + D_MODEL0]
        d[name + ".bias"] = layer.in_proj_bias[offset : offset + D_MODEL0]

    d['out_proj.weight'] = layer.out_proj.weight
    d['out_proj.bias'] = layer.out_proj.bias

    write_tensor_dict(d, 'self-attn.params.bin')

    with open('self-attn.test_tensors.bin', 'wb') as fp:
        inputs = torch.rand(BATCH_SIZE, SEQ_LEN, D_MODEL0)
        write_lrt_tensor(inputs, fp)

        mask = torch.logical_not(torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.bool)))
        print(mask)
        o, _ = layer(inputs, inputs, inputs, attn_mask=mask)
        print(o)
        write_lrt_tensor(o, fp)

def gen_gpt2():
    from transformers import GPT2Model, GPT2Tokenizer
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    d = {}
    d['wte'] = model.wte.weight
    write_tensor_dict(d, 'gpt2.params.bin')

    with open("gpt2.config.ini", 'w') as fp:
        fp.write("[model]\n")
        fp.write("params_file=gpt2.params.bin\n")
        fp.write("[config]\n")
        fp.write(f"d_model={model.config.n_embd}\n") 
        fp.write(f"vocab_size={model.config.vocab_size}\n") 

    with open('gpt2.test_tensors.bin', 'wb') as fp:
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text)["input_ids"]
        inputs = torch.tensor(inputs, dtype=torch.int64)
        write_lrt_tensor(inputs, fp)

if __name__ == '__main__':
    gen_gpt2()
