import torch
import math
import torch.nn.functional as F
from torch import nn
from lrt_tensor import write_tensor_dict, write_lrt_tensor
from torch.nn import MultiheadAttention

def export_gpt2():
    from transformers import GPT2Model
    model = GPT2Model.from_pretrained("gpt2")

    d = {}
    d['wte'] = model.wte.weight
    d['wpe'] = model.wpe.weight
    write_tensor_dict(d, 'gpt2.params.bin')

    with open("gpt2.config.ini", 'w') as fp:
        fp.write("[model]\n")
        fp.write("params_file=gpt2.params.bin\n")
        fp.write("[config]\n")
        fp.write(f"n_embd={model.config.n_embd}\n") 
        fp.write(f"n_ctx={model.config.n_ctx}\n") 
        fp.write(f"vocab_size={model.config.vocab_size}\n") 

if __name__ == '__main__':
    export_gpt2()
