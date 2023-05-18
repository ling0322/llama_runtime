import torch
import math
import torch.nn.functional as F
from typing import Dict
from torch import nn, Tensor
from lrt_tensor import write_tensor_dict

import transformers.models.gpt2.modeling_gpt2 as gpt2

class Context:
    """same as nn::Context, stores the context of current module"""
    def __init__(self, name="") -> None:
        self._ns = name

    def name(self, name: str) -> str:
        """get name under current namespace"""
        return name if not self._ns else self._ns + '.' + name
    
    def with_name(self, name: str) -> 'Context':
        """get the context object with a sub-namespace"""
        ctx = Context()
        ctx._ns = self.name(name)
        return ctx

class ModelExporter:
    _d: Dict[str, Tensor]

    def __init__(self) -> None:
        self._d = {}

    def _put(self, ctx: Context, name: str, tensor: Tensor) -> None:
        self._d[ctx.name(name)] = tensor

    def export_layer_norm(self, ctx: Context, module: nn.LayerNorm):
        self._put(ctx, "weight", module.weight)
        self._put(ctx, "bias", module.bias)

class GPT2Exporter(ModelExporter):
    """exporter for GPT2 model from transformers"""

    def __init__(self) -> None:
        super().__init__()

    def export_conv1d(self, ctx: Context, module: gpt2.Conv1D):
        self._put(ctx, "weight", module.weight.T)
        self._put(ctx, "bias", module.bias)

    def export_attn(self, ctx: Context, module: gpt2.GPT2Attention):
        weight = module.c_attn.weight.T
        bias = module.c_attn.bias
        d_model = weight.shape[1]
        for i, name in enumerate(['q_proj', 'k_proj', 'v_proj']):
            offset = i * d_model
            proj_ctx = ctx.with_name(name)
            self._put(proj_ctx, "weight", weight[offset : offset + d_model])
            self._put(proj_ctx, "bias", bias[offset : offset + d_model])

        weight = module.c_proj.weight.T
        bias = module.c_proj.bias
        proj_ctx = ctx.with_name('out_proj')
        self._put(proj_ctx, "weight", weight)
        self._put(proj_ctx, "bias", bias)
    
    def export_decoder_layer(self, ctx: Context, module: gpt2.GPT2Block):
        self.export_layer_norm(ctx.with_name("ln1"), module.ln_1)
        self.export_layer_norm(ctx.with_name("ln2"), module.ln_2)
        self.export_attn(ctx.with_name("attn"), module.attn)
        self.export_conv1d(ctx.with_name("fc"), module.mlp.c_fc)
        self.export_conv1d(ctx.with_name("proj"), module.mlp.c_proj)


    def export_gpt2(self, model_name="gpt2"):
        from transformers import GPT2Model
        model = GPT2Model.from_pretrained(model_name)
        ctx = Context("gpt2")

        self._put(ctx, "wte", model.wte.weight)
        self._put(ctx, "wpe", model.wpe.weight)

        for layer_idx, layer in enumerate(model.h):
            self.export_decoder_layer(ctx.with_name(f"block{layer_idx}"), layer)

        write_tensor_dict(self._d, 'gpt2.params.bin')

        n_inner = model.config.n_inner if model.config.n_inner else model.config.hidden_size * 4
        with open("gpt2.config.ini", 'w') as fp:
            fp.write("[model]\n")
            fp.write("type=gpt2\n")
            fp.write("params_file=gpt2.params.bin\n")
            fp.write("\n")
            fp.write("[config]\n")
            fp.write(f"n_embd={model.config.n_embd}\n") 
            fp.write(f"n_ctx={model.config.n_ctx}\n") 
            fp.write(f"n_inner={n_inner}\n") 
            fp.write(f"n_head={model.config.n_head}\n") 
            fp.write(f"vocab_size={model.config.vocab_size}\n")
            fp.write(f"hidden_size={model.config.hidden_size}\n")

if __name__ == '__main__':
    GPT2Exporter().export_gpt2()
