import torch
import math
import torch.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../bloom-560m")
model = AutoModelForCausalLM.from_pretrained("../bloom-560m")
inputs = tokenizer("CCP is evil or good? ", return_tensors="pt").input_ids
outputs = model.generate(inputs)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(output_text)
