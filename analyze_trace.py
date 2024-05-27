import numpy as np
import re
import torch

def custom_formatter(x):
    return "{:4d}".format(x)

np.set_printoptions(linewidth=2000, formatter={'all': custom_formatter})
torch.set_printoptions(linewidth=2000)
score_path = "./reduced.scores.out"

spars_path = "./reduced.scores.5.out"

def extract_numbers(tensor_str):
    # 正则表达式匹配浮点数
    pattern = r"[-+]?\d*\.\d+|\d+"
    numbers = re.findall(pattern, tensor_str)
    # return list(map(float, numbers))
    return np.array(numbers, dtype=float)

f1 = open(score_path, "r")
f2 = open(spars_path, "r")

baseline_lines = f1.readlines()
# baseline_lines = baseline_lines[8:]
sparsity_lines = f2.readlines()
# sparsity_lines = sparsity_lines[8:]

assert len(sparsity_lines) == len(baseline_lines)

sparsity = 0.5
prompt_num = 8
for token_id in range(prompt_num, len(sparsity_lines)):
    token_num = token_id
    n_global_token = int(token_num * sparsity) // 2
    n_local_token  = int(token_num * sparsity) - n_global_token
    numbers = extract_numbers(baseline_lines[token_id])
    sorted_indices = np.argsort(numbers)
    rankings = [sorted_indices.tolist().index(i) for i in range(len(numbers))]
    rankings = torch.tensor(rankings)
    cpu_tokens = torch.topk(rankings[: token_num - n_local_token], token_num - n_local_token - n_global_token, largest=False).indices
    # print(rankings)
    # rankings = ['{:.4f}'.format(val / len(numbers)) for val in rankings]
    
    baseline = torch.ones(token_num + 1, dtype=bool)
    baseline[cpu_tokens] = 0
    # print(baseline)
    sparsity_numbers = extract_numbers(sparsity_lines[token_id])
    sparsity_numbers = torch.tensor(sparsity_numbers)
    # print(sparsity_lines[token_id])
    # print(sparsity_numbers.bool())
    print(baseline == sparsity_numbers.bool())


f1.close()
f2.close()