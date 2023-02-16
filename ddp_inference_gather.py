import torch
import numpy as np

gpu_all_result = list()
gpu0_result = torch.load("distributed_result/predictions_0.pt")
for item in gpu0_result:
    gpu_all_result.extend(item)
gpu1_result = torch.load("distributed_result/predictions_1.pt")
for item in gpu1_result:
    gpu_all_result.extend(item)
gpu2_result = torch.load("distributed_result/predictions_2.pt")
for item in gpu2_result:
    gpu_all_result.extend(item)
gpu3_result = torch.load("distributed_result/predictions_3.pt")
for item in gpu3_result:
    gpu_all_result.extend(item)
gpu_all_result = sorted(gpu_all_result, key=lambda x: x[0])
all_result = list()
for item in gpu_all_result:
    # item is (index, torch([int]))
    all_result.append(item[1].detach()[0])
test = torch.stack(all_result)
# test is torch([all result of int])
np.savetxt("deepspeed_all_result.txt", test.numpy())
