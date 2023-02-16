# Warning Use DeepSpeed

DeepSpeed **ONLY** can be ran on **Multi GPU**



# Warning Use Data parallel or Distributed Data Parallel on Torchmetrics

https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-dataparallel-dp-mode

https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-distributed-data-parallel-ddp-mode



This `run_train_gpu.sh` is **ONLY** can be ran on **SINGLE GPU!**

**If you want to use Multi GPU DP, you have to write additional gather source code on evaluation_step_end**



`run_train_gpu_ddp.sh` is ran well, **but I'm not sure on metric gather is correctly**

