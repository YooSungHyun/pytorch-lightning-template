# pytorch-lightning-template
very simple but, write down is boring </br>
boring boiling code rolling ⚡ </br>
- DataModule more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)
- Model more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
- Inference more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html)
- WanDB with lightning more detail
    - [Weight & Bias Dev Guid 1](https://wandb.ai/wandb_fc/korean/reports/Weights-Biases-Pytorch-Lightning---VmlldzozNzAxOTg)
    - [Weight & Bias Dev Guid 2](https://docs.wandb.ai/guides/integrations/lightning)

# Training Detail
- Using DDP, Not DP or CPU</br>
    Maybe want to using DP or CPU, Change some argument or python Script</br>
    See more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html)
- Optimizer: AdamW
    - LearningRate Scheduler: OneCycleLR
    - see more detail: [PyTorch Dev Guide](https://pytorch.org/docs/stable/optim.html)
- Monitoring Tool: WanDB
