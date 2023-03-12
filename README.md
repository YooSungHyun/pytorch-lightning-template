# NEW Release 20230312!!!

update dropout **Dropout Reduces Underfitting**

[paper](https://arxiv.org/abs/2303.01500)

[git](https://github.com/facebookresearch/dropout)

## Usage

CHECK PLZ `run_train_gpu.sh` and `./models/dense_model/`

### 1. make drop_scheduler.py

### 2. make dropout scheduling function your each model classes

**Example**

```python
# 1) update dropout function
def update_dropout(self, drop_rate):
    self.drop_rate = drop_rate
    for module in self.modules():
        if isinstance(module, nn.Dropout):
            module.p = drop_rate

# 2) we have to count each step (for calculate schedulering global steps)
def on_train_start(self):
    from models.dense_model.drop_scheduler import drop_scheduler

    self.drop_scheduler = {}
    if self.args.dropout_p > 0.0:
        self.drop_scheduler["do"] = drop_scheduler(
            self.args.dropout_p,
            self.args.max_epochs,
            self.trainer.num_training_batches,
            self.args.cutoff_epoch,
            self.args.drop_mode,
            self.args.drop_schedule,
        )
        print(
            "on_train_start :: Min DO = %.7f, Max DO = %.7f"
            % (min(self.drop_scheduler["do"]), max(self.drop_scheduler["do"]))
        )

# 3) Finally, you can scheduling dropout prob in your training_step
if "do" in self.drop_scheduler:
        dropout_p = self.drop_scheduler["do"][self.trainer.global_step]
        self.update_dropout(dropout_p)
        self.log("dropout_p", dropout_p, sync_dist=(self.device != "cpu"))
```

### 3. we have to input dropout schedule setting in training_arg

### 4. replace run script

``` bash
--dropout_p=0.1 \
--cutoff_epoch=1 \
--drop_mode=standard \
--drop_schedule=constant
```



### If you want to use, normal style dropout, input dropout_p and `drop_mode=standard` (default) and `drop_schedule=constant` (default)
you can check your dropout scheduling process in wandb

![image](https://user-images.githubusercontent.com/34292279/224522943-37e5779a-3f40-44c0-8aec-5a13f84ff47c.png)


# pytorch-lightning-template

very simple but, write down is boring </br>
boring boiling code rolling ⚡ </br>
**If you need some function or someting, plz comment issues (plz write eng or ko). I reply and implement ASAP!!** </br>

- DataModule more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)
- Model more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
- Inference more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html)
- WanDB with lightning more detail
  - [Weight & Bias Dev Guide 1](https://wandb.ai/wandb_fc/korean/reports/Weights-Biases-Pytorch-Lightning---VmlldzozNzAxOTg)
  - [Weight & Bias Dev Guide 2](https://docs.wandb.ai/guides/integrations/lightning)

# WanDB

https://docs.wandb.ai/v/ko/quickstart

# Training Detail

- Using DDP, Not DP or CPU</br>
  Maybe want to using DP or CPU, Change some argument or python Script</br>
  See more detail: [PyTorch-Lightning Dev Guide](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html)
- Optimizer: AdamW
  - LearningRate Scheduler: OneCycleLR
  - see more detail: [PyTorch Dev Guide](https://pytorch.org/docs/stable/optim.html)
- Monitoring Tool: WanDB

# Pytorch-Lightning Life Cycle

## Training

1. train.py(main) -> argparse
   - using [simple_parsing library](https://github.com/lebrice/SimpleParsing) looks like HFArgumentParser
   - Trainer Argument placed with `pl.Trainer.add_argparse_args` (automatic define argparse)
2. def] WandbLogger, set seed(os, random, np, torch, torch.cuda)
3. def] CustomDataModule (`LightningDataModule`)
   - You Not have to using `LightningDataModule`. but, if you implement that in 'LightningModule', source code is looked mess
   - DataModule important `prepare_data` and `setup`
     - `prepare_data` is only run on cpu and not multi processing (Warning, if you using distributed learning, this place's variable is not share)
       - I recommand, It just using data download or datasets save
     - `setup` is run on gpu or cpu and distributed. using map or dataload or something!
       - `setup` can have stage(fit (train), test, predict)
   - DataModule can have each stage's dataloader
     - using default or someting
   - Dataset can define this section or making each python script and just import & using!
4. def] CustomNet (`LightningModule`)
   - each step and step_end or epoch, epoch_end
   - i think using just training_step, validation_step, validation_epoch_end is simple and best
     - training_step -> forward -> configure_optimizers
     - when count in each validation step (each batch step validation) -> validation_epoch_end (all batch result gather) -> log (on wandb)
5. wandb logger additional setting
6. checkpoint setting
   - monitor name is same on your each step's log name
7. learning_rate monitor setting
8. ddp strategy modify
   - if your dataset is so big to ddp, timeout parameter change like that
   - huggingface is so hard to make it. but lightning is feel free
9. make trainer to your arg
10. training run and model save!

### Training Script Usage

1. cd your project root(./pytorch-lightning-template)

```
# Don't Script RUN in your scripts FOLDER!!!!! CHK PLZ!!!!!!!
bash scripts/run_train_~~~.sh
```

## Inference

1. inference.py(main) -> argparse
2. set seed
3. model load (second param is your model init param)
4. simply torch inference & END!

### Inference Script Usage

1. cd your project root(./pytorch-lightning-template)

```
# Don't Script RUN in your scripts FOLDER!!!!! CHK PLZ!!!!!!!
bash scripts/run_inference~~~.sh
```

# (Optional) Install DeepSpeed

1. run pip_install_deepspeed.sh

```
bash pip_install_deepspeed.sh
```
