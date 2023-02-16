# config_loader.py

It is looked very useful, just like HuggingFace model config loader. But I feel something uncomportable.

Because, When if you use `pl.LightningModule` on your model code, you have to write `optimizing_step` on your same class.

I'm enjoying modify training argument in scripts(not model config), but if you used `pl.LightningModule` and `config_loader.py` , maybe you have to load 2 argument in class `__init__` about `training script argument` and  `model_config` class