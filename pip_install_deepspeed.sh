#!/bin/bash
CUDA_ARCH_LIST=`CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"`
arch="`echo $CUDA_ARCH_LIST | cut -c2`.`echo $CUDA_ARCH_LIST | cut -c5`"

git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST=$arch DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log