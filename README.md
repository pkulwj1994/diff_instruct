## Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models (Diff-Instruct)<br><sub>Official PyTorch implementation of the NeurIPS 2023 paper</sub>

**Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models**<br>
Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li and Zhihua Zhang.
<br>https://openreview.net/forum?id=MLIs5iRq4w<br>

Abstract: *Due to the ease of training, ability to scale, and high sample quality, diffusion models (DMs) have become the preferred option for generative modeling, with numerous pre-trained models available for a wide variety of datasets. Containing intricate information about data distributions, pre-trained DMs are valuable assets for downstream applications. In this work, we consider learning from pre-trained DMs and transferring their knowledge to other generative models in a data-free fashion. Specifically, we propose a general framework called Diff-Instruct to instruct the training of arbitrary generative models as long as the generated samples are differentiable with respect to the model parameters. Our proposed Diff-Instruct is built on a rigorous mathematical foundation where the instruction process directly corresponds to minimizing a novel divergence we call Integral Kullback-Leibler (IKL) divergence. IKL is tailored for DMs by calculating the integral of the KL divergence along a diffusion process, which we show to be more robust in comparing distributions with misaligned supports. We also reveal non-trivial connections of our method to existing works such as DreamFusion \citep{poole2022dreamfusion}, and generative adversarial training. To demonstrate the effectiveness and universality of Diff-Instruct, we consider two scenarios: distilling pre-trained diffusion models and refining existing GAN models. The experiments on distilling pre-trained diffusion models show that Diff-Instruct results in state-of-the-art single-step diffusion-based models. The experiments on refining GAN models show that the Diff-Instruct can consistently improve the pre-trained generators of GAN models across various settings. Our official code is released through \url{https://github.com/pkulwj1994/diff_instruct}.*

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

Code was based on Pytorch implementation of EDM diffusion model: https://github.com/NVlabs/edm.

## Prepare conda env

git clone https://github.com/pkulwj1994/diff_instruct.git
cd diff_instruct

source activate
conda create -n di_v100 python=3.8
conda activate di_v100
pip install torch==1.12.1 torchvision==0.13.1 tqdm click psutil scipy

## Pre-trained models

We use pre-trained EDM models:

- [https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/)


## Preparing datasets

Datasets are stored in the same format as in [StyleGAN](https://github.com/NVlabs/stylegan3): uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information.

**CIFAR-10:** Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool_edm.py --source=/data/downloads/cifar-10-python.tar.gz --dest=/data/datasets/cifar10-32x32.zip
```

**ImageNet:** Download the [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=/data/downloads/imagenet/ILSVRC/Data/CLS-LOC/train --dest=/data/datasets/imagenet-64x64.zip --resolution=64x64 --transform=center-crop
```

## Distill single-step models for CIFAR10 unconditional generation on a single V100 GPU (result in an FID <= 4.5)

You can run diffusion distillation using `di_train.py`. For example:

```.bash
# Train one-step DI model for unconditional CIFAR-10 using 1 GPUs
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 --master_port=25678 di_train.py --outdir=/logs/di/ci10-uncond --data=/data/datasets/cifar10-32x32.zip --arch=ddpmpp --batch 128 --edm_model cifar10-uncond --cond=0 --metrics fid50k_full --tick 10 --snap 50 --lr 0.00002 --glr 0.00002 --init_sigma 1.0 --fp16=0 --lr_warmup_kimg -1 --ls 1.0 --sgls 1.0
```

In the experiment, the FID will be calculated automatically for each "snap" of rounds. 

## License

Copyright &copy; 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

All material, including source code and pre-trained models, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

```
@inproceedings{
luo2023diffinstruct,
title={Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models},
author={Weijian Luo and Tianyang Hu and Shifeng Zhang and Jiacheng Sun and Zhenguo Li and Zhihua Zhang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=MLIs5iRq4w}
}
```

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgments

We thank EDM paper ""Elucidating the Design Space of Diffusion-Based Generative Models"" for its great implementation of EDM diffusion models in https://github.com/NVlabs/edm. We thank Shuchen Xue, and Zhengyang Geng for constructive feedback on code implementations.


