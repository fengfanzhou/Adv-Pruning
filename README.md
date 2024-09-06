# Adversarial Pruning (Adv-Pruning) 

This is the PyTorch implementation of our paper:

[Paper] Rethinking Impersonation and Dodging Attacks on Face Recognition Systems

Fengfan Zhou, Qianyu Zhou, Bangjie Yin, Hui Zheng, Xuequan Lu, Lizhuang Ma, Hefei Ling.

The 32nd ACM International Conference on Multimedia (MM), 2024

**[[Arxiv]](https://arxiv.org/pdf/2401.08903)**


## Environment Settings and Libraries

This project is tested under the following environment settings:
- OS: Ubuntu 20.04.6
- GPU: NVIDIA GeForce RTX 3090
- Cuda: 11.7
- Python: 3.8.12
- PyTorch: 2.0.1+cu117
- Torchvision: 0.15.2+cu117

## Pre-trained Models Preparation
Below, we outline the steps to obtain the checkpoints of the pre-trained models.
For the FR model, we can download the checkpoints from [the official AMT-GAN GitHub page](https://github.com/CGCL-codes/AMT-GAN).

The other custom models, we can integrate them conveniently by changing the `get_fr_model` in `utils/fr_model/fr_interface.py` for loading the models.

## Craft the Adversarial Examples
After the models are prepared, we can craft the adversarial examples using the following command:
```shell
CUDA_VISIBLE_DEVICES=3 python main.py --fr_model_ckpt_path=[the path of the FR checkpoint] --attacker_img_path=[the path of the attacker image] --victim_img_path=[the path of the victim image] --adv_img_path=[the path of the adversarial example to be saved]
```

# Citing Adv-Pruning
If you find IADG useful in your research, please consider citing:
```
@inproceedings{zhou2024rethinking,
  title={Rethinking impersonation and dodging attacks on face recognition systems},
  author={Zhou, Fengfan and Zhou, Qianyu and Yin, Bangjie and Zheng, Hui and Lu, Xuequan and Ma, Lizhuang and Ling, Hefei},
  booktitle={ACM Multimedia 2024},
  year={2024}
}
```
# License
This project is released under the [MIT LICENSE](https://github.com/fengfanzhou/adv_pruning/blob/main/LICENSE).

