# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

[2021 April](202104.md)

[2021 June](202106.md)

[2021 July](202107.md)

[2021 Aug](202108.md)

[2021 Sep](202109.md)

## CV (Daily)

#### 20211108

* :star: ​[EditGAN: High-Precision Semantic Image Editing](https://arxiv.org/pdf/2111.03186.pdf) (NVIDIA, MIT) NIPS
* [Improving Visual Quality of Image Synthesis by A Token-based Generator with Transformers](https://arxiv.org/pdf/2111.03481.pdf) (jianlong fu) NIPS
  * 用transformer做图像生成，避免生搬硬套的角度：（1）逐个token的生成有利于保证生成图像的局部特性（2）new perspective: token-based generator

#### 20211110

* [Data Augmentation Can Improve Robustness](https://arxiv.org/pdf/2111.05328.pdf) (NIPS21)

  * 对抗训练会面临鲁棒性过拟合问题，本文提出一种数据增强方法来提升鲁棒性
  * Adversarial training suffers from robust overfitting, a phenomenon where the robust test accuracy starts to decrease during training. In this paper, we focus on reducing robust overfitting by using common data augmentation schemes. 
  * We demonstrate that, contrary to previous findings, when combined with model weight averaging, data augmentation can significantly boost robust accuracy.

* [Sliced Recursive Transformer](https://arxiv.org/pdf/2111.05297.pdf) (Eric Xing)

  * ICLR submission (65553)
  * We present a neat yet effective recursive operation on vision transformers that can improve parameter utilization without involving additional parameters. This is achieved by sharing weights across depth of transformer networks.

* [MixACM: Mixup-Based Robustness Transfer via Distillation of Activated Channel Maps](https://arxiv.org/pdf/2111.05073.pdf)  (NIPS21)

  * 提升对抗鲁棒性的最常用方法还是对抗训练，本文提出一种alternative，通过知识蒸馏用鲁棒teacher网络提升student鲁棒性

  * First, we theoretically show the transferability of robustness from an adversarially trained teacher model to a student model with the help of mixup augmentation. 
  * MixACM transfers robustness from a robust teacher to a student by matching activated channel maps generated without expensive adversarial perturbations

* [Self-Interpretable Model with Transformation Equivariant Interpretation]() (NIPS21)

  * 解释性网络稳定性很差，容易受到数据扰动或变换的干扰。本文提出一种鲁棒的解释性方法，它在self-interpretable model中引入变换的不变性约束。
  * Recent studies have found that interpretation methods can be sensitive and unreliable, where the interpretations can be disturbed by perturbations or transformations of input data. 
  * To address this issue, we propose to learn robust interpretations through transformation equivariant regularization in a self-interpretable model.

