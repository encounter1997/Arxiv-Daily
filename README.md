# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

[2021 April](202104.md)

[2021 June](202106.md)

[2021 July](202107.md)

## CV (Daily)

#### 20210803

* [HiFT: Hierarchical Feature Transformer for Aerial Tracking](https://arxiv.org/pdf/2108.00202.pdf)  (ICCV'21)
  * 用DETR做tracking
* [S^2-MLPv2: Improved Spatial-Shift MLP Architecture for Vision](https://arxiv.org/pdf/2108.01072.pdf)
* :star: [Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/pdf/2108.01073.pdf)  ([Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/))
* [Multilevel Knowledge Transfer for Cross-Domain Object Detection](https://arxiv.org/pdf/2108.00977.pdf)
  * incremental 结合图像翻译、对抗训练和伪标签
* [Word2Pix: Word to Pixel Cross Attention Transformer in Visual Grounding](https://arxiv.org/pdf/2108.00205.pdf)
  * In this paper we propose Word2Pix: a one-stage visual grounding network based on encoder-decoder transformer architecture that enables learning for textual to visual feature correspondence via word to pixel attention. 
  * The embedding of each word from the query sentence is treated alike by attending to visual pixels individually instead of single holistic sentence embedding.
* [CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention](https://arxiv.org/pdf/2108.00154.pdf)  (Deng Cai)
  * However, existing vision transformers still do not possess an ability that is important to visual input: building the attention among features of different scales
  * we propose Cross-scale Embedding Layer (CEL) and Long Short Distance Attention (LSDA).
* :star: [StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators ](https://arxiv.org/pdf/2108.00946.pdf) (NVIDIA)
  * Leveraging the semantic power of large scale Contrastive-Language-Image-Pretraining (CLIP) models, we present a text-driven method that allows shifting a generative model to new domains, without having to collect even a single image from those domains. 
  * We show that **through natural language prompts** and a few minutes of training, our method can adapt a generator across a multitude of domains characterized by diverse styles and shapes.
* [GTNet:Guided Transformer Network for Detecting Human-Object Interactions](https://arxiv.org/pdf/2108.00596.pdf)
* [GraphFPN: Graph Feature Pyramid Network for Object Detection](https://arxiv.org/pdf/2108.00580.pdf)  (ICCV'21)
  * State-of-the-art methods for multi-scale feature learning focus on performing feature interactions across space and scales using neural networks **with a fixed topology**.
  * In this paper, we propose graph feature pyramid networks that are capable of **adapting their topological structures to varying intrinsic image structures**, and **supporting simultaneous feature interactions across all scales**.
* [Greedy Network Enlarging ](https://arxiv.org/pdf/2108.00177.pdf) (Yunhe Wang)
  * 针对CNN的scaling
* [Multi-scale Matching Networks for Semantic Correspondence](https://arxiv.org/pdf/2108.00211.pdf)  (ICCV'21)
* [Learning Instance-level Spatial-Temporal Patterns for Person Re-identification](https://arxiv.org/pdf/2108.00171.pdf)  (Tieniu Tan)
* :star: [Multi-Head Self-Attention via Vision Transformer for Zero-Shot Learning](https://arxiv.org/pdf/2108.00045.pdf)  [code](https://github.com/FaisalAlamri0/ViT-ZSL)
* [Object-aware Contrastive Learning for Debiased Scene Representation](https://arxiv.org/pdf/2108.00049.pdf)  [code](git@github.com:alinlab/occon.git)
  * 解决对比学习过度关注背景区域的问题（真的是个问题？）
  * However, the learned representations are often contextually biased to the spurious scene correlations of different objects or object and background, which may harm their generalization on the downstream tasks.
  *  To tackle the issue, we develop a novel object-aware contrastive learning framework that first (a) localizes objects in a self-supervised manner and then (b) debias scene correlations via appropriate data augmentations considering the inferred object locations
* [Conditional Bures Metric for Domain Adaptation](https://arxiv.org/pdf/2108.00302.pdf)  (CVPR'21)
* [Group Fisher Pruning for Practical Network Compression](https://arxiv.org/pdf/2108.00708.pdf)  (ICML'21)

#### 20210802

* :star: [Perceiver IO: A General Architecture for Structured Inputs & Outputs ](https://arxiv.org/pdf/2107.14795.pdf)  （Andrew Zisserman, deepmind） [code](https://github.com/deepmind/deepmind-research/tree/master/perceiver)
  * The recently-proposed Perceiver model obtains good results on several domains (images, audio, multimodal, point clouds) while scaling linearly in compute and memory with the input size.
  * While the Perceiver supports many kinds of inputs, it can only produce very simple outputs such as class scores. Perceiver IO overcomes this limitation without sacrificing the original’s appealing properties by learning to flexibly query the model’s latent space to produce outputs of arbitrary size and semantics.
* :star: [Dynamic Neural Representational Decoders for High-Resolution Semantic Segmentation](https://arxiv.org/pdf/2107.14428.pdf)  (Zhi Tian,  Chunhua Shen)
  * Here, we propose a novel decoder, termed dynamic neural representational decoder (NRD), which is simple yet significantly more efficient. 
  * As each location on the encoder’s output corresponds to a local patch of the semantic labels, in this work, **we represent these local patches of labels with compact neural networks**. This neural representation enables our decoder to leverage the **smoothness prior** in the semantic label space, and thus makes our decoder more efficient. 
  *  Furthermore, these neural representations are **dynamically generated and conditioned on the outputs of the encoder networks**. The desired semantic labels can be efficiently decoded from the neural representations, resulting in high-resolution semantic segmentation predictions
* [DPT: Deformable Patch-based Transformer for Visual Recognition ](https://arxiv.org/pdf/2107.14467.pdf) (MM'21)
  * straight forward, but hard to reject.  [code](https://github.com/CASIA-IVA-Lab/DPT)
  * Existing methods usually use a fixed-size patch embedding which might destroy the semantics of objects. 
  * To address this problem, we propose a new Deformable Patch (DePatch) module which learns to adaptively split the images into patches with different positions and scales in a data-driven way rather than using predefined fixed patches. In this way, our method can well preserve the semantics in patches.
  * The DePatch module can work as a plug-and-play module, which can easily be incorporated into different transformers to achieve an end-to-end training.
* [T-SVDNet: Exploring High-Order Prototypical Correlations for Multi-Source Domain Adaptation](https://arxiv.org/pdf/2107.14447.pdf)  (ICCV'21)
* [Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation](https://arxiv.org/pdf/2107.14724.pdf)
  * With the rise of multi-modal datasets, large amount of 2D images are accessible besides 3D point clouds. In light of this, we propose to further leverage 2D data for 3D domain adaptation by intra and inter domain cross modal learning
* [Product1M: Towards Weakly Supervised Instance-Level Product Retrieval via Cross-modal Pretraining](https://arxiv.org/pdf/2107.14572.pdf)
* [On the Efficacy of Small Self-Supervised Contrastive Models without Distillation Signals](https://arxiv.org/pdf/2107.14762.pdf)  (Yueting Zhuang)
  * 提出问题：It is a consensus that small models perform quite poorly under the paradigm of self-supervised contrastive learning.  （Really?）
  * 分析问题：In this paper, we study the issue of training self-supervised small models without distillation signals. We first evaluate the representation spaces of the small models and make two non-negligible observations: (i) small models can complete the pretext task without overfitting despite its limited capacity; (ii) small models universally suffer the problem of overclustering.
  * 解决问题：Finally, we combine the validated techniques and improve the baseline of five small architectures with considerable margins, which indicates that training small self-supervised contrastive models is feasible even without distillation signals
* [ADeLA: Automatic Dense Labeling with Attention for Viewpoint Adaptation in Semantic Segmentation](https://arxiv.org/pdf/2107.14285.pdf)
  * We describe an unsupervised domain adaptation method for image content shift caused by viewpoint changes for a semantic segmentation task.
* [Fourier Series Expansion Based Filter Parametrization for Equivariant Convolutions](https://arxiv.org/pdf/2107.14519.pdf)
* [Manipulating Identical Filter Redundancy for Efficient Pruning on Deep and Complicated CNN](https://arxiv.org/pdf/2107.14444.pdf)
* [OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild](https://arxiv.org/pdf/2107.14480.pdf)  (ICCV'21)

