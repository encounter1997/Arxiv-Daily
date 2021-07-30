# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

[2021 April](202104.md)

[2021 June](202106.md)

## CV (Daily)

#### 20210730

* :star: [Open-World Entity Segmentation](https://arxiv.org/pdf/2107.14228.pdf)  (Jiaya Jia)  [code](https://github.com/dvlab-research/Entity)
  * We introduce a new image segmentation task, termed Entity Segmentation (ES) with the aim to segment all visual entities in an image without considering semantic category labels
  * It has many practical applications in image manipulation/editing where the segmentation mask quality is typically crucial but category labels are less important. 
  * In this setting, all semantically-meaningful segments are equally treated as categoryless entities and there is no thing-stuff distinction.
  * ES enables the following: (1) merging multiple datasets to form a large training set without the need to resolve label conflicts; (2) any model trained on one dataset can generalize exceptionally well to other datasets with unseen domains.
* [Learning with Noisy Labels for Robust Point Cloud Segmentation](https://arxiv.org/pdf/2107.14230.pdf)  (Dongdong Chen)
  * Point cloud segmentation is a fundamental task in 3D. Object class labels are often mislabeled in real-world point cloud datasets. In this work, we take the lead in solving this issue by proposing a novel Point Noise-Adaptive Learning (PNAL) framework.
  * noise-rate blind, to cope with the spatially variant noise rate problem specific to point clouds .
* :star: [Rethinking and Improving Relative Position Encoding for Vision Transformer](https://arxiv.org/pdf/2107.14222.pdf)  (Jianlong Fu) [code](https://github.com/microsoft/AutoML/tree/main/iRPE)
  * whether relative position encoding can work equally well as absolute position? In order to clarify this, we first review existing relative position encoding methods and analyze their pros and cons when applied in vision transformers.
  * We then propose new relative position encoding methods dedicated to 2D images, called image RPE (iRPE). Our methods consider **directional relative distance modeling** as well as the interactions between queries and relative position embeddings in self-attention mechanism.
  * Experiments demonstrate that solely due to the proposed encoding methods, DeiT and DETR obtain up to 1.5% (top-1 Acc) and 1.3% (mAP) stable improvements
* [A Unified Efficient Pyramid Transformer for Semantic Segmentation](https://arxiv.org/pdf/2107.14209.pdf)  (Mu Li)
  * 使用了deformable attention
* [ReFormer: The Relational Transformer for Image Captioning](https://arxiv.org/pdf/2107.14178.pdf)
  * we propose a novel architecture ReFormer- a RElational transFORMER to generate features with relation information embedded and to explicitly express the pair-wise relationships between objects in the image
  * ReFormer incorporates the objective of scene graph generation with that of image captioning using one modified Transformer model. 

* [Probabilistic and Geometric Depth: Detecting Objects in Perspective](https://arxiv.org/pdf/2107.14160.pdf)  (Dahua Lin)
* [Personalized Image Semantic Segmentation](https://arxiv.org/pdf/2107.13978.pdf)  (Ming-Ming Cheng)  (ICCV'21)
  * The objective is to generate more accurate segmentation results on unlabeled personalized images by investigating the data’s personalized traits.
  * To open up future research in this area, we collect a large dataset containing various users’ personalized images called PIS (Personalized Image Semantic Segmentation)
* [FREE: Feature Refinement for Generalized Zero-Shot Learning](https://arxiv.org/pdf/2107.13807.pdf)  (Ling Shao)  (ICCV'21)
  * Generalized zero-shot learning (GZSL) has achieved significant progress, with many efforts dedicated to overcoming the problems of **visual-semantic domain gap** and **seen unseen bias**.
  * However, most existing methods directly use feature extraction models trained on ImageNet alone, ignoring the **cross-dataset bias between ImageNet and GZSL benchmark**
* [Geometry Uncertainty Projection Network for Monocular 3D Object Detection](https://arxiv.org/pdf/2107.13774.pdf)  (Wanli Ouyan)  (ICCV'21)
* [Discovering 3D Parts from Image Collections](https://arxiv.org/pdf/2107.13629.pdf)  (Ming-Hsuan Yang)  (ICCV'21)
* [Few-Shot and Continual Learning with Attentive Independent Mechanisms](https://arxiv.org/pdf/2107.14053.pdf)  (ICCV'21)

#### 20210728  

* [Exploring Sequence Feature Alignment for Domain Adaptive Detection Transformers](https://arxiv.org/pdf/2107.12636.pdf) (MM'21)  [code](https://github.com/encounter1997/SFA)
* [Enriching Local and Global Contexts for Temporal Action Localization](https://arxiv.org/pdf/2107.12960.pdf) (ICCV'21)
* [Adaptive Denoising via GainTuning](https://arxiv.org/pdf/2107.12815.pdf)

#### 20210727

* [Improve Unsupervised Pretraining for Few-label Transfer](https://arxiv.org/pdf/2107.12369.pdf) (Suichan Li, Dongdong Chen, Nenghai Yu)  (ICCV'21)
  * Based on the analysis, we interestingly discover that only involving some unlabeled target domain into the unsupervised pretraining can improve the clustering quality, subsequently reducing the transfer performance gap with supervised pretraining.
* [Spatial-Temporal Transformer for Dynamic Scene Graph Generation](https://arxiv.org/pdf/2107.12309.pdf) (ICCV'21)
  * Dynamic scene graph generation aims at generating a scene graph of the given video
  * In this paper, we propose Spatial-temporal Transformer (STTran), a neural network that consists of two core modules: (1) a spatial encoder that takes an input frame to extract spatial context and reason about the visual relationships within a frame, and (2) a temporal decoder which takes the output of the spatial encoder as input in order to capture the temporal dependencies between frames and infer the dynamic relationships
* [Contextual Transformer Networks for Visual Recognition  ](https://arxiv.org/pdf/2107.12292.pdf)  (Ting Yao, Tao Mei)
  * most of existing designs directly employ self-attention over a 2D feature map to obtain the attention matrix based on pairs of isolated queries and keys at each spatial location, but leave the rich contexts among neighbor keys under-exploited
  * CoT block first mines the static context among keys via a 3×3 convolution. Next, based on the query and contextualized key, two consecutive 1×1 convolutions are utilized to perform self-attention, yielding the dynamic context. The static and dynamic contexts are finally fused as outputs
  * Our CoT block is appealing in the view that it can readily replace each 3 × 3 convolution in ResNet architectures, yielding a Transformer-style backbone named as Contextual Transformer Networks (CoTNet).
* [Adaptive Hierarchical Graph Reasoning with Semantic Coherence for Video-and-Language Inference](https://arxiv.org/pdf/2107.12270.pdf)  (Yi Yang, Yueting Zhuang)
  * TASK: Video-and-Language Inference is a recently proposed task for joint video-and-language understanding. This new task requires a model to draw inference on whether a natural language statement entails or contradicts a given video clip.
* [Text is Text, No Matter What: Unifying Text Recognition using Knowledge Distillation](https://arxiv.org/pdf/2107.12087.pdf)  (ICCV'21)
  * The challenging nature of the text recognition problem however dictated a fragmentation of research efforts: Scene Text Recognition (STR) that deals with text in everyday scenes, and Handwriting Text Recognition (HTR) that tackles hand-written text. 
  * In this paper, for the first time, we argue for their unification – we aim for a single model that can compete favourably with two separate state-of-the-art STR and HTR models
* [Parametric Contrastive Learning](https://arxiv.org/pdf/2107.12028.pdf)  (Jiaya Jia)  (ICCV'21)
  * we propose Parametric Contrastive Learning (PaCo) to tackle long-tailed recognition
  * Based on theoretical analysis, we observe **supervised contrastive loss** tends to bias on high-frequency classes and thus increases the difficulty of imbalance learning.
  * We introduce a set of **parametric class-wise learnable centers** to rebalance from an optimization perspective. Further, we analyze our PaCo loss under a balanced setting.
* [Language Models as Zero-shot Visual Semantic Learner](https://arxiv.org/pdf/2107.12021.pdf)
  * 

#### 20210719

* :star: [The Benchmark Lottery](https://arxiv.org/pdf/2107.07002.pdf)

* [A Survey on Bias in Visual Datasets](https://arxiv.org/pdf/2107.07919.pdf)  

  i) describe the biases that can affect visual datasets; ii) review the literature on methods for bias discovery and quantification in visual datasets; iii) discuss existing attempts to collect bias-aware visual datasets.

* [Rectifying the Shortcut Learning of Background: Shared Object Concentration for Few-Shot Image Recognition](https://arxiv.org/pdf/2107.07746.pdf)
  In this paper, we observe that image background serves as a source of domain-specific knowledge, which is a shortcut for models to learn in the source dataset, but is harmful when adapting to brand-new classes.

* [CutDepth: Edge-aware Data Augmentation in Depth Estimation](https://arxiv.org/pdf/2107.07684.pdf) 
  In this paper, we propose a data augmentation method, called CutDepth. In CutDepth, part of the depth is pasted onto an input image during training. The method extends variations data without destroying edge features.

* [Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/pdf/2107.07651.pdf)

* [Multi-Level Contrastive Learning for Few-Shot Problems](https://arxiv.org/pdf/2107.07608.pdf)
Most current applications of contrastive learning benefit only a single representation from the last layer of an encoder.In this paper, we propose a multi-level contrasitive learning approach which applies contrastive losses at different layers of an encoder to learn multiple representations from the encoder.


#### 20210716

* [Recurrent Parameter Generators](https://arxiv.org/pdf/2107.07110.pdf)  (Yann LeCun)
* [From Show to Tell: A Survey on Image Captioning](https://arxiv.org/pdf/2107.06912.pdf)
* [yleFusion: A Generative Model for Disentangling Spatial Segments](https://arxiv.org/pdf/2107.07437.pdf)

#### 20210715

* [Deep Neural Networks are Surprisingly Reversible: A Baseline for Zero-Shot Inversion](https://arxiv.org/pdf/2107.06304.pdf)  (NVIDIA)
* [A Generalized Lottery Ticket Hypothesis](https://arxiv.org/pdf/2107.06825.pdf)
* [How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/pdf/2107.06383.pdf)


#### 20210706

* [What Makes for Hierarchical Vision Transformer?](https://arxiv.org/pdf/2107.02174.pdf) (Xinggang Wang)
* [MixStyle Neural Networks for Domain Generalization and Adaptation](https://arxiv.org/pdf/2107.02053.pdf)
* [On Model Calibration for Long-Tailed Object Detection and Instance Segmentation](https://arxiv.org/pdf/2107.02170.pdf)  (Boqing Gong)
* [Test-Time Personalization with a Transformer for Human Pose Estimation](https://arxiv.org/pdf/2107.02133.pdf)  (Xiaolong Wang)

#### 20210705
* :star: [Simpler, Faster, Stronger: Breaking The log-K Curse On Contrastive Learners With FlatNCE](https://arxiv.org/pdf/2107.01152.pdf)
* [How Incomplete is Contrastive Learning? An Inter-intra Variant Dual Representation Method for Self-supervised Video Recognition](https://arxiv.org/pdf/2107.01194.pdf)
* [A Survey on Deep Learning Technique for Video Segmentation](https://arxiv.org/pdf/2107.01153.pdf)  (Wenguan Wang,  Luc Van Gool)
* [Collaborative Visual Navigation](https://arxiv.org/pdf/2107.01151.pdf)  (Wenguan Wang, Xizhou Zhu, Jifeng Dai)
* [Unsupervised Single Image Super-resolution Under Complex Noise](https://arxiv.org/pdf/2107.00986.pdf)
* [Polarized Self-Attention: Towards High-quality Pixel-wise Regression](https://arxiv.org/pdf/2107.00782.pdf)
* [Blind Image Super-Resolution via Contrastive Representation Learning](https://arxiv.org/pdf/2107.00708.pdf)
* [Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets](https://arxiv.org/pdf/2107.00860.pdf)  (ICLR'21)

#### 20210702

* [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](https://arxiv.org/pdf/2107.00652.pdf)  (Dongdong Chen, Nenghai Yu, Baining Guo)
* [AutoFormer: Searching Transformers for Visual Recognition](https://arxiv.org/pdf/2107.00651.pdf) (Jianlong Fu)
* :star: [CLIP-It! Language-Guided Video Summarization](https://arxiv.org/pdf/2107.00650.pdf)  (Trevor Darrell)
* [On the Practicality of Deterministic Epistemic Uncertainty](https://arxiv.org/pdf/2107.00649.pdf)  (Luc Van Gool, Fisher Yu)
* [Global Filter Networks for Image Classification](https://arxiv.org/pdf/2107.00645.pdf)  transformer + 傅里叶变换
* [Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/pdf/2107.00641.pdf)
* :star: [CBNetV2: A Composite Backbone Network Architecture for Object Detection](https://arxiv.org/pdf/2107.00420.pdf)  In this paper, we propose a novel backbone network, namely CBNetV2, by constructing compositions of existing open-sourced pretrained backbones.
* [OPT: Omni-Perception Pre-Trainer for Cross-Modal Understanding and Generation](https://arxiv.org/pdf/2107.00249.pdf)  (Hanqing Lu) we propose an Omni-perception PreTrainer (OPT) for cross-modal understanding and generation, by jointly modeling visual, text and audio resources.
* :star: [Simple Training Strategies and Model Scaling for Object Detection](https://arxiv.org/pdf/2107.00057.pdf)  (Tsung-Yi Lin)
* [CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation](https://arxiv.org/pdf/2107.00085.pdf)
* [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/pdf/2107.00135.pdf)
* [Learning to See before Learning to Act: Visual Pre-training for Manipulation](https://arxiv.org/pdf/2107.00646.pdf)  (Phillip Isola, Tsung-Yi Lin)
* [Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation](https://arxiv.org/pdf/2107.00644.pdf)  (Xiaolong Wang)
* :star: [AdaXpert: Adapting Neural Architecture for Growing Data](https://arxiv.org/pdf/2107.00254.pdf)
* [FedMix: Approximation of Mixup under Mean Augmented Federated Learning](https://arxiv.org/pdf/2107.00233.pdf)  (ICLR'21)
* [Scalable Certified Segmentation via Randomized Smoothing](https://arxiv.org/pdf/2107.00228.pdf)  (ICML'21)
* [Revisiting Knowledge Distillation: An Inheritance and Exploration Framework](https://arxiv.org/pdf/2107.00181.pdf)  (CVPR'21, Tongliang Liu, Xinmei Tian, Houqiang Li, Xian-Sheng Hua)
* [Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?](https://arxiv.org/pdf/2107.00166.pdf)


#### 20210701

* [SOLO: A Simple Framework for Instance Segmentation](https://arxiv.org/pdf/2106.15947.pdf)  SOLO TPAMI version  
* [Align Yourself: Self-supervised Pre-training for Fine-grained Recognition via Saliency Alignment](https://arxiv.org/pdf/2106.15788.pdf)
* [Augmented Shortcuts for Vision Transformers](https://arxiv.org/pdf/2106.15941.pdf)  (Yunhe Wang)
* [Multi-Source Domain Adaptation for Object Detection](https://arxiv.org/pdf/2106.15793.pdf)