# Arxiv-Daily
My daily arxiv reading notes


## CV (Daily)
#### 20210330
CVPR21：
* 

Vision Transformer:
* [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808.pdf) This is accomplished through two primary modifications: a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer block leveraging a convolutional projection. [code](https://github.com/leoxiaobin/CvT)
* [PixelTransformer: Sample Conditioned Signal Generation](https://arxiv.org/pdf/2103.15813.pdf) We propose a generative model that can infer a distribution for the underlying spatial signal conditioned on sparse samples e.g. plausible images given a few observed pixels.
* [ViViT: A Video Vision Transformer](https://arxiv.org/pdf/2103.15691.pdf) transformer做视频分类 In order to handle the long sequences of tokens encountered in video, we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input. 
* [On the Adversarial Robustness of Visual Transformers](https://arxiv.org/pdf/2103.15670.pdf) Tested on various white-box and transfer attack settings, we find that ViTs possess better adversarial robustness when compared with convolutional neural networks (CNNs). We summarize the following main observations contributing to the improved robustness of ViTs: 1) Features learned by ViTs contain less low-level information and are more generalizable, which contributes to superior robustness against adversarial perturbations. 2) Introducing convolutional or tokens-to-token blocks for learning low-level features in ViTs can improve classification accuracy but at the cost of adversarial robustness. 3) Increasing the proportion of transformers in the model structure (when the model consists of both transformer and CNN blocks) leads to better robustness. But for a pure transformer model, simply increasing the size or adding layers cannot guarantee a similar effect. 4) Pre-training on larger datasets does not significantly improve adversarial robustness though it is critical for training ViTs. 5) Adversarial training is also applicable to ViT for training robust models. The results show that ViTs are less sensitive to high-frequency perturbations than CNNs and there isa high correlation between how well the model learns low level features and its robustness against different frequencybased perturbations.
* [Transformer Tracking](https://arxiv.org/pdf/2103.15436.pdf) [code](https://github.com/chenxin-dlut/TransT) 从tracking中关系建模和特征融合的重要性和复杂性讲起，引入transformer (Huchuan Lu)
* [Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding](https://arxiv.org/pdf/2103.15358.pdf) Multi-Scale Vision Longformer, which significantly enhances the ViT of [11] for encoding highresolution images using two techniques. multi-scale model structure, attention mechanism of vision Longformer, which is a variant of Longformer [2]. 实验：image classification, object detection, and segmentation. 
* [TFPose: Direct Human Pose Estimation with Transformers](https://arxiv.org/pdf/2103.15320.pdf) we formulate the pose estimation task into a sequence prediction problem that can effectively be solved by transformers. 关键：分析引入transformer解决的问题和带来的好处：Our framework is simple and direct, bypassing the drawbacks of the heatmapbased pose estimation. Moreover, with the attention mechanism in transformers, our proposed framework is able to adaptively attend to the features most relevant to the target keypoints, which largely overcomes the feature misalignment issue of previous regression-based methods and considerably improves the performance. [AdelaiDet](https://github.com/aim-uofa/AdelaiDet/)  (Chunhua Shen, Zhi Tian, Xinlong Wang, et al.)
* [HiT: Hierarchical Transformer with Momentum Contrast for Video-Text Retrieval](https://arxiv.org/pdf/2103.15049.pdf) 讲Transformer和对比学习结合做Video-Text Retrieval
* [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/pdf/2103.14899.pdf) 在ViT中引入多尺度和efficiency. To reduce computation, we develop a simple yet effective token fusion module based on cross attention, which uses a single token for each branch as a query to exchange information with other branches. (Linear)
* [Looking Beyond Two Frames: End-to-End Multi-Object Tracking Using Spatial and Temporal Transformers](https://arxiv.org/pdf/2103.14829.pdf)
* [Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers](https://arxiv.org/pdf/2103.15679.pdf) [code](https://github.com/hila-chefer/Transformer-MM-Explainability) 研究transformer可解释性. Unlike Transformers that only use self-attention, Transformers with coattention require to consider multiple attention maps in
parallel in order to highlight the information that is relevant to the prediction in the model’s input. 
(coattention就是cross attention?)
* [Face Transformer for Recognition](https://arxiv.org/pdf/2103.14803.pdf) 单纯拿ViT在人脸识别上测试 We wonder if transformer can be used in face recognition and whether it is better than CNNs. Therefore, we investigate the performance of Transformer models in face recognition.
* [TransCenter: Transformers with Dense Queries for Multiple-Object Tracking](https://arxiv.org/pdf/2103.15145.pdf) Inspired by recent research, we propose TransCenter, the first transformer-based architecture for tracking the centers of multiple targets. Methodologically, we propose the use of dense queries in a double-decoder network, to be able to robustly infer the heatmap of targets’ centers and associate them through time.

#### 20210329
CVPR21
* [PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds](https://arxiv.org/pdf/2103.14635.pdf) 动态网络Position Adaptive Convolution (PAConv) for r 3D point cloud processing. The key of PAConv is to construct the convolution kernel by dynamically assembling basic weight matrices stored in Weight Bank, where the coefficients of these weight matrices are self-adaptively learned from point positions through ScoreNet. [code](https://github.com/CVMI-Lab/PAConv)
* [Distilling Object Detectors via Decoupled Features](https://arxiv.org/pdf/2103.14475.pdf) 知识蒸馏做目标检测，方法采用前后景特征解耦 In this paper, we point out that the information of features derived from regions excluding objects are also essential for distilling the student detector, which is usually ignored in existing approaches.  two levels of decoupled features will be processed for embedding useful information into the student, i.e., decoupled features from neck and decoupled proposals from classification head. [code](https://github.com/ggjy/DeFeat.pytorch) (Yunhe Wang, Chang Xu)
* [Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling](https://arxiv.org/pdf/2103.14338.pdf) 
* [Confluent Vessel Trees with Accurate Bifurcations](https://arxiv.org/pdf/2103.14268.pdf)  unsupervised reconstruction of complex near-capillary vasculature with thousands of bifurcations where supervision and learning are infeasible.
* [Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification](https://arxiv.org/pdf/2103.14267.pdf) 对比学习做长尾分类 [code](https://www.kaihan.org/HybridLT/)  (Xiu-Shen Wei)
* [OTA: Optimal Transport Assignment for Object Detection](https://arxiv.org/pdf/2103.14259.pdf) 受到detr中label assignment的启发，提出基于optimal transport的global label assignment机制. we innovatively revisit the label assignment from a global perspective and propose to formulate the assigning procedure as an Optimal Transport (OT) problem – a well-studied topic in Optimization Theory. After formulation, finding the best assignment solution is converted to solve the optimal transport plan at minimal transportation costs, which can be solved via Sinkhorn-Knopp Iteration.  [code](https://github.com/Megvii-BaseDetection/OTA) (Jian Sun)
* [MagDR: Mask-guided Detection and Reconstruction for Defending Deepfakes](https://arxiv.org/pdf/2103.14211.pdf)
* [Equivariant Point Network for 3D Point Cloud Analysis](https://arxiv.org/pdf/2103.14147.pdf) In this paper, we propose an effective and practical SE(3) (3D translation and rotation) equivariant network for point cloud analysis that addresses both problems. [code](https://github.com/nintendops/EPN_PointCloud)
* [ACRE: Abstract Causal REasoning Beyond Covariation](https://arxiv.org/pdf/2103.14232.pdf) (朱松纯)
* [Abstract Spatial-Temporal Reasoning via Probabilistic Abduction and Execution](https://arxiv.org/pdf/2103.14230.pdf) （朱松纯）

其他
* [Understanding Robustness of Transformers for Image Classification](https://arxiv.org/pdf/2103.14586.pdf) 探索vision transformer鲁棒性 We investigate robustness to input perturbations as well as robustness to model perturbations. We find that when pre-trained with a sufficient amount of data, ViT models are at least as robust as the ResNet counterparts on a broad range of perturbations. We also find that Transformers are robust to the removal of almost any single layer, and that while activations from later layers are highly correlated with each other, they nevertheless play an important role in classification. (Google Research)

* [COTR: Correspondence Transformer for Matching Across Images](https://arxiv.org/pdf/2103.14167.pdf) 问题设置类似Jianlong Fu, et al用transformer解决有reference的超分辨

* [Lifting Transformer for 3D Human Pose Estimation in Video](https://arxiv.org/pdf/2103.14304.pdf) 

* [Training a Better Loss Function for Image Restoration](https://arxiv.org/pdf/2103.14616.pdf) In this work, we explore the question of what makes a good loss function for an image restoration task. [code](https://github.com/gfxdisp/mdf)

* [Marine Snow Removal Benchmarking Dataset](https://arxiv.org/pdf/2103.14249.pdf) low-level 水下新任务和benchmark [code](https://github.com/ychtanaka/marine-snow)

* [DivAug: Plug-in Automated Data Augmentation with Explicit Diversity Maximization](https://arxiv.org/pdf/2103.14545.pdf)

* [On Generating Transferable Targeted Perturbations](https://arxiv.org/pdf/2103.14641.pdf) changing an unseen model’s decisions to a specific ‘targeted’ class remains a challenging feat. In this paper, we propose a new generative approach for highly transferable targeted perturbations (TTP). [code](https://github.com/Muzammal-Naseer/TTP)

* [Unsupervised Robust Domain Adaptation without Source Data](https://arxiv.org/pdf/2103.14577.pdf) 将Domain Adaptation without Source Data和对抗鲁棒性问题结合进行研究 This paper aims at answering the question of finding the right strategy to make the target model robust and accurate in the setting of unsupervised domain adaptation without source data. (Luc Van Gool)

* [Geometry-Aware Unsupervised Domain Adaptation for Stereo Matching](https://arxiv.org/pdf/2103.14333.pdf) DA for Stereo Matching （期刊）

* [Non-Salient Region Object Mining for Weakly Supervised Semantic Segmentation](https://arxiv.org/pdf/2103.14581.pdf) However, existing works mainly concentrate on expanding the seed of pseudo labels within the image’s salient region. In this work, we propose a non-salient region object mining approach for weakly supervised semantic segmentation. [code](https://github.com/NUST-Machine-Intelligence-Laboratory/nsrom)

* [Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings](https://arxiv.org/pdf/2103.14572.pdf) 生物图像实例分割 We propose to address the dense annotation bottleneck by introducing a proposal-free segmentation approach based on non-spatial embeddings, which exploits the structure of the learned embedding space to extract individual instances in a differentiable way. [code](https://github.com/kreshuklab/spoco)

* [Towards a Unified Approach to Single Image Deraining and Dehazing](https://arxiv.org/pdf/2103.14204.pdf) （期刊）

  


#### 20210326
TOP
* [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf) a hierarchical Transformer whose representation is computed with shifted windows. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. BOOM RESULTS. [code](https://github.com/microsoft/Swin-Transformer) (Han Hu, Yue Cao, Stephen Lin et al)     
* [Vision Transformers for Dense Prediction](https://arxiv.org/pdf/2103.13413.pdf) 提出针对dense prediction task的vision transformer，采用类似fpn的结构汇聚transformer各层的特征，在monocular depth estimation, semantic segmentation上验证其性能。 [code](https://github.com/intel-isl/DPT) 
* [High-Fidelity Pluralistic Image Completion with Transformers](https://arxiv.org/pdf/2103.14031.pdf) This paper brings the best of both worlds to pluralistic image completion: appearance prior reconstruction with transformer and texture replenishment with CNN. (Dongdong Chen)   
* [AutoLoss-Zero: Searching Loss Functions from Scratch for Generic Tasks](https://arxiv.org/pdf/2103.14026.pdf) In this paper, we propose AutoLoss-Zero, the first general framework for searching loss functions from scratch for generic tasks. (Jifeng Dai, Hongsheng Li, Gao Huang, Xizhou Zhu, et al)    
* [Orthogonal Projection Loss](https://arxiv.org/pdf/2103.14021.pdf) 改进交叉熵损失：Motivated by the observation that groundtruth class representations in CE loss are orthogonal (onehot encoded vectors), we develop a novel loss function termed ‘Orthogonal Projection Loss’ (OPL) which imposes orthogonality in the feature space.   Given the plug-and-play nature of OPL, we evaluate it on a diverse range of tasks including image recognition (CIFAR-100), large-scale classification (ImageNet), domain generalization (PACS) and few-shot learning (miniImageNet, CIFAR-FS, tiered-ImageNet and Meta-dataset) and demonstrate its effectiveness across the board. （但每个提升都不多） [code](https://github.com/kahnchana/opl)

CVPR21:
* [More Photos are All You Need: Semi-Supervised Learning for Fine-Grained Sketch Based Image Retrieval](https://arxiv.org/pdf/2103.13990.pdf) semi-supervised Fine-Grained Sketch-Based Image Retrieval (FG-SBIR) 采用图像翻译生成更多的成对数据用于训练（low-level主流） [code](https://github.com/AyanKumarBhunia/semisupervised-FGSBIR)
* [Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/pdf/2103.13886.pdf) This work instead augments the fine-tuning stage for object detectors by exploring adversarial examples, which can be viewed as a model-dependent data augmentation. Our method dynamically selects the stronger adversarial images sourced from a detector’s classification and localization branches and evolves with the detector to ensure the augmentation policy stays current and relevant. (Boqing Gong)  
* [Learning Dynamic Alignment via Meta-filter for Few-shot Learning](https://arxiv.org/pdf/2103.13582.pdf) Most of the existing methods for feature alignment in few-shot learning only consider image-level or spatial-level alignment while omitting the channel disparity.   Therefore, in this paper, we propose to learn a dynamic alignment, which can effectively highlight both query regions and channels according to different local support information.
* [MetaAlign: Coordinating Domain Alignment and Classification for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2103.13575.pdf) 解决训练损失和测试metric之间mismatch的问题，指出domain alignment和task的优化方向之间存在差异，提出将其看作meta-learning问题，即插即用，在分类、检测域适应和域泛化问题上验证性能。 Motivation：However, the optimization objective of such domain alignment is generally not coordinated with that of the object classification task itself such that their descent directions for optimization may be inconsistent. In this paper, we aim to study and alleviate the optimization inconsistency problem between the domain alignment and classification tasks. 方法：MetaAlign, where we treat the domain alignment objective and the classification objective as the meta-train and meta-test tasks in a meta-learning scheme. (曾文君，陈志波)
* [I^3Net: Implicit Instance-Invariant Network for Adapting One-Stage Object Detectors](https://arxiv.org/pdf/2103.13757.pdf)
* [Vectorization and Rasterization: Self-Supervised Learning for Sketch and Handwriting](https://arxiv.org/pdf/2103.13716.pdf) 自监督Sketch and Handwriting表示学习，将其看作文本和图像之间的模态。In this paper, we are interested in defining a self-supervised pre-text task for sketches and handwriting data. This data is uniquely characterised by its existence in dual modalities of rasterized images and vector coordinate sequences.   two novel cross-modal translation pre-text tasks for selfsupervised feature learning: Vectorization and Rasterization.
* [OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations](https://arxiv.org/pdf/2103.13843.pdf) 新的迁移任务：同时考虑domain和task的迁移。Transfer learning across heterogeneous data distributions (a.k.a. domains) and distinct tasks is a more general and challenging problem than conventional transfer learning, where either domains or tasks are assumed to be the same.  We propose a transferability metric called Optimal Transport based Conditional Entropy (OTCE), to analytically predict the transfer performance for supervised classification tasks in such cross domain and cross-task feature transfer settings
* [Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation](https://arxiv.org/pdf/2103.13660.pdf) 图像翻译做去雨(low-level主流)

Others 
* [An Image is Worth 16x16 Words, What is a Video Worth?](https://arxiv.org/pdf/2103.13915.pdf) transformer做action recognition. [code](https://github.com/Alibaba-MIIL/STAM)  

* [Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy Labels](https://arxiv.org/pdf/2103.13646.pdf) we identify a “warm-up obstacle”: the inability of standard warm-up stages to train high quality feature extractors and avert memorization of noisy labels. We propose “Contrast to Divide” (C2D), a simple framework that solves this problem by pre-training the feature extractor in a self-supervised fashion. 

* [Universal Representation Learning from Multiple Domains for Few-shot Classification](https://arxiv.org/pdf/2103.13841.pdf) In this work, we propose to learn a single set of universal deep representations by distilling knowledge of multiple separately trained networks after co-aligning their features with the help of adapters and centered kernel alignment. 

* [Inferring Latent Domains for Unsupervised Deep Domain Adaptation](https://arxiv.org/pdf/2103.13873.pdf)   (TPAMI)  

* [USB: Universal-Scale Object Detection Benchmark](https://arxiv.org/pdf/2103.14027.pdf) In this paper, we introduce the UniversalScale object detection Benchmark (USB). USB has variations in object scales and image domains by incorporating COCO with the recently proposed Waymo Open Dataset and Manga109-s dataset. UniverseNets. [code](https://github.com/shinya7y/UniverseNet)  

* [Multi-Target Domain Adaptation via Unsupervised Domain Classification for Weather Invariant Object Detection](https://arxiv.org/pdf/2103.13970.pdf) However, most existing domain adaptation methods either handle singletarget domain or require domain labels. We propose a novel unsupervised domain classification method which can be used to generalize single-target domain adaptation methods to multi-target domains, and design a weather-invariant object detector training framework based on it.  

* [StyleLess layer: Improving robustness for real-world driving](https://arxiv.org/pdf/2103.13905.pdf)  

* [GridDehazeNet+: An Enhanced Multi-Scale Network with Intra-Task Knowledge Transfer for Single Image Dehazing](https://arxiv.org/pdf/2103.13998.pdf) 

* [Hierarchical Deep CNN Feature Set-Based Representation Learning for Robust Cross-Resolution Face Recognition](https://arxiv.org/pdf/2103.13851.pdf) (Guo-Jun Qi, TCSVT)

* [Self-Supervised Training Enhances Online Continual Learning](https://arxiv.org/pdf/2103.14010.pdf)

  

#### 20210325
CVPR21: 
* [M3DSSD: Monocular 3D Single Stage Object Detector](https://arxiv.org/pdf/2103.13164.pdf) 1. feature mismatching -> shape alignment, center alignment  2. global information -> asymmetric non-local attention  
* [Temporal Context Aggregation Network for Temporal Action Proposal Refinement](https://arxiv.org/pdf/2103.13141.pdf) Temporal action proposal generation aims to estimate temporal intervals of actions in untrimmed videos.  
* [Learning Salient Boundary Feature for Anchor-free Temporal Action Localization](https://arxiv.org/pdf/2103.13137.pdf) Temporal action localization aims at inferring both the action category and localization of the start and end frame for each action instance in a long, untrimmed video.  
* [Revamping Cross-Modal Recipe Retrieval with Hierarchical Transformers and Self-supervised Learning](https://arxiv.org/pdf/2103.13061.pdf) Cross-modal recipe retrieval,包含image-to-recipe和recipe-to-image, recipe is in text. [code](https://github.com/amzn/image-to-recipe-transformers)  
* [Coarse-to-Fine Domain Adaptive Semantic Segmentation with Photometric Alignment and Category-Center Regularization](https://arxiv.org/pdf/2103.13041.pdf) 认为domain shift主要发生在image-level和category-level两个层面，并分别提出a photometric alignment module和a category-oriented triplet loss for source, a self-supervised consistency regularization for target  
* [From Shadow Generation to Shadow Removal](https://arxiv.org/pdf/2103.12997.pdf) Follow现有low-level使用图像翻译方法这一大方向，训练不需要shadow free image。  
* [Relation-aware Instance Refinement for Weakly Supervised Visual Grounding](https://arxiv.org/pdf/2103.12989.pdf) Visual grounding, which aims to build a correspondence between visual objects and their language entities.  
* [Scene-Intuitive Agent for Remote Embodied Visual Grounding](https://arxiv.org/pdf/2103.12944.pdf) an agent that mimics human behaviors: The agent learns where to stop in the Scene Grounding task and what to attend to in the Object Grounding task respectively.  
* [Efficient Regional Memory Network for Video Object Segmentation](https://arxiv.org/pdf/2103.12934.pdf) Regional Memory Network: a novel local-to-local matching solution for semi-supervised VOS   
* [Weakly Supervised Instance Segmentation for Videos with Temporal Mask Consistency](https://arxiv.org/pdf/2103.12886.pdf) Problems in weakly supervised instance segmentation:(a) partial segmentation of objects and (b) missing object predictions. We are the first to explore the use of these video signals to tackle weakly supervised instance segmentation. Keys: 1. inter-pixel relation network 2. MaskConsist module (Alan Yuille)   
* [Convex Online Video Frame Subset Selection using Multiple Criteria for Data Efficient Autonomous Driving](https://arxiv.org/pdf/2103.13021.pdf) 自动驾驶  
* [Dynamic Slimmable Network](https://arxiv.org/pdf/2103.13258.pdf) Problem: dynamic sparse patterns on convolutional filters fail to achieve actual acceleration in real-world implementation, due to the extra burden of indexing, weight-copying, or zero-masking. Method: 1. double-headed dynamic gate that comprises an attention head and a slimming head 2. a disentangled two-stage training scheme inspired by one-shot NAS  
* [Affective Processes: stochastic modelling of temporal context for emotion and facial expression recognition](https://arxiv.org/pdf/2103.13372.pdf)
* [Structure-Aware Face Clustering on a Large-Scale Graph with 10^7 Nodes](https://arxiv.org/pdf/2103.13225.pdf)
* [The Blessings of Unlabeled Background in Untrimmed Videos](https://arxiv.org/pdf/2103.13183.pdf) 因果推断做Weakly-supervised Temporal Action Localization. While previous works treat the background as “curses”, we consider it as “blessings” (Hanwang Zhang)  

其他：
* [Can Vision Transformers Learn without Natural Images?](https://arxiv.org/pdf/2103.13023.pdf) 

  


#### 20210324
CVPR21：
* [Lifelong Person Re-Identification via Adaptive Knowledge Accumulation](https://arxiv.org/pdf/2103.12462.pdf) 提出lifelong person re-identification任务，要求模型在多个域上持续学习，并能泛化到没见过的域上。提出研究此问题的数据集和针对性的解决方案Adaptive Knowledge Accumulation framework. [code](https://github.com/TPCD/LifelongReID)  
* [Group-aware Label Transfer for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/2103.12366.pdf) 
* [Transferable Semantic Augmentation for Domain Adaptation](https://arxiv.org/pdf/2103.12562.pdf) 语义增强做域适应（王雨霖） [code](https://github.com/BIT-DA/TSA)    
* [MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition](https://arxiv.org/pdf/2103.12579.pdf) 语义增强做长尾识别（王雨霖） [code](https://github.com/BIT-DA/MetaSAug)   

其他：
* [Learning without Seeing nor Knowing: Towards Open Zero-Shot Learning](https://arxiv.org/pdf/2103.12437.pdf) 

* [BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search](https://arxiv.org/pdf/2103.12424.pdf)  

* [Global Correlation Network: End-to-End Joint Multi-Object Detection and Tracking](https://arxiv.org/pdf/2103.12511.pdf)   

* [End-to-End Trainable Multi-Instance Pose Estimation with Transformers](https://arxiv.org/pdf/2103.12115.pdf)  

  


#### 20210323


* [Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking](https://arxiv.org/pdf/2103.11681.pdf) 将transformer运用到vision tracking任务上，将transformer的encoder和decoder分解为两个平行的分支，嵌入一个Siamese-like tracking pipelines（周文罡，李厚强）  
* [DeepViT: Towards Deeper Vision Transformer](https://arxiv.org/pdf/2103.11886.pdf) 指出ViT不能像CNN那样通过增加深度提升性能(the attention collapse issue)，并基于以上观察提出Re-attention，重新生成具有多样性的attention map（Jiashi Feng）   
* [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/pdf/2103.11816.pdf) Convolution-enhanced image Transformer (CeiT)将CNN设计理念引入ViT，提出（1）Image-to-Tokens (I2T) module that extracts patches from generated low-level features（2）Locally-enhanced Feed-Forward (LeFF)提升相邻token之间的关联性（3）Layer-wise Class token Attention (LCA)，提升ViT性能和训练速度。（刘子纬）  
* [Multimodal Motion Prediction with Stacked Transformers](https://arxiv.org/pdf/2103.11624.pdf) transformer做Multimodal Motion Prediction（交通流预测）（周博磊）  
* [Learning Multi-Scene Absolute Pose Regression with Transformers](https://arxiv.org/pdf/2103.11468.pdf) transformer做Multi-Scene Absolute Pose Regression（从采集的图片上判断相机的位置和朝向）  



#### 20210322
CVPR21:  
* [Generic Perceptual Loss for Modeling Structured Output Dependencies](https://arxiv.org/pdf/2103.10571.pdf) 指出perceptual loss起作用的原因不是网络预训练的权重，而是网络本身的拓扑结构，使用随机初始化的网络计算perceptual loss在语义分割、深度估计、实例分割等任务上取得良好效果（Chunhua Shen）  
* [CDFI: Compression-Driven Network Design for Frame Interpolation](https://arxiv.org/pdf/2103.10559.pdf) Compression-Driven Network Design for Frame Interpolation  
* [Skeleton Merger: an Unsupervised Aligned Keypoint Detector](https://arxiv.org/pdf/2103.10814.pdf) an Unsupervised Aligned Keypoint Detector （卢策吾）  
* [XProtoNet: Diagnosis in Chest Radiography with Global and Local Explanations](https://arxiv.org/pdf/2103.10663.pdf) 医学诊断  
* [Learning the Superpixel in a Non-iterative and Lifelong Manner](https://arxiv.org/pdf/2103.10681.pdf) 将superpixel segmentation看作lifelong clustering task，提出一个CNN-based superpixel segmentation方法  
* [Degrade is Upgrade: Learning Degradation for Low-light Image Enhancement](https://arxiv.org/pdf/2103.10621.pdf) 低光照图像增强  
* [Dynamic Transfer for Multi-Source Domain Adaptation](https://arxiv.org/pdf/2103.10583.pdf) 用Dynamic Network做Multi-Source Domain Adaptation  
* [Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark](https://arxiv.org/pdf/2103.10895.pdf) 下水道缺陷检测benchmark  

Vision Transofomer：  
* [Reading Isn't Believing: Adversarial Attacks On Multi-Modal Neurons](https://arxiv.org/ftp/arxiv/papers/2103/2103.10480.pdf) 多模态预训练模型的对抗鲁棒性（主要基于CLIP做研究）  
* [UNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/pdf/2103.10504.pdf) Transformers for 3D Medical Image Segmentation  
* [3D Human Pose Estimation with Spatial and Temporal Transformers](https://arxiv.org/pdf/2103.10455.pdf) pure transformer做3D Human Pose Estimation  
* [Hopper: Multi-hop Transformer for Spatiotemporal Reasoning](https://arxiv.org/pdf/2103.10574.pdf) transformer做Spatiotemporal Reasoning（ICLR21）  
* [Scalable Visual Transformers with Hierarchical Pooling](https://arxiv.org/pdf/2103.10619.pdf) 指出ViT ，DeiT在整个inference过程中使用固定长度的sequence可能是冗余的，提出Hierarchical Visual Transformer (HVT)，采用类似CNN池化的方式对sequence长度做下采样，并对depth/width/resolution/patch size等维度做scaling。发现平均池化做全局信息聚合的判别性好于cls token  
* [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/pdf/2103.10697.pdf) ConViT 提出gated positional self-attention (GPSA)，以一种“soft”的方式在vision transformer中引入CNN的inductive bias  
* [MDMMT: Multidomain Multimodal Transformer for Video Retrieval](https://arxiv.org/pdf/2103.10699.pdf) 用transformer做text to video Retrieval  

其他：  
* [Robustness via Cross-Domain Ensembles](https://arxiv.org/pdf/2103.10919.pdf) Robustness via Cross-Domain Ensembles基不确定度于对多个域上的输出做ensemble提升模型鲁棒性   
* [Paint by Word](https://arxiv.org/pdf/2103.10951.pdf) 用word引导inpainting内容  
* [Boosting Adversarial Transferability through Enhanced Momentum](https://arxiv.org/pdf/2103.10609.pdf) （胡瀚，王井东）  
* [UniMoCo: Unsupervised, Semi-Supervised and Full-Supervised Visual Representation Learning](https://arxiv.org/pdf/2103.10773.pdf) 提出UniMoCo，将无监督、半监督和全监督任务结合，基于MoCo做表征学习  
* [ClawCraneNet: Leveraging Object-level Relation for Text-based Video Segmentation](https://arxiv.org/pdf/2103.10702.pdf) 提出一种top-down方法做Text-based Video Segmentation（杨易）  

#### 20210319
* [Generating Diverse Structure for Image Inpainting With Hierarchical VQ-VAE](https://arxiv.org/pdf/2103.10022.pdf) Diverse Image Inpainting (刘东，李厚强，CVPR2021)    
* [Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks](https://arxiv.org/pdf/2103.10429.pdf)  （CVPR2021）    
* [Learning to Recommend Frame for Interactive Video Object Segmentation in the Wild](https://arxiv.org/pdf/2103.10391.pdf) 交互式视频分割 （CVPR2021）   
* [Large Scale Image Completion via Co-Modulated Generative Adversarial Networks](https://arxiv.org/pdf/2103.10428.pdf) GAN Image Completion (ICLR21)   
* [Using latent space regression to analyze and leverage compositionality in GANs](https://arxiv.org/pdf/2103.10426.pdf) 用latent space回归理解和运用的GAN组成 (ICLR21, Phillip Isola)    
* [Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring](https://arxiv.org/pdf/2103.09962.pdf) 将Wiener反卷积与深度学习结合做图像去模糊 (NIPS20)  
* [Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/pdf/2103.09950.pdf) 可学习的图像resize，虽然视觉效果不好，但显著提升ImageNet分类即下游任务的性能 (Google Research)   
* [The Untapped Potential of Off-the-Shelf Convolutional Neural Networks](https://arxiv.org/pdf/2103.09891.pdf) 在测试时dynamically改变预训练网络的topology做预测，能将ResNet50 top1准确率提升到95% （陈怡然）  
* [The Low-Rank Simplicity Bias in Deep Networks](https://arxiv.org/pdf/2103.10427.pdf) 指出DNN倾向于学到lower rank solutions，因此不会overfit训练集，具有较好泛化能力   
* [DanceNet3D: Music Based Dance Generation with Parametric Motion Transformer](https://arxiv.org/pdf/2103.10206.pdf) transformer从音乐中生成舞蹈视频    
* [Consistency-based Active Learning for Object Detection](https://arxiv.org/pdf/2103.10374.pdf) acitve learning做目标检测   
* [SG-Net: Spatial Granularity Network for One-Stage Video Instance Segmentation](https://arxiv.org/pdf/2103.10284.pdf) One-Stage Video Instance Segmentation   
* [Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser](https://arxiv.org/pdf/2103.10234.pdf) 做真实场景去噪（Wangmeng Zuo）   
* [Decoupled Spatial Temporal Graphs for Generic Visual Grounding](https://arxiv.org/pdf/2103.10191.pdf) 提出Generic Visual Grounding，更接近真实场景的Visual grounding问题设置，并提出新的数据集和方法 (程明明，杨易)    
* [Space-Time Crop & Attend: Improving Cross-modal Video Representation Learning](https://arxiv.org/pdf/2103.10211.pdf) 指出自监督视频特征表示学习中，spatial augmentations如cropping十分重要，提出Feature Crop并利用transformer-based attention替代平均池化    
* [TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation](https://arxiv.org/pdf/2103.10158.pdf) 提出一个简单有效的自动数据增强方法TrivialAugment    
* [Self-Supervised Adaptation for Video Super-Resolution](https://arxiv.org/pdf/2103.10081.pdf) 自监督adaptation做视频超分辨   
* [Enhancing Transformer for Video Understanding Using Gated Multi-Level Attention and Temporal Adversarial Training](https://arxiv.org/pdf/2103.10043.pdf) Transformer做Video Understanding    
* [RangeDet: In Defense of Range View for LiDAR-based 3D Object Detection](https://arxiv.org/pdf/2103.10039.pdf) 基于Lidar的3D目标检测（Naiyan Wang）  
* [SparsePoint: Fully End-to-End Sparse 3D Object Detector](https://arxiv.org/pdf/2103.10042.pdf) Follow DETR和SparseRCNN做Sparse的3D目标检测

#### 20210318
CVPR21：  
* [Learning Discriminative Prototypes with Dynamic Time Warping](https://arxiv.org/pdf/2103.09458.pdf) 做Dynamic Time Warping（视频summary）  
* [You Only Look One-level Feature](https://arxiv.org/pdf/2103.09460.pdf) YOLOF提出FPN的成功来自于其divide-and-conquer的设计，而非多尺度特征融合。YOLOF采用Dilated Encoder和Uniform Matching（孙剑，张祥宇）   

其他：  
* [Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network](https://arxiv.org/pdf/2103.09377.pdf) Multi-Prize Lottery Ticket Hypothesis，在Lottery Ticket Hypothesis基础上提出另外两个假设，实现Binary Neural Networks剪枝（ICLR21）  
* [Training GANs with Stronger Augmentations via Contrastive Discriminator](https://arxiv.org/pdf/2103.09742.pdf) 在GAN中引入对比学习（ICLR21）  
* [Bio-inspired Robustness: A Review](https://arxiv.org/ftp/arxiv/papers/2103/2103.09265.pdf) 生物启发鲁棒性综述  
* [Pros and Cons of GAN Evaluation Measures: New Developments](https://arxiv.org/pdf/2103.09396.pdf) GAN近期发展综述  
* [LightningDOT: Pre-training Visual-Semantic Embeddings for Real-Time Image-Text Retrieval](https://arxiv.org/pdf/2103.08784.pdf) vision language预训练，替换耗时的跨模态attention，实现快速Image-Text Retrieval （NAACL）  
* [Revisiting the Loss Weight Adjustment in Object Detection](https://arxiv.org/pdf/2103.09488.pdf) 探讨如何对目标检测中分类损失和定位损失做合适的加权，提出Adaptive Loss Weight Adjustment(ALWA)，在训练过程中根据各损失的统计特性自适应调整损失权重  
* [Large-Scale Zero-Shot Image Classification from Rich and Diverse Textual Descriptions](https://arxiv.org/pdf/2103.09669.pdf) 在zero-shot learning（ZSL）中加入丰富的文本描述，显著提升ZSL性能（NAACL）  
* [Single Underwater Image Restoration by Contrastive Learning](https://arxiv.org/pdf/2103.09697.pdf) 对比学习和图像翻译结合做水下图像增强  
* [Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer](https://arxiv.org/pdf/2103.09712.pdf) transformer做医学手术阶段识别  
* [Prediction-assistant Frame Super-Resolution for Video Streaming](https://arxiv.org/pdf/2103.09455.pdf) 视频压缩和超分，以实现高效的数据传输  
* [Disentangled Cycle Consistency for Highly-realistic Virtual Try-On](https://arxiv.org/pdf/2103.09479.pdf) Disentangled Cycle Consistency for Highly-realistic Virtual Try-On（Ping Luo）  
* [PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](https://arxiv.org/pdf/2103.09504.pdf) 做Spatiotemporal Predictive Learning（视频帧预测）（龙明盛）  


#### 20210317
CVPR21：  
* [Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection](https://arxiv.org/pdf/2103.09096.pdf) 提出Frequency-aware特征学习和新的损失函数做Face Forgery Detection（Yongdong Zhang）  
* [BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation](https://arxiv.org/pdf/2103.08907.pdf) 弱监督实例分割  
* [Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation](https://arxiv.org/pdf/2103.08896.pdf) 弱监督语义分割  
* [Track to Detect and Segment: An Online Multi-Object Tracker](https://arxiv.org/pdf/2103.08808.pdf) Multi-Object Tracking  

其他：  
* [Multilingual Multimodal Pre-training for Zero-Shot Cross-Lingual Transfer of Vision-Language Models](https://arxiv.org/pdf/2103.08849.pdf) 多语言的vision language预训练模型（NAACL）  
* [Is it Enough to Optimize CNN Architectures on ImageNet?](https://arxiv.org/pdf/2103.09108.pdf) 探讨ImageNet预训练CNN的泛化性  
* [Balancing Biases and Preserving Privacy on Balanced Faces in the Wild](https://arxiv.org/pdf/2103.09118.pdf)（TPAMI，Yun Fu）  
* [QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection](https://arxiv.org/pdf/2103.09136.pdf) 在FPN中加入object query提升对小物体的检测（Naiyan Wang）  
* [Dense Interaction Learning for Video-based Person Re-identification](https://arxiv.org/pdf/2103.09013.pdf) transormer做视频reid（Zhibo Chen, Xian-Sheng Hua）  
* [Super-Resolving Cross-Domain Face Miniatures by Peeking at One-Shot Exemplar](https://arxiv.org/pdf/2103.08863.pdf) 跨域人脸超分辨（杨易）  
* [A Large-Scale Dataset for Benchmarking Elevator Button Segmentation and Character Recognition](https://arxiv.org/pdf/2103.09030.pdf) 电梯按钮识别数据集  


#### 20210316
CVPR21：  
* [Detecting Human-Object Interaction via Fabricated Compositional Learning](https://arxiv.org/pdf/2103.08214.pdf) 指出并应对HOI中的长尾分布问题（侯志，陶老师）  
* [Beyond Image to Depth: Improving Depth Prediction using Echoes](https://arxiv.org/pdf/2103.08468.pdf) 从回声和RGB图像中预测深度（多模态）  
* [Semi-Supervised Video Deraining with Dynamic Rain Generator](https://arxiv.org/pdf/2103.07939.pdf) 半监督视频去雨  
* [DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network](https://arxiv.org/pdf/2103.07893.pdf) 对比学习提升条件GAN图像合成的多样性（CHHK）  
* [Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion](https://arxiv.org/pdf/2103.07941.pdf) 交互式视频分割  
* [ReDet: A Rotation-equivariant Detector for Aerial Object Detection](https://arxiv.org/pdf/2103.07733.pdf) Rotation-equivariant Detector for Aerial Object Detection  
* [Uncertainty-guided Model Generalization to Unseen Domains](https://arxiv.org/pdf/2103.07531.pdf) 基于不确定度评估做single domain generalization  
* [Cross-Domain Similarity Learning for Face Recognition in Unseen Domains](https://arxiv.org/pdf/2103.07503.pdf) 人脸识别域泛化（Yi-Hsuan Tsai）  
* [Refine Myself by Teaching Myself : Feature Refinement via Self-Knowledge Distillation](https://arxiv.org/pdf/2103.08273.pdf) 采用类似BiFPN的连接方式做知识蒸馏  

其他：  
* [TransFG: A Transformer Architecture for Fine-grained Recognition](https://arxiv.org/pdf/2103.07976.pdf) transformer细粒度分类（Alan Yuille组）  
* [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/pdf/2103.07579.pdf) 改进resnet，以更快的推理速度取得和EfficientNet相当的性能（Tsung-Yi Lin）  
* [Unsupervised Image Transformation Learning via Generative Adversarial Networks](https://arxiv.org/pdf/2103.07751.pdf) GAN做图像变换（周博磊组）  
* [Learning Frequency-aware Dynamic Network for Efficient Super-Resolution](https://arxiv.org/pdf/2103.08357.pdf) Frequency-aware Dynamic Network for Efficient Super-Resolution （王云鹤）  


#### 20210312
CVPR21：  
* [CoMoGAN: continuous model-guided image-to-image translation](https://arxiv.org/pdf/2103.06879.pdf) CoMoGAN做连续图像翻译  
* [Fast and Accurate Model Scaling](https://arxiv.org/pdf/2103.06877.pdf) FAIR组对标谷歌efficientnet，指出现有模型scaling方法只考虑准确率和计算量，忽视了实际的运行时间。通过限制网络的activation数目，实现Fast and Accurate Model Scaling  
* [Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations](https://arxiv.org/pdf/2103.06342.pdf) 持续学习语义分割  

其他：  
* [WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training](https://arxiv.org/pdf/2103.06561.pdf) 人大、中科院大型vision language预训练模型WenLan，对标CLIP  
* [Level-aware Haze Image Synthesis by Self-Supervised Content-Style Disentanglement](https://arxiv.org/pdf/2103.06501.pdf) 基于自监督图像解耦的雾天图像合成  


#### 20210311
CVPR21:  
* [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/pdf/2103.06255.pdf) 内卷网络（self-attention变种）  
* [Spatially Consistent Representation Learning](https://arxiv.org/pdf/2103.06122.pdf) 能应对dense prediction task的对比学习方法  
* [Reformulating HOI Detection as Adaptive Set Prediction](https://arxiv.org/pdf/2103.05983.pdf) HOI transformer  
* [FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding](https://arxiv.org/pdf/2103.05950.pdf) 对比学习做few shot目标检测  
* [VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples](https://arxiv.org/pdf/2103.05905.pdf) VideoMoCo对比学习做视频任务  
* [Capturing Omni-Range Context for Omnidirectional Segmentation](https://arxiv.org/pdf/2103.05687.pdf) 用attention做广角镜头下的语义分割  
* [AutoDO: Robust AutoAugment for Biased Data with Label Noise via Scalable Probabilistic Implicit Differentiation](https://arxiv.org/pdf/2103.05863.pdf) AutoAugmentation升级版，能处理噪声数据  

其他：
* [Regressive Domain Adaptation for Unsupervised Keypoint Detection](https://arxiv.org/pdf/2103.06175.pdf) 回归域适应关键点检测（龙明盛组）  
* [U-Net Transformer: Self and Cross Attention for Medical Image Segmentation](https://arxiv.org/pdf/2103.06104.pdf) UNet Transformer  


#### 20210310
CVPR2021：  
* [MetaCorrection: Domain-aware Meta Loss Correction for Unsupervised Domain Adaptation in Semantic Segmentation](https://arxiv.org/pdf/2103.05254.pdf) 语义分割域适应  
* [ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection](https://arxiv.org/pdf/2103.05346.pdf) 3D目标检测域适应  
* [Contrastive Neural Architecture Search with Neural Architecture Comparators](https://arxiv.org/pdf/2103.05471.pdf) contrastive NAS  

其他：  
* [Deep Learning based 3D Segmentation: A Survey](https://arxiv.org/pdf/2103.05423.pdf) 3D语义分割综述  


#### 20210309
CVPR21：  
* [Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation](https://arxiv.org/pdf/2103.04717.pdf) multi-source语义分割域适应  
* [Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation](https://arxiv.org/pdf/2103.04705.pdf) 半监督语义分割域适应  
* [MeGA-CDA: Memory Guided Attention for Category-Aware Unsupervised Domain Adaptive Object Detection](https://arxiv.org/pdf/2103.04224.pdf) 目标检测域适应  
* [End-to-End Human Object Interaction Detection with HOI Transformer](https://arxiv.org/pdf/2103.04503.pdf) human object interaction transformer  

其他：  
* [Unsupervised Pretraining for Object Detection by Patch Reidentification](https://arxiv.org/pdf/2103.04814.pdf) 自监督目标检测  
* [Perspectives and Prospects on Transformer Architecture for Cross-Modal Tasks with Language and Vision](https://arxiv.org/pdf/2103.04037.pdf) 探讨多模态transformer  
* [TransBTS: Multimodal Brain Tumor Segmentation Using Transformer](https://arxiv.org/pdf/2103.04430.pdf) 医学分割transformer  

#### 20210302
* [Generative Adversarial Transformers](https://arxiv.org/pdf/2103.01209.pdf) transformer GAN  
* [OmniNet: Omnidirectional Representations from Transformers](https://arxiv.org/pdf/2103.01075.pdf) OmniNet让transformer能够建模网络不同层特征的全局信息，在cv和nlp任务上均有效  
* [Single-Shot Motion Completion with Transformer](https://arxiv.org/pdf/2103.00776.pdf) 用transformer做Single-Shot Motion Completion  
* [Transformer in Transformer](https://arxiv.org/pdf/2103.00112.pdf) 基于vit，用一个inner transformer建模pixel级别的特征表示  


## NLP (Weekly)
