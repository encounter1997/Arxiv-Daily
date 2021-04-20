# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

<font color=LightSkyBlue>The update will be suspended for two weeks ultil April 17,  due to the recent rush of conference paper submission deadline. Thanks for understanding.</font>


## CV (Daily)
#### 20210419
CVPR21:
* [Fusing the Old with the New: Learning Relative Camera Pose with Geometry-Guided Uncertainty](https://arxiv.org/pdf/2104.08278.pdf)
* [Divide-and-Conquer for Lane-Aware Diverse Trajectory Prediction](https://arxiv.org/pdf/2104.08277.pdf) Our work addresses two key challenges in trajectory prediction, learning multimodal outputs, and better predictions by imposing constraints using driving knowledge.
* [Ego-Exo: Transferring Visual Representations from Third-person to First-person Videos](https://arxiv.org/pdf/2104.07905.pdf)  We introduce an approach for pre-training egocentric video models using large-scale third-person video datasets. Learning from purely egocentric data is limited by low dataset scale and diversity, while using purely exocentric (third-person) data introduces a large domain mismatch.   Funny

Others:
* [Deep Stable Learning for Out-Of-Distribution Generalization](https://arxiv.org/pdf/2104.07876.pdf) 问题设置：探究更开放的域适应问题 Conventional methods assume either the known heterogeneity of training data (e.g. domain labels) or the approximately equal capacities of different domains. In this paper, we consider a more challenging case where neither of the above assumptions holds. 采用类似因果的解决方案 We propose to address this problem by removing the dependencies between features via learning weights for training samples, which helps deep models get rid of spurious correlations and, in turn, concentrate more on the true connection between discriminative features and labels.
* [“BNN - BN = ?”: Training Binary Neural Networks without Batch Normalization](https://arxiv.org/pdf/2104.08215.pdf)  问题：However, the BN layer is costly to calculate and is typically implemented with non-binary parameters, leaving a hurdle for the efficient implementation of BNN training. It also introduces undesirable dependence between samples
within each batch. 工作：Inspired by the latest advance on Batch Normalization Free (BN-Free) training [7], we extend their framework to training BNNs, and for the first time demonstrate that BNs can be completed removed from BNN training and inference regimes. (Zhangyang Wang) (CVPRW)
* [Dual Contrastive Learning for Unsupervised Image-to-Image Translation](https://arxiv.org/pdf/2104.07689.pdf) 背景：Contrastive learning for Unpaired image-to-image Translation (CUT) yields state-of-the-art results in modeling unsupervised image-toimage translation by maximizing mutual information between input and output patches using only one encoder for both domains.   贡献：In this paper, we propose a novel method based on contrastive learning and a dual learning setting (exploiting two encoders) to infer an efficient mapping between unpaired data. Additionally, while CUT suffers from mode collapse, a variant of our method efficiently addresses this issue.
* [Contrastive Learning with Stronger Augmentations](https://arxiv.org/pdf/2104.07713.pdf) 现有对比学习问题：However, those carefully designed transformations limited us to further explore the novel patterns exposed by other transformations. Meanwhile, as found in our experiments, the strong augmentations distorted the images’ structures, resulting in difficult retrieval.  方法：Thus, we propose a general framework called Contrastive Learning with Stronger Augmentations (CLSA) to complement current contrastive learning approaches. Here, the distribution divergence between the weakly and strongly augmented images over the representation bank is adopted to supervise the retrieval of strongly augmented queries from a pool of instances. (Guojun Qi)
* [Meta Faster R-CNN: Towards Accurate Few-Shot Object Detection with Attentive Feature Alignment](https://arxiv.org/pdf/2104.07719.pdf)  We propose a meta-learning based few-shot object detection method by transferring meta-knowledge learned from data-abundant base classes to data-scarce novel classes.      To improve proposal generation for few-shot novel classes, we propose to learn a lightweight matching network to measure the similarity between each spatial position in the query image feature map and spatially-pooled class features, instead of the traditional object/nonobject classifier, thus generating category-specific proposals and improving proposal recall for novel classes. (Shih-Fu Chang)
* [Pareto Self-Supervised Training for Few-Shot Learning](https://arxiv.org/pdf/2104.07841.pdf) 探究few-shot learning和自监督学习的结合。 问题：Previous works benefit from sharing inductive bias between the main task (FSL) and auxiliary tasks (SSL), where the shared parameters of tasks are optimized by minimizing a linear combination of task losses. However, it is challenging to select a proper weight to balance tasks and reduce task conflict. 方法：To handle the problem as a whole, we propose a novel approach named as Pareto self-supervised training (PSST) for FSL. PSST explicitly decomposes the few-shot auxiliary problem into multiple constrained multi-objective subproblems with different trade-off preferences, and here a preference region in which the main task achieves the best performance is identified. Then, an effective preferred Pareto exploration is proposed to find a set of optimal solutions in such a preference region. 
* [Weakly Supervised Object Localization and Detection: A Survey](https://arxiv.org/pdf/2104.07918.pdf) (Ming-Hsuan Yang)
* [Self-supervised Video Retrieval Transformer Network](https://arxiv.org/pdf/2104.07993.pdf)任务及其应用Content-based video retrieval aims to find videos from a large video database that are similar to or even nearduplicate of a given query video. It plays an important role in many video related applications, including copyright protection, recommendation, filtering and etc.. 方法： We propose a novel video retrieval system, termed SVRTN, It first applies self-supervised training to effectively learn video representation from unlabeled data to avoid the expensive cost of manual annotation. Then, it exploits transformer structure to aggregate frame-level features into clip-level to reduce both storage space and search complexity. It can learn the complementary and discriminative information from the interactions among clip frames, as well as acquire the frame permutation and missing invariant ability to support more flexible retrieval manners.
* [Spatial-Temporal Correlation and Topology Learning for Person Re-Identification in Videos](https://arxiv.org/pdf/2104.08241.pdf) The key factor for video person reidentification is to effectively exploit both spatial and temporal clues from video sequences. In this work, we propose a novel Spatial-Temporal Correlation and Topology Learning framework (CTL) to pursue discriminative and robust representation by modeling cross-scale spatial-temporal correlation. (Jiawei Liu, Zheng-Jun Zha, Kecheng Zheng)


#### 20210405

CVPR21:

* [Group Collaborative Learning for Co-Salient Object Detection](https://arxiv.org/pdf/2104.01108.pdf) [code](https://github.com/fanq15/GCoNet)  (Deng-Ping Fan, Ling Shao)
* [MOST: A Multi-Oriented Scene Text Detector with Localization Refinement](https://arxiv.org/pdf/2104.01070.pdf)  (Xiang Bai)
* [Visual Semantic Role Labeling for Video Understanding](https://arxiv.org/pdf/2104.00990.pdf)
* [UAV-Human: A Large Benchmark for Human Behavior Understanding with Unmanned Aerial Vehicles](https://arxiv.org/pdf/2104.00946.pdf)
* [Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning](https://arxiv.org/pdf/2104.00924.pdf)
* [Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation](https://arxiv.org/pdf/2104.00905.pdf)
* [Network Quantization with Element-wise Gradient Scaling](https://arxiv.org/pdf/2104.00903.pdf)
* [HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection](https://arxiv.org/pdf/2104.00902.pdf)
* [Adaptive Class Suppression Loss for Long-Tail Object Detection](https://arxiv.org/pdf/2104.00885.pdf)
* [S2R-DepthNet: Learning a Generalizable Depth-specific Structural Representation](https://arxiv.org/pdf/2104.00877.pdf)  (Xuejin Chen, Wenjun Zeng)
* [Self-supervised Video Representation Learning by Context and Motion Decoupling](https://arxiv.org/pdf/2104.00862.pdf)
* [Fully Understanding Generic Objects: Modeling, Segmentation, and Reconstruction](https://arxiv.org/pdf/2104.00858.pdf)
* [Towards High Fidelity Face Relighting with Realistic Shadows](https://arxiv.org/pdf/2104.00825.pdf)
* [Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation](https://arxiv.org/pdf/2104.00808.pdf)
* [FESTA: Flow Estimation via Spatial-Temporal Attention for Scene Point Clouds](https://arxiv.org/pdf/2104.00798.pdf)

Vision Transformer:
* [LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf) 从speed-acc tradeoff的角度讲故CNN与ViT结合，提出attention bias, a new way to integrate positional information in vision transformers.：We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime.   We revisit principles from the extensive literature on convolutional neural networks to apply them to transformers, in particular activation maps with decreasing resolutions. We also introduce the attention bias, a new way to integrate positional information in vision transformers.  As a result, we propose LeVIT: a hybrid neural network for fast inference image classification.  For example, at 80% ImageNet top-1 accuracy, LeViT is 3.3 times faster than EfficientNet on the CPU. 
* [Language-based Video Editing via Multi-Modal Multi-Level Transformer](https://arxiv.org/pdf/2104.01122.pdf)  
* [AAformer: Auto-Aligned Transformer for Person Re-Identification](https://arxiv.org/pdf/2104.00921.pdf)
* [TubeR: Tube-Transformer for Action Detection](https://arxiv.org/pdf/2104.00969.pdf) 
* [TFill: Image Completion via a Transformer-Based Architecture](https://arxiv.org/pdf/2104.00845.pdf)  [code](https://github.com/lyndonzheng/TFill)  (Jianfei Cai)
* [VisQA: X-raying Vision and Language Reasoning in Transformers](https://arxiv.org/pdf/2104.00926.pdf)

Others:
* [Scene Graphs: A Survey of Generations and Applications](https://arxiv.org/pdf/2104.01111.pdf) 
* 

#### 20210402

TOP:
* [Group-Free 3D Object Detection via Transformers](https://arxiv.org/pdf/2104.00678.pdf) In this paper, we present a simple yet effective method for directly detecting 3D objects from the 3D point cloud. Instead of grouping local points to each object candidate, our method computes the feature of an object from all the points in the point cloud with the help of an attention mechanism in the Transformers, where the contribution of each point is automatically learned in the network training. [code](https://github.com/zeliu98/Group-Free-3D)  (Ze Liu, Yue Cao, Han Hu)
* [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf) 考虑训练的Efficiency (1) To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency.   Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller.  (2) we propose an improved method of progressive learning, which adaptively adjusts regularization (e.g., dropout and data augmentation) along with image size. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources.  [code](https://github.com/google/automl/efficientnetv2)   (Mingxing Tan, Quoc V. Le) 
* [UC2: Universal Cross-lingual Cross-modal Vision-and-Language Pre-training](https://arxiv.org/pdf/2104.00332.pdf) To generalize this success to non-English languages, we introduce UC2 , the first machine translation-augmented framework for cross-lingual cross-modal representation learning. (1 ) augment existing English-only datasets with other languages via machine translation (MT) (2) shared visual context (i.e., using image as pivot) (3) To facilitate the learning of a joint embedding space of images and all languages of interest, we further propose two novel pre-training tasks, namely Masked Region-to-Token Modeling (MRTM) and Visual Translation Language Modeling (VTLM), leveraging MT-enhanced translated data.
* [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/pdf/2104.00650.pdf)  Our objective in this work is video-text retrieval – in particular a joint embedding that enables efficient text-to-video retrieval.  We propose an end-to-end trainable model that is designed to take advantage of both large-scale image and video captioning datasets. Our model is an adaptation and extension of the recent ViT and Timesformer architectures, and consists of attention in both space and time.   It is trained with a curriculum learning schedule that begins by treating images as ‘frozen’ snapshots of video, and then gradually learns to attend to increasing temporal context when trained on video datasets. (Andrew Zisserman)
* [Jigsaw Clustering for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/2104.00323.pdf)  有趣的pretext task设计。 We propose a new jigsaw clustering pretext task in this paper, which only needs to forward each training batch itself, and reduces the training cost. Our method makes use of information from both intra- and inter-images, and outperforms previous single-batch based ones by a large margin. It is even comparable to the contrastive learning methods when only half of training batches are used.   Our method indicates that multiple batches during training are not necessary, and opens the door for future research of single-batch unsupervised methods.  [code](https://github.com/Jia-Research-Lab/JigsawClustering)  (Jiaya Jia, CVPR21) 
* [Unsupervised Sound Localization via Iterative Contrastive Learning](https://arxiv.org/pdf/2104.00315.pdf)  Sound localization aims to find the source of the audio signal in the visual scene.  In this work, we propose an iterative contrastive learning framework that requires no data annotations. At each iteration, the proposed method takes the 1) localization results in images predicted in the previous iteration, and 2) semantic relationships inferred from the audio signals as the pseudolabels.   Our iterative strategy gradually encourages the localization of the sounding objects and reduces the correlation between the non-sounding regions and the reference audio. (如何保证基于伪标签的迭代是变好，而非变差？) (Ming-Hsuan Yang)
* [In&Out : Diverse Image Outpainting via GAN Inversion](https://arxiv.org/pdf/2104.00675.pdf) GAN inversion逐渐成为GAN研究的主流方向，本文借GAN inversion做Image outpainting.   Image outpainting seeks for a semantically consistent extension of the input image beyond its available content.   In this work, we formulate the problem from the perspective of inverting generative adversarial networks. Our generator renders micro-patches conditioned on their joint latent code as well as their individual positions in the image. [code](https://github.com/yccyenchicheng/InOut) (Ming-Hsuan Yang)

CVPR21:

* [Online Multiple Object Tracking with Cross-Task Synergy](https://arxiv.org/pdf/2104.00380.pdf)  [code](https://github.com/songguocode/TADAM)  (Dacheng Tao)
* [Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty Estimation for Facial Expression Recognition](https://arxiv.org/pdf/2104.00232.pdf)  (Tao Mei)
* [Divergence Optimization for Noisy Universal Domain Adaptation](https://arxiv.org/pdf/2104.00246.pdf)
* [FAPIS: A Few-shot Anchor-free Part-based Instance Segmenter](https://arxiv.org/pdf/2104.00073.pdf)
* [Self-supervised Motion Learning from Static Images](https://arxiv.org/pdf/2104.00240.pdf)
* [Learning to Track Instances without Video Annotations](https://arxiv.org/pdf/2104.00287.pdf) Tracking segmentation masks of multiple instances has been intensively studied, but still faces two fundamental challenges: 1) the requirement of large-scale, frame-wise annotation, and 2) the complexity of two-stage approaches. 本文利用自监督实现单阶段 with only a labeled image dataset and unlabeled video sequences
* [Improving Calibration for Long-Tailed Recognition](https://arxiv.org/pdf/2104.00466.pdf) [code](https://github.com/Jia-Research-Lab/MiSLAS)  (Jiaya Jia)
* [Towards Evaluating and Training Verifiably Robust Neural Networks](https://arxiv.org/pdf/2104.00447.pdf)  (Dahua Lin)
* [One-Shot Neural Ensemble Architecture Search by Diversity-Guided Search Space Shrinking](https://arxiv.org/pdf/2104.00597.pdf)  (Jianlong Fu)
* [Unsupervised Degradation Representation Learning for Blind Super-Resolution](https://arxiv.org/pdf/2104.00416.pdf) Funmy 构建不同程度降质的图像做对比学习 In this paper, we propose an unsupervised degradation representation learning scheme for blind SR without explicit degradation estimation. Specifically, we learn abstract representations to distinguish various degradations in the representation space rather than explicit estimation in the pixel space. [code](https://github.com/LongguangWang/DASR)
* [Bipartite Graph Network with Adaptive Message Passing for Unbiased Scene Graph Generation](https://arxiv.org/pdf/2104.00308.pdf)  long-tailed class distribution and large intra-class variation. To address these issues, we introduce a novel confidence-aware bipartite graph neural network with adaptive message propagation mechanism for unbiased scene graph generation. In addition, we propose an efficient bi-level data resampling strategy to alleviate the imbalanced data distribution problem in training our graph network. 
* [A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification](https://arxiv.org/pdf/2104.00679.pdf)
* [RGB-D Local Implicit Function for Depth Completion of Transparent Objects](https://arxiv.org/pdf/2104.00622.pdf)
* [SimPoE: Simulated Character Control for 3D Human Pose Estimation](https://arxiv.org/pdf/2104.00683.pdf) 
* [NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video](https://arxiv.org/pdf/2104.00681.pdf)
* [PhySG: Inverse Rendering with Spherical Gaussians for Physics-based Material Editing and Relighting](https://arxiv.org/pdf/2104.00674.pdf)
* [LED2 -Net: Monocular 360◦ Layout Estimation via Differentiable Depth Rendering](https://arxiv.org/pdf/2104.00568.pdf)  Towards reconstructing the room layout in 3D, we formulate the task of 360◦ layout estimation as a problem of predicting depth on the horizon line of a panorama.
* [Reconstructing 3D Human Pose by Watching Humans in the Mirror](https://arxiv.org/pdf/2104.00340.pdf) In this paper, we introduce the new task of reconstructing 3D human pose from a single image in which we can see the person and the person’s image through a mirror. [code](https://github.com/zju3dv/Mirrored-Human)
* [Wide-Depth-Range 6D Object Pose Estimation in Space](https://arxiv.org/pdf/2104.00337.pdf) 有趣的应用 [code](https://github.com/cvlab-epfl/wide-depth-range-pose)
* [Fostering Generalization in Single-view 3D Reconstruction by Learning a Hierarchy of Local and Global Shape Priors](https://arxiv.org/pdf/2104.00476.pdf)
* [Deep Two-View Structure-from-Motion Revisited](https://arxiv.org/pdf/2104.00556.pdf)

Vision Transformer:

* [Group-Free 3D Object Detection via Transformers](https://arxiv.org/pdf/2104.00678.pdf) In this paper, we present a simple yet effective method for directly detecting 3D objects from the 3D point cloud. Instead of grouping local points to each object candidate, our method computes the feature of an object from all the points in the point cloud with the help of an attention mechanism in the Transformers, where the contribution of each point is automatically learned in the network training. [code](https://github.com/zeliu98/Group-Free-3D)  (Ze Liu, Yue Cao, Han Hu)
* [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/pdf/2104.00650.pdf)
* [Spatial-Temporal Graph Transformer for Multiple Object Tracking](https://arxiv.org/pdf/2104.00194.pdf) Tracking multiple objects in videos relies on modeling the spatial-temporal interactions of the objects. In this paper, we propose a solution named Spatial-Temporal Graph Transformer (STGT), which leverages powerful graph transformers to efficiently model the spatial and temporal interactions among the objects.
* [Latent Variable Nested Set Transformers & AutoBots](https://arxiv.org/pdf/2104.00563.pdf) We validate the Nested Set Transformer for autonomous driving settings which we refer to as (“AutoBot”), where we model the trajectory of an ego-agent based on the sequential observations of key attributes of multiple agents in a scene.
* [LoFTR: Detector-Free Local Feature Matching with Transformers](https://arxiv.org/pdf/2104.00680.pdf)  (CVPR21)
* [Mesh Graphormer](https://arxiv.org/pdf/2104.00272.pdf)


Others:

* [The surprising impact of mask-head architecture on novel class segmentation](https://arxiv.org/pdf/2104.00613.pdf) We address the partially supervised instance segmentation problem in which one can train on (significantly cheaper) bounding boxes for all categories but use masks only for a subset of categories.     [code](https://google.github.io/deepmac/) 
* [In&Out : Diverse Image Outpainting via GAN Inversion](https://arxiv.org/pdf/2104.00675.pdf) GAN inversion逐渐成为GAN研究的主流方向，本文借GAN inversion做Image outpainting.   Image outpainting seeks for a semantically consistent extension of the input image beyond its available content.   In this work, we formulate the problem from the perspective of inverting generative adversarial networks. Our generator renders micro-patches conditioned on their joint latent code as well as their individual positions in the image. [code](https://github.com/yccyenchicheng/InOut) (Ming-Hsuan Yang)
* [Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study](https://arxiv.org/pdf/2104.00676.pdf) (ICLR21)
* [CUPID: Adaptive Curation of Pre-training Data for Video-and-Language Representation Learning](https://arxiv.org/pdf/2104.00285.pdf)
* [Composable Augmentation Encoding for Video Representation Learning](https://arxiv.org/pdf/2104.00616.pdf)  To overcome this limitation, we propose an ‘augmentation aware’ contrastive learning framework, where we explicitly provide a sequence of augmentation parameterisations (such as the values of the time shifts used to create data views) as composable augmentation encodings (CATE) to our model when projecting the video representations for contrastive learning. 
* [Text to Image Generation with Semantic-Spatial Aware GAN](https://arxiv.org/pdf/2104.00567.pdf)
* [Linear Semantics in Generative Adversarial Networks](https://arxiv.org/pdf/2104.00487.pdf)
* [Unsupervised Foreground-Background Segmentation with Equivariant Layered GANs](https://arxiv.org/pdf/2104.00483.pdf)
* [Improved Image Generation via Sparse Modeling](https://arxiv.org/pdf/2104.00464.pdf)
* [Exploiting Relationship for Complex-scene Image Generation](https://arxiv.org/pdf/2104.00356.pdf) (Tao Mei)
* [MeanShift++: Extremely Fast Mode-Seeking With Applications to Segmentation and Object Tracking](https://arxiv.org/pdf/2104.00303.pdf)
* [SCALoss: Side and Corner Aligned Loss for Bounding Box Regression](https://arxiv.org/pdf/2104.00462.pdf)  IoU-based loss has the gradient vanish problem in the case of low overlapping bounding boxes, and the model could easily ignore these simple cases. In this paper, we propose Side Overlap (SO) loss by maximizing the side overlap of two bounding boxes, which puts more penalty for low overlapping bounding box cases.
* [Anchor Pruning for Object Detection](https://arxiv.org/pdf/2104.00432.pdf)  This paper proposes anchor pruning for object detection in one-stage anchor-based detectors. In this work, we show that many anchors in the object detection head can be removed without any loss in accuracy. With additional retraining, anchor pruning can even lead to improved accuracy.  没引DETR和Sparse RCNN.  (Deng Cai)
* [Modular Adaptation for Cross-Domain Few-Shot Learning](https://arxiv.org/pdf/2104.00619.pdf)
* [A Survey on Natural Language Video Localization](https://arxiv.org/pdf/2104.00234.pdf)



#### 20210401

TOP: 

* [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf)  However the optimization of image transformers has been little studied so far. In this work, we build and optimize deeper transformer networks for image classification.  This leads us to produce models whose performance does not saturate early with more depth, for instance we obtain 86.3% top-1 accuracy on Imagenet when training with no external data  (Facebook, DeiT团队)  
* [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://arxiv.org/pdf/2103.17249.pdf) However, discovering semantically meaningful latent manipulations typically involves painstaking human examination of the many degrees of freedom, or an annotated collection of images for each desired manipulation. In this work, we explore leveraging the power of recently introduced Contrastive Language-Image Pre-training (CLIP) models in order to develop a text-based interface for StyleGAN image manipulation that does not require such manual effort.  [code](https://github.com/orpatashnik/StyleCLIP)
* [PiCIE: Unsupervised Semantic Segmentation using Invariance and Equivariance in Clustering](https://arxiv.org/pdf/2103.17070.pdf) 

CVPR21:

* [Scale-aware Automatic Augmentation for Object Detection](https://arxiv.org/pdf/2103.17220.pdf) [code](https://github.com/Jia-Research-Lab/SA-AutoAug) (Jiaya Jia)
* [Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark](https://arxiv.org/pdf/2103.16746.pdf) Tracking by natural language specification is a new rising research topic that aims at locating the target object in the video sequence based on its language description.  In this work, we propose a new benchmark specifically dedicated to the tracking-by-language, including a large scale dataset, strong and diverse baseline methods.  We also introduce two new challenges into TNL2K for the object tracking task, i.e., adversarial samples and modality switch.   (Feng Wu)
* [SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification](https://arxiv.org/pdf/2103.16725.pdf)
* [Denoise and Contrast for Category Agnostic Shape Completion](https://arxiv.org/pdf/2103.16671.pdf)
* [DAP: Detection-Aware Pre-training with Weak Supervision](https://arxiv.org/pdf/2103.16651.pdf) we transform a classification dataset into a detection dataset through a weakly supervised object localization method based on Class Activation Maps to directly pre-train a detector, making the pre-trained model location-aware and capable of predicting bounding boxes.
* [Unsupervised Disentanglement of Linear-Encoded Facial Semantics](https://arxiv.org/pdf/2103.16605.pdf)
* [ArtFlow: Unbiased Image Style Transfer via Reversible Neural Flows](https://arxiv.org/pdf/2103.16877.pdf)  (Jiebo Luo)
* [Online Learning of a Probabilistic and Adaptive Scene Representation](https://arxiv.org/pdf/2103.16832.pdf)  (Hongbin Zha)
* [Convolutional Hough Matching Networks](https://arxiv.org/pdf/2103.16831.pdf)  
* [Rectification-based Knowledge Retention for Continual Learning](https://arxiv.org/pdf/2103.16597.pdf)
* [Learning Scalable l∞-constrained Near-lossless Image Compression via Joint Lossy Image and Residual Compression](https://arxiv.org/pdf/2103.17015.pdf)
* [Mask-ToF: Learning Microlens Masks for Flying Pixel Correction in Time-of-Flight Imaging](https://arxiv.org/pdf/2103.16693.pdf)
* [Neural Response Interpretation through the Lens of Critical Pathways](https://arxiv.org/pdf/2103.16886.pdf)  (VGG)
* [Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation](https://arxiv.org/pdf/2103.16765.pdf)
* [Dense Relation Distillation with Context-aware Aggregation for Few-Shot Object Detection](https://arxiv.org/pdf/2103.17115.pdf)
* [ReMix: Towards Image-to-Image Translation with Limited Data](https://arxiv.org/pdf/2103.16835.pdf)
* [DER: Dynamically Expandable Representation for Class Incremental Learning](https://arxiv.org/pdf/2103.16788.pdf)
* [GrooMeD-NMS: Grouped Mathematically Differentiable NMS for Monocular 3D Object Detection](https://arxiv.org/pdf/2103.17202.pdf)
* [A Closer Look at Fourier Spectrum Discrepancies for CNN-generated Images Detection](https://arxiv.org/pdf/2103.17195.pdf)
* [Semi-supervised Synthesis of High-Resolution Editable Textures for 3D Humans](https://arxiv.org/pdf/2103.17266.pdf)
* [VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization](https://arxiv.org/pdf/2103.16874.pdf) While an increasing number of studies have been conducted, the resolution of synthesized images is still limited to low (e.g., 256×192), which acts as the critical limitation against satisfying online consumers.   To address the challenges, we propose a novel virtual try-on method called VITON-HD that successfully synthesizes 1024×768 virtual try-on images.
* [Learning Camera Localization via Dense Scene Matching](https://arxiv.org/pdf/2103.16792.pdf)
* [Embracing Uncertainty: Decoupling and De-bias for Robust Temporal Grounding](https://arxiv.org/pdf/2103.16848.pdf)
* [Human POSEitioning System (HPS): 3D Human Pose Estimation and Self-localization in Large Scenes from Body-Mounted Sensors](https://arxiv.org/pdf/2103.17265.pdf) We introduce (HPS) Human POSEitioning System, a method to recover the full 3D pose of a human registered with a 3D scan of the surrounding environment using wearable sensors.
* [Learning by Aligning Videos in Time](https://arxiv.org/pdf/2103.17260.pdf)
* [Dogfight: Detecting Drones from Drones Videos](https://arxiv.org/pdf/2103.17242.pdf)
* [Rainbow Memory: Continual Learning with a Memory of Diverse Samples](https://arxiv.org/pdf/2103.17230.pdf)
* [Layout-Guided Novel View Synthesis from a Single Indoor Panorama](https://arxiv.org/pdf/2103.17022.pdf)


Vision Transformer:

* [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf)  However the optimization of image transformers has been little studied so far. In this work, we build and optimize deeper transformer networks for image classification.  This leads us to produce models whose performance does not saturate early with more depth, for instance we obtain 86.3% top-1 accuracy on Imagenet when training with no external data  (Facebook, DeiT团队)  
* [Learning Spatio-Temporal Transformer for Visual Tracking](https://arxiv.org/pdf/2103.17154.pdf)   The encoder models the global spatio-temporal feature dependencies between target objects and search regions, while the decoder learns a query embedding to predict the spatial positions of the target objects.  The whole method is endto-end, does not need any postprocessing steps such as cosine window and bounding box smoothing, thus largely simplifying existing tracking pipelines. [code](https://github.com/researchmm/Stark)  (Jianlong Fu, Huchuan Lu) 
* [Robust Facial Expression Recognition with Convolutional Visual Transformers](https://arxiv.org/pdf/2103.16854.pdf)  Different from previous pure CNNs based methods, we argue that it is feasible and practical to translate facial images into sequences of visual words and perform expression recognition from a global perspective.  (Shutao Li)
* [DA-DETR: Domain Adaptive Detection Transformer by Hybrid Attention](https://arxiv.org/pdf/2103.17084.pdf) 基于Deformable DETR的域适应目标检测



## NLP (Weekly)
