# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

[2021 April](202104.md)

## CV (Daily)

#### 20210611
- :star: :star: ​[MST: Masked Self-Supervised Transformer for Visual Representation](https://arxiv.org/pdf/2106.05656.pdf)

> 通过attention mask在transformer上实现局部对比学习，对dense的下游任务更有利，性能超越DINO. 
> METHOD: Specifically, inspired by the Masked Language Modeling (MLM) in NLP, we propose a masked token strategy based on the multi-head self-attention map, which dynamically masks some tokens of local patches without damaging the crucial structure for self-supervised learning. More importantly, the masked tokens together with the remaining tokens are further recovered by a global image decoder, which preserves the spatial information of the image and is more friendly to the downstream dense prediction tasks. 


- :star: :star: [Revisiting Contrastive Methods for Unsupervised Learning of Visual Representations](https://arxiv.org/pdf/2106.05967.pdf)  (Luc Van Gool)

> However, current methods are still primarily applied to curated datasets like ImageNet. In this paper, we first study how biases in the dataset affect existing methods. Our results show that an approach like MoCo [22] works surprisingly well across: (i) object- versus scene-centric, (ii) uniform versus long-tailed and (iii) general versus domain-specific datasets.
> Second, given the generality of the approach, we try to realize further gains with minor modifications. We show that learning additional invariances - through the use of multi-scale cropping, stronger augmentations and nearest neighbors - improves the representations.
> Finally, we observe that MoCo learns spatially structured representations when trained with a multi-crop strategy. The representations can be used for semantic segment retrieval and video instance segmentation without finetuning. Moreover, the results are on par with specialized models

- :star: [Learning to See by Looking at Noise ](https://arxiv.org/pdf/2106.05963.pdf) (MIT)

> FUNNY
> Current vision systems are trained on huge datasets, and these datasets come with costs: curation is expensive, they inherit human biases, and there are concerns over privacy and usage rights. To counter these costs, interest has surged in learning from cheaper data sources, such as unlabeled images. In this paper we go a step further and ask if we can do away with real image datasets entirely, instead learning from noise processes. We investigate a suite of image generation models that produce images from simple random processes. These are then used as training data for a visual representation learner with a contrastive loss. 

- :star: [What Does Rotation Prediction Tell Us about Classifier Accuracy under Varying Testing Environments?](https://arxiv.org/pdf/2106.05961.pdf)

> ICML'21  VERY FUNNY
> A natural question then arises: given a trained classifier, can we evaluate its accuracy on varying unlabeled test sets? In this work, we train semantic classification and rotation prediction in a multi-task way. On a series of datasets, we report an interesting finding, i.e., the semantic classification accuracy exhibits a strong linear relationship with the accuracy of the rotation prediction task (Pearson’s Correlation r > 0.88). This finding allows us to utilize linear regression to estimate classifier performance from the accuracy of rotation prediction which can be obtained on the test set through the freely generated rotation labels.

- [Implicit Feature Alignment: Learn to Convert Text Recognizer to Text Spotter](https://arxiv.org/pdf/2106.05920.pdf)

> In this paper, we propose a simple, elegant and effective paradigm called Implicit Feature Alignment (IFA), which can be easily integrated into current text recognizers, resulting in a novel inference mechanism called IFA inference. This enables an ordinary text recognizer to process multi-line text such that text detection can be completely freed. S

- [CAT: Cross Attention in Vision Transformer](https://arxiv.org/pdf/2106.05786.pdf)

> In this paper, we propose a new attention mechanism in Transformer termed Cross Attention, which alternates attention inner the image patch instead of the whole image to capture local information and apply attention between image patches which are divided from single-channel feature maps to capture global information. Both operations have less computation than standard self-attention in Transformer.
> [code](https://github.com/linhezheng19/CAT)

- [Deep neural network loses attention to adversarial images](https://arxiv.org/pdf/2106.05657.pdf)

> 从attention map研究对抗样本的可解释性

- [Space-time Mixing Attention for Video Transformer](https://arxiv.org/pdf/2106.05968.pdf)

> In this work, we propose a Video Transformer model the complexity of which scales linearly with the number of frames in the video sequence and hence induces no overhead compared to an image-based Transformer model.

- [Multi-Dataset Benchmarks for Masked Identification using Contrastive Representation Learning](https://arxiv.org/pdf/2106.05596.pdf)

> 戴口罩情况下的人脸识别

- [Progressive Stage-wise Learning for Unsupervised Feature Representation Enhancement](https://arxiv.org/pdf/2106.05554.pdf)  (Alan Yuille, Bingbing Ni, Wen Gao)

> Progressive Stage-wise Learning (PSL) framework. For a given unsupervised task, we design multilevel tasks and define different learning stages for the deep network. Early learning stages are forced to focus on lowlevel tasks while late stages are guided to extract deeper information through harder tasks.
> 在自监督/无监督学习中引入课程学习/Self-paced Learning

- :star: [Cross-domain Contrastive Learning for Unsupervised Domain Adaptation](https://arxiv.org/pdf/2106.05528.pdf)  (Guo-Jun Qi)

> In this work, we build upon contrastive self-supervised learning to align features so as to reduce the domain discrepancy between training and testing sets. 
> Exploring the same set of categories shared by both domains, we introduce a simple yet effective framework CDCL, for domain alignment. In particular, given an anchor image from one domain, we minimize its distances to cross-domain samples from the same class relative to those from different categories. Since target labels are unavailable, we use a clustering-based approach with carefully initialized centers to produce pseudo labels.
> In addition, we demonstrate that CDCL is a general framework and can be adapted to the data-free setting, where the source data are unavailable during training, with minimal modification

- [Learning to Affiliate: Mutual Centralized Learning for Few-shot Classification](https://arxiv.org/pdf/2106.05517.pdf)  (Deng Cai)

> 现有工作：They generally explore a unidirectional query-to-support paradigm in FSL, e.g., find the nearest/optimal support feature for each query feature and aggregate these local matches for a joint classification. 
> 本文：In this paper, we propose a new method Mutual Centralized Learning (MCL) to fully affiliate the two disjoint sets of dense features in a bidirectional paradigm

- :star: [Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers](https://arxiv.org/pdf/2106.05392.pdf)  (FAIR, Oxford)

> In video transformers, the time dimension is often treated in the same way as the two spatial dimensions. However, in a scene where objects or the camera may move, a physical point imaged at one location in frame t may be entirely unrelated to what is found at that location in frame t + k. These temporal correspondences should be modeled to facilitate learning about dynamic scenes.
> To this end, we propose a new drop-in block for video transformers—trajectory attention—that aggregates information along implicitly determined motion paths. We additionally propose a new method to address the quadratic dependence of computation and memory on the input size, which is particularly important for high resolution or long videos.
> [code](https://github.com/facebookresearch/Motionformer)

- :star: [Beyond BatchNorm: Towards a General Understanding of Normalization in Deep Learning](https://arxiv.org/pdf/2106.05956.pdf)  

> In this work, we take a first step towards this goal by extending known properties of BatchNorm in randomly initialized deep neural networks (DNNs) to nine recently proposed normalization layers.
> Our primary findings follow: (i) Similar to BatchNorm, activations-based normalization layers can avoid exploding activations in ResNets; (ii) Use of GroupNorm ensures rank of activations is at least Ω(pwidth/Group Size), thus explaining why LayerNorm witnesses slow optimization speed; and (iii) Small group sizes result in large gradient norm in earlier layers, hence justifying training instability issues in Instance Normalization and illustrating a speed-stability tradeoff in GroupNorm.


- [AFAN: Augmented Feature Alignment Network for Cross-Domain Object Detection](https://arxiv.org/pdf/2106.05499.pdf)  (Ling Shao)

#### 20210610

* [VALUE: A Multi-Task Benchmark for Video-and-Language Understanding Evaluation](https://arxiv.org/pdf/2106.04632.pdf) 效仿GLUE的video-and-language benchmark: an assemblage of 11 video-and-language datasets over 3 popular tasks: (i) text-to-video retrieval; (ii) video question answering; and (iii) video captioning.
* [Check It Again: Progressive Visual Question Answering via Visual Entailment](https://arxiv.org/pdf/2106.04605.pdf) ACL
* :star: [Distilling Image Classifiers in Object Detectors](https://arxiv.org/pdf/2106.05209.pdf)  Nevertheless, the knowledge distillation literature remains limited to the scenario where the student and the teacher tackle the same task. Here, we investigate the problem of transferring knowledge not only across architectures but also across tasks. To this end, we study the case of object detection and, instead of following the standard detector-to-detector distillation approach, introduce a classifier-to-detector knowledge transfer framework.

#### 20210503

* [A Good Image Generator Is What You Need for High-Resolution Video Synthesis](https://arxiv.org/pdf/2104.15069.pdf)  (ICLR'21)
* [Semantic Relation Preserving Knowledge Distillation for Image-to-Image Translation](https://arxiv.org/pdf/2104.15082.pdf)  (ECCV'20)
* [DriveGAN: Towards a Controllable High-Quality Neural Simulation](https://arxiv.org/pdf/2104.15060.pdf)  (CVPR'21, Oral)
* [Faster Meta Update Strategy for Noise-Robust Deep Learning](https://arxiv.org/pdf/2104.15092.pdf)  (Yi Yang)
* [Updatable Siamese Tracker with Two-stage One-shot Learning](https://arxiv.org/pdf/2104.15049.pdf)
* [Unsupervised Data Augmentation for Object Detection](https://arxiv.org/pdf/2104.14965.pdf)
* [Learning Multi-Granular Hypergraphs for Video-Based Person Re-Identification](https://arxiv.org/pdf/2104.14913.pdf)  (CVPR'20, Ling Shao)
* [BiCnet-TKS: Learning Efficient Spatial-Temporal Representation for Video Person Re-Identification](https://arxiv.org/pdf/2104.14783.pdf)
* [CoCon: Cooperative-Contrastive Learning](https://arxiv.org/pdf/2104.14764.pdf)  
* [MOOD: Multi-level Out-of-distribution Detection](https://arxiv.org/pdf/2104.14726.pdf)

##### vision transformers

* [CAT: Cross-Attention Transformer for One-Shot Object Detection](https://arxiv.org/pdf/2104.14984.pdf)
* [End-to-End Attention-based Image Captioning](https://arxiv.org/pdf/2104.14721.pdf)
* [Chop Chop BERT: Visual Question Answering by Chopping VisualBERT’s Heads](https://arxiv.org/pdf/2104.14741.pdf)
* [HandsFormer: Keypoint Transformer for Monocular 3D Pose Estimation of Hands and Object in Interaction](https://arxiv.org/pdf/2104.14639.pdf)
* [Pyramid Medical Transformer for Medical Image Segmentation](https://arxiv.org/ftp/arxiv/papers/2104/2104.14702.pdf)
* [CoSformer: Detecting Co-Salient Object with Transformers](https://arxiv.org/pdf/2104.14729.pdf)
* [Perceptual Image Quality Assessment with Transformers](https://arxiv.org/pdf/2104.14730.pdf)

