# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

[2021 April](202104.md)

[2021 June](202106.md)

[2021 July](202107.md)

## CV (Daily)

#### 20210805

* [Enhancing Self-supervised Video Representation Learning via Multi-level Feature Optimization](https://arxiv.org/pdf/2108.02183.pdf)  (ICCV'21,  John See)
  * However, most recent works have mainly focused on high-level semantics and neglected lower-level representations and their temporal relationship which are crucial for general video understanding.
  * Concretely, high-level features obtained from naive and prototypical contrastive learning are utilized to build distribution graphs, guiding the process of low-level and mid-level feature learning.    We also devise a simple temporal modeling module from multi-level features to enhance motion pattern learning.
* [Towards Coherent Visual Storytelling with Ordered Image Attention](https://arxiv.org/pdf/2108.02180.pdf)
  * We address the problem of visual storytelling, i.e., generating a story for a given sequence of images. While each sentence of the story should describe a corresponding image, a coherent story also needs to be consistent and relate to both future and past images
* [Armour: Generalizable Compact Self-Attention for Vision Transformers](https://arxiv.org/pdf/2108.01778.pdf)
  * This paper introduces a compact selfattention mechanism that is fundamental and highly generalizable. The proposed method reduces redundancy and improves efficiency on top of the existing attention optimizations. 
  * We show its drop-in applicability for both the regular attention mechanism and some most recent variants in vision transformers
* [Vision Transformer with Progressive Sampling](https://arxiv.org/pdf/2108.01684.pdf)  (ICCV'21)  (Dahua Lin) 
  * STRAIGHTFORWARD: 改进ViT的patch spliting
  * However, such naive tokenization could destruct object structures, assign grids to uninterested regions such as background, and introduce interference signals. To mitigate the above issues, in this paper, we propose an iterative and progressive sampling strategy to locate discriminative regions
* [Generic Neural Architecture Search via Regression](https://arxiv.org/pdf/2108.01899.pdf)
  * These observations inspire us to ask: Is it necessary to use the performance of specific downstream tasks to evaluate and search for good neural architectures? Can we perform NAS effectively and efficiently while being agnostic to the downstream task? 
  * GenNAS does not use task-specific labels but instead adopts regression on a set of manually designed synthetic signal bases for architecture evaluation. Such a self-supervised regression task can effectively evaluate the intrinsic power of an architecture to capture and transform the input signal patterns, and allow more sufficient usage of training samples.

#### 20210804

* [Generalized Source-free Domain Adaptation](https://arxiv.org/pdf/2108.01614.pdf)  (ICCV'21)
  * Some recent works tackle source-free domain adaptation (SFDA) where only a source pre-trained model is available for adaptation to the target domain. However, those methods do not consider keeping source performance which is of high practical value in real world applications. In this paper, we propose a new domain adaptation paradigm called Generalized Source-free Domain Adaptation (G-SFDA), where the learned model needs to perform well on both the target and source domains, with only access to current unlabeled target data during adaptation.
* [Boosting Weakly Supervised Object Detection via Learning Bounding Box Adjusters](https://arxiv.org/pdf/2108.01499.pdf)  (Wangmeng Zuo, ICCV'21)
  * In this paper, we defend the problem setting for improving localization performance by leveraging the bounding box regression knowledge from a well-annotated auxiliary dataset.
* [Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer](https://arxiv.org/pdf/2108.01390.pdf)
  * ViT模型压缩
  * Recent efficient designs for vision transformers follow two pipelines, namely, structural compression based on local spatial prior and non-structural token pruning.  However, token pruning breaks the spatial structure that is indispensable for local spatial prior.
  *  To take advantage of both two pipelines, this work seeks to dynamically identify uninformative tokens for each instance and trim down both the training and inference complexity while maintaining complete spatial structure and information flow
* [Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability](https://arxiv.org/pdf/2108.01335.pdf)  (Funny)
  * Conventional saliency maps highlight input features to which neural network predictions are highly sensitive. We take a different approach to saliency, in which we identify and analyze the network parameters, rather than inputs, which are responsible for erroneous decisions
  * We find that samples which cause similar parameters to malfunction are semantically similar. We also show that pruning the most salient parameters for a wrongly classified sample often improves model behavior. Furthermore, fine-tuning a small number of the most salient parameters on a single sample results in error correction on other samples that are misclassified for similar reasons.
  * 从参数上提升可解释性是否靠谱？（对不同结构模型的参数是否适用？在训练不同阶段是否都适用？对不同输入数据是否都适用？）
* [CanvasVAE: Learning to Generate Vector Graphic Documents](https://arxiv.org/pdf/2108.01249.pdf)  (ICCV'21)
* [Domain Generalization via Gradient Surgery](https://arxiv.org/pdf/2108.01621.pdf)   (ICCV'21)
  * Our hypothesis is that when training with multiple domains, conflicting gradients within each mini-batch contain information specific to the individual domains which is irrelevant to the others, including the test domain. If left untouched, such disagreement may degrade generalization performance.
  * In this work, we characterize the conflicting gradients emerging in domain shift scenarios and devise novel gradient agreement strategies based on gradient surgery to alleviate their effect.
  * 和zhibo chen的UDA文章思想类似
* [Elastic Architecture Search for Diverse Tasks with Different Resources](https://arxiv.org/pdf/2108.01224.pdf)  (Jianfei Cai)
  * We study a new challenging problem of efficient deployment for diverse tasks with different resources, where the resource constraint and task of interest corresponding to a group of classes are dynamically specified at testing time.
  * we present a novel and general framework, called Elastic Architecture Search (EAS), permitting instant specializations at runtime for diverse tasks with various resource constraints.
  * To this end, we first propose to effectively train the over-parameterized network via a task dropout strategy to disentangle the tasks during training. In this way, the resulting model is robust to the subsequent task dropping at inference time. Based on the well-trained over-parameterized network, we then propose an efficient architecture generator to obtain optimal architectures within a single forward pass.
* [Toward Spatially Unbiased Generative Models](https://arxiv.org/pdf/2108.01285.pdf)  (ICCV'21, Funny)  [code]()
  * Recent image generation models show remarkable generation performance. However, they mirror strong location preference in datasets, which we call spatial bias. Therefore, generators render poor samples at unseen locations and scales.
  * We argue that the generators rely on their implicit positional encoding to render spatial content. From our observations, the generator’s implicit positional encoding is translation-variant, making the generator spatially biased.
  * To address this issue, we propose injecting explicit positional encoding at each scale of the generator. By learning the spatially unbiased generator, we facilitate the robust use of generators in multiple tasks, such as GAN inversion, multi-scale generation, generation of arbitrary sizes and aspect ratios. 



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

