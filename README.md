# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

[2021 April](202104.md)

[2021 June](202106.md)

[2021 July](202107.md)

## CV (Daily)

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

