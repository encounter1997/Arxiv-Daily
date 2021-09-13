# Arxiv-Daily

My daily arxiv reading notes.  

[2021 March](202103.md)

[2021 April](202104.md)

[2021 June](202106.md)

[2021 July](202107.md)

[2021 Aug](202108.md)

## CV (Daily)

#### 20210910

* [Leveraging Local Domains for Image-to-Image Translation](https://arxiv.org/pdf/2109.04468.pdf) 
  * In this paper, we leverage human knowledge about spatial domain characteristics which we refer to as ’local domains’ and demonstrate its benefit for image-to-image translation. Relying on a simple geometrical guidance, we train a patch-based GAN on few source data and hallucinate a new unseen domain which subsequently eases transfer learning to target
* :star: [NEAT: Neural Attention Fields for End-to-End Autonomous Driving](https://arxiv.org/pdf/2109.04456.pdf)  (ICCV'21)
  * [code](https://github.com/autonomousvision/neat)
* [ConvMLP: Hierarchical Convolutional MLPs for Vision](https://arxiv.org/pdf/2109.04454.pdf)
  * To tackle these problems, we propose ConvMLP: a hierarchical Convolutional MLP for visual recognition, which is a light-weight, stage-wise, co-design of convolution layers, and MLPs. 
  * In particular, ConvMLPS achieves 76.8% top-1 accuracy on ImageNet-1k with 9M parameters and 2.4G MACs (15% and 19% of MLPMixer-B/16, respectively)
  * Experiments on object detection and semantic segmentation further show that visual representation learned by ConvMLP can be seamlessly transferred and achieve competitive results with fewer parameters
* [TxT: Crossmodal End-to-End Learning with Transformers](https://arxiv.org/pdf/2109.04422.pdf)
* [IICNet: A Generic Framework for Reversible Image Conversion](https://arxiv.org/pdf/2109.04242.pdf)
* [Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers](https://arxiv.org/pdf/2109.04448.pdf)

#### 20210913

* :star: [Is Attention Better Than Matrix Decomposition?](https://arxiv.org/pdf/2109.04553.pdf)  (ICLR'21)
  * Our intriguing finding is that self-attention is not better than the matrix decomposition (MD) model developed 20 years ago regarding the performance and computational cost for encoding the long-distance dependencies.
  * We model the global context issue as a low-rank recovery problem and show that its optimization algorithms can help design global information blocks. 
  * This paper then proposes a series of Hamburgers, in which we employ the optimization algorithms for solving MDs to factorize the input representations into sub-matrices and reconstruct a low-rank embedding. Hamburgers with different MDs can perform favorably against the popular global context module self-attention when carefully coping with gradients back-propagated through MDs
* [TADA: Taxonomy Adaptive Domain Adaptation](https://arxiv.org/pdf/2109.04813.pdf)  (Dengxin Dai, Wenguan Wang, Fisher Yu , Luc Van Gool)
  * Funny
  * We therefore introduce the more general taxonomy adaptive domain adaptation (TADA) problem, allowing for inconsistent taxonomies between the two domains. 
  * We further propose an approach that jointly addresses the imagelevel and label-level domain adaptation. On the label-level, we employ a bilateral mixed sampling strategy to augment the target domain, and a relabelling method to unify and align the label spaces. We address the image-level domain gap by proposing an uncertainty-rectified contrastive learning method, leading to more domain-invariant and class discriminative features.
  * different TADA settings: open taxonomy, coarse-to-fine taxonomy, and partially-overlapping taxonomy
* [EfficientCLIP: Efficient Cross-Modal Pre-training by Ensemble Confident Learning and Language Modeling](https://arxiv.org/pdf/2109.04699.pdf)
  * EfficientCLIP method via Ensemble Confident Learning to obtain a less noisy data subset. Extra rich non-paired single-modal text data is used for boosting the generalization of text branch.
  * We achieve the state-of-theart performance on Chinese cross-modal retrieval tasks with only 1/10 training resources compared to CLIP and WenLan,
* [LAViTeR: Learning Aligned Visual and Textual Representations Assisted by Image and Caption Generation](https://arxiv.org/pdf/2109.04993.pdf)
  * This paper proposes LAViTeR, a novel architecture for visual and textual representation learning. The main module, Visual Textual Alignment (VTA) will be assisted by two auxiliary tasks, GAN-based image synthesis and Image Captioning. We also propose a new evaluation metric measuring the similarity between the learnt visual and textual embedding.
* [LibFewShot: A Comprehensive Library for Few-shot Learning ](https://arxiv.org/pdf/2109.04898.pdf) (Jiebo Luo)
  * 代码库
* [An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA](https://arxiv.org/pdf/2109.05014.pdf)

