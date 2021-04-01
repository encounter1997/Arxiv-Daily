# Arxiv-Daily
My daily arxiv reading notes.  

[2021 March](202103.md)


## CV (Daily)
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
* 


Vision Transformer:
* [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf)  However the optimization of image transformers has been little studied so far. In this work, we build and optimize deeper transformer networks for image classification.  This leads us to produce models whose performance does not saturate early with more depth, for instance we obtain 86.3% top-1 accuracy on Imagenet when training with no external data  (Facebook, DeiT团队)  
* [Learning Spatio-Temporal Transformer for Visual Tracking](https://arxiv.org/pdf/2103.17154.pdf)   The encoder models the global spatio-temporal feature dependencies between target objects and search regions, while the decoder learns a query embedding to predict the spatial positions of the target objects.  The whole method is endto-end, does not need any postprocessing steps such as cosine window and bounding box smoothing, thus largely simplifying existing tracking pipelines. [code](https://github.com/researchmm/Stark)  (Jianlong Fu, Huchuan Lu) 
* [Robust Facial Expression Recognition with Convolutional Visual Transformers](https://arxiv.org/pdf/2103.16854.pdf)  Different from previous pure CNNs based methods, we argue that it is feasible and practical to translate facial images into sequences of visual words and perform expression recognition from a global perspective.  (Shutao Li)
* [DA-DETR: Domain Adaptive Detection Transformer by Hybrid Attention](https://arxiv.org/pdf/2103.17084.pdf) 基于Deformable DETR的域适应目标检测



## NLP (Weekly)
