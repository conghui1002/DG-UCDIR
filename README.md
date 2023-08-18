# Unsupervised Feature Representation Learning for Domain-generalized Cross-domain Image Retrieval

This repository contains the PyTorch implementation for our ICCV2023 paper "Unsupervised Feature Representation Learning for Domain-generalized Cross-domain Image Retrieval".

Cross-domain image retrieval has been extensively stud- ied due to its high practical value. In recently proposed unsupervised cross-domain image retrieval methods, efforts are taken to break the data annotation barrier. However, applicability of the model is still confined to domains seen during training. This limitation motivates us to present the first attempt at domain-generalized unsupervised cross- domain image retrieval (DG-UCDIR) aiming at facilitat- ing image retrieval between any two unseen domains in an unsupervised way. To improve domain generalizability of the model, we thus propose a new two-stage domain aug- mentation technique for diversified training data genera- tion. DG-UCDIR also shares all the challenges present in the unsupervised cross-domain image retrieval, where domain-agnostic and semantic-aware feature representa- tions are supposed to be learned without external super- vision. To accomplish this, we introduce a novel cross- domain contrastive learning strategy by utilizing phase im- age as a proxy to mitigate the domain gap. Extensive exper- iments are carried out using PACS and DomainNet dataset, and consistently illustrate the superior performance of our framework compared to existing state-of-the-art methods.
