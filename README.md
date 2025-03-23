# densenet-for-smartwatch-ppg
Darren, last edit 03/23/2025.

This repository contains the pretrained DenseNet models, source code, and data loading documentation for two versions (please visit the corresponding branches to see each version):
- [BSN2024](https://github.com/chen-darren/densenet-for-smartwatch-ppg/tree/BSN2024)
  - The paper titled "Smartwatch Photoplethysmogram-Based Atrial Fibrillation Detection with Premature Atrial and Ventricular Contraction Differentiation Using Densely Connected Convolutional Neural Networks" by Darren Chen, Dong Han, Luis R. Mercado-Díaz, Jihye Moon, and Ki H. Chon, presented at the 2024 IEEE 20th International Conference on Body Sensor Networks (BSN).
    - For reference, see:
      - D. Chen, D. Han, L. R. Mercado-Díaz, J. Moon, and K. H. Chon, “Smartwatch Photoplethysmogram-Based Atrial Fibrillation Detection with Premature Atrial and Ventricular Contraction Differentiation Using Densely Connected Convolutional Neural Networks,” in 2024 IEEE 20th International Conference on Body Sensor Networks (BSN), Oct. 2024, pp. 1–4. doi: 10.1109/BSN63547.2024.10780734. 
- [UConnThesis](https://github.com/chen-darren/densenet-for-smartwatch-ppg/tree/UConnThesis)
  - For Darren Chen's Honors Undergraduate Thesis, titled "Application of Deep Learning and Data Balancing Methods for Multiclass Cardiac Rhythm Detection and Classification Using Real-World Smartwatch Photoplethysmography" at the University of Connecticut.

Keywords:

PPG: photoplethysmography
AF: atrial fibrilaltion
PAC/PVC or PAC_PVC: premature atrial and ventricular contractions
NSR: normal sinus rhythm
The study focuses on the Pulsewatch dataset to develop and evaluate deep learning models for detecting and classifying various cardiac rhythms using real-world smartwatch PPG data. This work strongly builds upon the DenseNet architecture and extends its application to cardiac rhythm classification using real-world smartwatch PPG data.

DenseNet Repository: DenseNet by Zhuang Liu
For reference, see:
G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely Connected Convolutional Networks,” Jan. 28, 2018, arXiv: arXiv:1608.06993. doi: 10.48550/arXiv.1608.06993.
This project also builds upon and extends previous research on Pulsewatch. For more information, please refer to the Pulsewatch repository maintained by my postdoc mentor: PulsewatchRelease.
