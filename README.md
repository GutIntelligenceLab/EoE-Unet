# EoE-Unet

## Introduction
Implementation of "[Advancing Eosinophilic Esophagitis Diagnosis and Phenotype Assessment with Deep Learning Computer Vision](https://arxiv.org/abs/2101.05326)". This work was awarded best paper award in [BIOIMAGING 2021](https://bioimaging.scitevents.org/PreviousAwards.aspx).

## Abstract
Eosinophilic Esophagitis (EoE) is an inflammatory esophageal disease which is increasing in prevalence. The diagnostic gold-standard involves manual review of a patient's biopsy tissue sample by a clinical pathologist for the presence of 15 or greater eosinophils within a single high-power field (400x magnification). Diagnosing EoE can be a cumbersome process with added difficulty for assessing the severity and progression of disease. We propose an automated approach for quantifying eosinophils using deep image segmentation. A U-Net model and post-processing system are applied to generate eosinophil-based statistics that can diagnose EoE as well as describe disease severity and progression. These statistics are captured in biopsies at the initial EoE diagnosis and are then compared with patient metadata: clinical and treatment phenotypes. The goal is to find linkages that could potentially guide treatment plans for new patients at their initial disease diagnosis. A deep image classification model is further applied to discover features other than eosinophils that can be used to diagnose EoE. This is the first study to utilize a deep learning computer vision approach for EoE diagnosis and to provide an automated process for tracking disease severity and progression.

## How to Run
Use the main notebook for experimenting and orchestrating the training.

## Reference
If you find our work useful Please consider citing our paper:

```bash
@article{william2021advancing,
  title={Advancing Eosinophilic Esophagitis Diagnosis and Phenotype Assessment with Deep Learning Computer Vision},
  author={William Adorno, III and Catalano, Alexis and Ehsan, Lubaina and von Eckstaedt, Hans Vitzhum and Barnes, Barrett and McGowan, Emily and Syed, Sana and Brown, Donald E},
  year={2021}
}
```
