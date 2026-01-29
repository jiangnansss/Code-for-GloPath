# GloPath: An Entity-Centric Foundation Model for Glomerular Lesion Assessment and Clinicopathological Insights

This repository provides code, weights, and notebooks related to **GloPath**, a foundation model designed for **entity-centric modeling of glomerular pathology**. GloPath enables extraction of AI-derived glomerular morphological features and supports downstream clinical diagnosis and clinicopathological analyses.

> **Note:** This work is currently under peer review.

---

## Data

- **Private Data:** Most of the training data involve patient-derived pathology slides. Due to **legal, ethical, and privacy constraints**, these datasets are not publicly available. Access to private datasets can be requested via a reasonable research proposal.
- **Public Data:** Public datasets referenced in the paper (e.g., KPMP) are accessible via provided links in the manuscript.

---

## Model Weights

- Due to the use of private data for pretraining, **most model weights cannot be publicly shared at this stage**.
- We have uploaded a preliminary pretraining version of GloPath:  
  `Code-for-GloPath/0-Models_and_weights/0-GloPath/glopath_weight_0.pth`
- Additional weights can be **shared upon request** after publication or for collaborative research.

---

## Code Overview

The repository provides core code to facilitate model usage and inference:

1. **Model Code**
   - GloPath and Segmentor architecture and initialization: `Code-for-GloPath/0-Models_and_weights/`
2. **Pretraining Code**
   - Self-supervised pretraining code: `Code-for-GloPath/1-Codes/0-pretrain/`
   - Note: Parameters differ slightly from the final model in the paper.
3. **Fine-tuning / Downstream Training**
   - Example classification training code: `Code-for-GloPath/1-Codes/2-classification_train_simple.py`
4. **Inference / Testing**
   - Segmentation and morphological quantification of glomerular Bowman's capsule and tuft:  
     `Code-for-GloPath/1-Codes/3-segmentation_and_quantification.ipynb`
   - Demonstrates feature extraction and basic clinicopathological analysis.

---

## How to Use

### 1. Pretraining
- **Data:** Prepare large-scale glomerular image dataset (as in manuscript, ~1M images).
- **Command:** 
```bash
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
  --arch vit_small \
  --data_path /path/to/dataset \
  --output_dir /path/to/output # single node

python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
  --arch vit_small \
  --data_path /path/to/dataset \
  --output_dir /path/to/output # multiple nodes
```

### 2. Fine-tuning
- **Data:** For example, prepare PASM-stained glomeruli with lesion labels.
- **Command:** 
```bash
python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small \
  --data_path /path/to/dataset --output_dir /path/to/output
```
- Outputs fine-tuned weights for downstream tasks.

### 3. Inference / Application (Donwstream tasks: Lesion recogniiton, lesion grading, cross-modality diagnosis, few-shot based diagnosis, and clinicopathological correlation analysis.)
- **Data:** For example, prepare test images or pre-extracted glomeruli.
- **Notebook:** Run
```bash
Code-for-GloPath/1-Codes/3-segmentation_and_quantification.ipynb
```
to perform segmentation and extract quantitative features.

### 4. Notes
- Only core code, sample weights, and notebooks are provided due to data privacy restrictions.
- Users can reproduce inference workflows on their own datasets using the provided notebooks.
- We plan to release additional weights and more comprehensive datasets after publication.

### 5. Citation
If you use this code or GloPath model, please cite the corresponding manuscript once published.



