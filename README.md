# Multi-Task Learning for Image-Based Scene Understanding in Autonomous Driving

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-Cityscapes-blue.svg)](https://www.cityscapes-dataset.com/)
[![mIoU Benchmark](https://img.shields.io/badge/Segmentation%20mIoU-80.00%25-brightgreen.svg)]()
[![Depth AbsRel](https://img.shields.io/badge/Depth%20AbsRel-0.0944-success.svg)]()

A high-performance Computer Vision system for autonomous driving scene understanding. The system takes a **single RGB image** as input and simultaneously performs:
1. **Semantic Segmentation** (categorical dense classification: *What is present at every pixel?*)
2. **Monocular Depth Estimation** (continuous geometric regression: *How far is every pixel?*)

---

## Table of Contents
1. [Project Overview & Core Problem](#1-project-overview--core-problem)
2. [Architecture & Engineering Design Choices](#2-architecture--engineering-design-choices)
3. [Dataset Pipeline & Preprocessing](#3-dataset-pipeline--preprocessing)
4. [Training Methodology & Optimization Strategy](#4-training-methodology--optimization-strategy)
5. [Comprehensive Metric Formulation & Significance](#5-comprehensive-metric-formulation--significance)
6. [Experimental Results & Per-Class Breakdown](#6-experimental-results--per-class-breakdown)
7. [Qualitative Visualizations](#7-qualitative-visualizations)
8. [Multi-Task Learning (MTL) Research Strategy](#8-multi-task-learning-mtl-research-strategy)
9. [Repository Structure](#9-repository-structure)
10. [Usage Guide](#10-usage-guide)

---

## 1. Project Overview & Core Problem

### The Problem in Autonomous Driving
An autonomous vehicle requires real-time, comprehensive spatial and semantic understanding of its environment. Traditional architectures deploy independent models for segmentation and depth estimation:

```text
               RGB Image
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
Segmentation Model      Depth Model
   (Encoder+Dec)       (Encoder+Dec)
         │                   │
         ▼                   ▼
 Segmentation Mask       Depth Map
```

* **Computational Redundancy:** Both models independently extract low-level edges, textures, and high-level object representations, doubling GPU memory and latency.
* **Lack of Synergistic Cues:** Single-task models cannot share complementary features (e.g., 3D depth geometry assisting object boundary separation; semantic classes enforcing planar depth consistency).

### Our Proposed Solution
A **unified Multi-Task Learning framework** powered by a **Shared Hierarchical Vision Transformer (SegFormer MiT) Encoder** with **two task-specialized decoders**:

```text
                               Input: Single RGB Image
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │ Shared SegFormer MiT Encoder │
                           └──────────────┬───────────────┘
                                          │
                             Multi-Scale Features (f1–f4)
                                   /              \
                                  /                \
                                 ▼                  ▼
                        ┌─────────────────┐   ┌───────────────────────────┐
                        │ SegFormer       │   │ Progressive Fusion        │
                        │ All-MLP Decoder │   │ Depth Decoder             │
                        └────────┬────────┘   └─────────────┬─────────────┘
                                 │                          │
                                 ▼                          ▼
                        Predicted Segmentation       Predicted Normalized
                                 Mask                    Inverse Depth
```

---

## 2. Architecture & Engineering Design Choices

Each component was selected based on rigorous computer vision principles to maximize dense prediction accuracy and hardware efficiency.

### 2.1 Shared Vision Transformer Encoder: SegFormer MiT-B2
* **File:** [`models/segformer_encoder.py`](models/segformer_encoder.py)
* **Design Rationale:**
  1. **Hierarchical Multi-Scale Pyramids:** Produces 4 distinct resolution feature maps:
     * $f_1 \in \mathbb{R}^{B \times 64 \times \frac{H}{4} \times \frac{W}{4}}$ (High-resolution spatial details, crisp edges)
     * $f_2 \in \mathbb{R}^{B \times 128 \times \frac{H}{8} \times \frac{W}{8}}$ (Object parts and boundary transitions)
     * $f_3 \in \mathbb{R}^{B \times 320 \times \frac{H}{16} \times \frac{W}{16}}$ (Regional object semantics)
     * $f_4 \in \mathbb{R}^{B \times 512 \times \frac{H}{32} \times \frac{W}{32}}$ (High-level global context and scene semantics)
  2. **Overlapped Patch Merging:** Preserves local continuity around patch boundaries without hard block artifacts.
  3. **Positional-Encoding-Free Design:** Uses Mix-FFN ($3\times 3$ depth-wise convs) to encode positional information dynamically, allowing seamless execution at arbitrary input resolutions without positional embedding interpolation degradation.

### 2.2 Semantic Segmentation Decoder: SegFormer All-MLP Decoder
* **File:** [`models/segformer_decoder.py`](models/segformer_decoder.py)
* **Design Rationale:**
  * Avoids heavy, memory-intensive decoders.
  * Linearly projects each stage $f_i$ to an embedding dimension $C_{\text{embed}} = 256$.
  * Bilinearly upsamples all stages to $\frac{H}{4} \times \frac{W}{4}$ and concatenates them ($1024$ channels).
  * Fuses concatenated features with a $1\times 1$ Conv $\to$ BatchNorm $\to$ ReLU.
  * Classifies into 19 classes followed by a single $4\times$ bilinear upsample directly to full image resolution $(H, W)$, eliminating the severe spatial blur associated with direct $32\times$ upsampling.

### 2.3 Monocular Depth Decoder: Progressive Fusion Depth Decoder
* **File:** [`models/decoder/progressive_depth_decoder.py`](models/decoder/progressive_depth_decoder.py)
* **Design Rationale:**
  * Monocular depth estimation requires gradual, bottom-up geometric reconstruction.
  * The decoder starts from the deepest feature $f_4$ ($1/32$), progressively upsamples by $2\times$, and element-wise adds skip-projected features from $f_3, f_2,$ and $f_1$ with residual refinement convolutions at each scale.
  * Outputs continuous normalized inverse depth activated via Sigmoid: $\hat{d} \in [0, 1]$.

---

## 3. Dataset Pipeline & Preprocessing

### 3.1 Cityscapes Dataset Components
* **File:** [`datasets/cityscapes_dataset.py`](datasets/cityscapes_dataset.py)
* We utilize the official Cityscapes dataset:
  1. `leftImg8bit/`: 8-bit RGB road scene images (the **only input** given to the encoder).
  2. `gtFine/`: Fine pixel annotations for 19 evaluation classes (supervision ground truth for segmentation).
  3. `disparity/`: 16-bit stereo disparity maps (supervision ground truth for depth).

### 3.2 Strict Data Leakage Barrier
```text
                   ┌──────────┐
                   │ RGB Only │
                   └────┬─────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Model (Encoder) │
               └────────┬────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
      Predicted Seg           Predicted Depth
            │                       │
            ▼ (Loss Only)           ▼ (Loss Only)
     gtFine Target           Disparity Target
```
* The model **never receives** ground truth masks or disparity during training or inference.
* Segmentation and disparity are strictly supervision targets for computing loss.

### 3.3 Metric Disparity to Depth Conversion
Cityscapes stores disparity $D$ as 16-bit integers with a baseline $B = 0.209313\text{ m}$ and focal length $f = 2262.52\text{ px}$.
1. Real disparity: $d = \frac{D - 1}{256.0}$ (for $D > 0$)
2. Metric depth: $Z = \frac{f \cdot B}{d}$ (in meters)
3. Normalized Inverse Depth Target:
   $$d_{\text{norm}} = \frac{1/Z - 1/Z_{\max}}{1/Z_{\min} - 1/Z_{\max}}, \quad Z_{\min}=1.0\text{ m}, Z_{\max}=80.0\text{ m}$$
   * Closer objects $\to$ Higher value ($\to 1$)
   * Farther objects $\to$ Lower value ($\to 0$)
   * Invalid pixels $\to 0$ (masked out during loss computation)

### 3.4 Native Resolution ($1024 \times 2048$) vs Squashed ($1024 \times 1024$)
* Native Cityscapes images have an exact $1:2$ aspect ratio.
* Squashing to $1024 \times 1024$ distorts vertical structures by $50\%$.
* Training and evaluating at native $1024 \times 2048$ unlocked significant gains on thin/vertical classes:
  * **Poles:** $57.05\% \to \mathbf{66.71\%}$ ($+9.66\%$)
  * **Traffic Lights:** $66.05\% \to \mathbf{72.02\%}$ ($+5.97\%$)
  * **Traffic Signs:** $75.72\% \to \mathbf{80.25\%}$ ($+4.53\%$)
  * **Persons:** $78.49\% \to \mathbf{83.10\%}$ ($+4.61\%$)

---

## 4. Training Methodology & Optimization Strategy

### 4.1 Differential Learning Rate Optimization
* **Encoder:** Pretrained on large-scale data; fine-tuned gently at $\text{LR}_{\text{enc}} = 6 \times 10^{-6}$.
* **Decoder:** Initialized from scratch; trained with higher capacity at $\text{LR}_{\text{dec}} = 6 \times 10^{-5}$.
* **Scheduler:** Polynomial Learning Rate Decay with power $\gamma = 0.9$ and $\text{LR}_{\min} = 1 \times 10^{-7}$:
  $$\text{LR}(t) = (\text{LR}_{\text{base}} - \text{LR}_{\min}) \cdot \left(1 - \frac{t}{T_{\text{total}}}\right)^{0.9} + \text{LR}_{\min}$$

### 4.2 Mixed Precision & Gradient Accumulation
* Trained using PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`) and `GradScaler()`.
* Batch Configuration: $\text{Batch Size} = 1$ with $\text{Accumulation Steps} = 8 \implies \textbf{Effective Batch Size} = 8$.
* Peak VRAM consumption stays under **$8\text{ GB}$**, enabling full native $1024 \times 2048$ training on consumer and workstation GPUs (e.g., NVIDIA RTX 4500 Ada 24 GB).

### 4.3 Compound Loss Formulation
* **Segmentation Loss:**
  $$\mathcal{L}_{\text{seg}} = 0.7 \cdot \mathcal{L}_{\text{CE}} + 0.3 \cdot \mathcal{L}_{\text{Dice}}$$
  * $\mathcal{L}_{\text{CE}}$: Standard Cross-Entropy with `ignore_index=255` and $\text{label\_smoothing}=0.0$.
  * $\mathcal{L}_{\text{Dice}}$: Multi-class soft Dice loss ensuring sharp boundary alignment.
* **Depth Loss (BerHu - Reverse Huber):**
  $$\mathcal{L}_{\text{BerHu}}(y, \hat{y}) = \begin{cases} |y - \hat{y}| & \text{if } |y - \hat{y}| \le c \\ \frac{(y - \hat{y})^2 + c^2}{2c} & \text{if } |y - \hat{y}| > c \end{cases}, \quad c = 0.2 \cdot \max(|y - \hat{y}|)$$
  * Acts as $L_1$ penalty for small errors (preserving sharp depth discontinuities) and $L_2$ for large structural errors.

---

## 5. Comprehensive Metric Formulation & Significance

### 5.1 Semantic Segmentation Metrics

#### Mean Intersection over Union (mIoU)
For each class $c \in \{0, 1, \dots, 18\}$:
$$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c} = \frac{\sum_{i=1}^N |P_{i, c} \cap G_{i, c}|}{\sum_{i=1}^N |P_{i, c} \cup G_{i, c}|}$$
$$\text{mIoU} = \frac{1}{19} \sum_{c=0}^{18} \text{IoU}_c$$
* **Significance:** Standard pixel accuracy is misleading in autonomous driving because backgrounds (road, sky, buildings) cover $>80\%$ of pixels. mIoU gives equal weight to all classes, penalizing models that ignore safety-critical minorities (pedestrians, riders, traffic lights).

#### Global Accumulation vs Per-Image Averaging
* **Official Benchmark Standard:** Accumulates True Positives, False Positives, and False Negatives across the **entire 500-image dataset** before computing IoU.
* **Naive Per-Image Averaging:** Computes IoU per image and averages across images. If a rare class (e.g., train) has 0 true pixels in an image but receives 5 false positive pixels, per-image IoU records $0.0\%$, artificially deflating scores. Standard global accumulation avoids this artifact and is used in all academic benchmarks.

### 5.2 Depth Estimation Metrics
Computed over valid ground truth pixels ($\mathcal{M} = \{i : y_i > 0\}$):
1. **Absolute Relative Error (AbsRel - Primary Metric, Lower is Better):**
   $$\text{AbsRel} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \frac{|y_i - \hat{y}_i|}{y_i}$$
2. **Root Mean Squared Error (RMSE, Lower is Better):**
   $$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} (y_i - \hat{y}_i)^2}$$
3. **Threshold Accuracy ($\delta < 1.25$, Higher is Better):**
   $$\% \text{ of pixels where } \max\left(\frac{y_i}{\hat{y}_i}, \frac{\hat{y}_i}{y_i}\right) < 1.25$$

---

## 6. Experimental Results & Per-Class Breakdown

### 6.1 Single-Task Semantic Segmentation Baseline
* **Validation Set:** Cityscapes 500 Val Images ($1024 \times 2048$)
* **Standard Single-Pass mIoU:** **$79.69\%$**
* **With Test-Time Augmentation (TTA):** **$\mathbf{80.00\%}$**

| Class ID | Class Name | Standard IoU | TTA (+ Flip) IoU | Qualitative Evaluation |
| :---: | :--- | :---: | :---: | :--- |
| 0 | **Road** | **$98.40\%$** | **$98.37\%$** | Flawless road plane segmentation |
| 1 | **Sidewalk** | **$86.71\%$** | **$86.53\%$** | Sharp curb and walkway distinction |
| 2 | **Building** | **$93.04\%$** | **$93.22\%$** | Accurate vertical architectural facade |
| 3 | **Wall** | **$62.05\%$** | **$64.23\%$** | Clean texture separation from buildings |
| 4 | **Fence** | **$61.07\%$** | **$61.60\%$** | High-fidelity barrier detection |
| 5 | **Pole** | **$66.25\%$** | **$66.71\%$** | Superior thin vertical structure recovery |
| 6 | **Traffic Light** | **$71.57\%$** | **$72.02\%$** | Accurate small-object localization |
| 7 | **Traffic Sign** | **$79.99\%$** | **$80.25\%$** | Sharp geometric symbol recognition |
| 8 | **Vegetation** | **$92.94\%$** | **$93.00\%$** | Excellent tree and foliage coverage |
| 9 | **Terrain** | **$64.42\%$** | **$64.75\%$** | Reliable grass/ground surface classification |
| 10 | **Sky** | **$95.25\%$** | **$95.29\%$** | Perfect horizon and background capture |
| 11 | **Person** | **$82.94\%$** | **$83.10\%$** | Precise pedestrian silhouette detection |
| 12 | **Rider** | **$62.85\%$** | **$62.97\%$** | Distinct from bikes and motorcycles |
| 13 | **Car** | **$95.21\%$** | **$95.25\%$** | Near-perfect multi-vehicle boundaries |
| 14 | **Truck** | **$84.15\%$** | **$84.47\%$** | Accurate heavy vehicle classification |
| 15 | **Bus** | **$88.17\%$** | **$88.35\%$** | Clean public transport detection |
| 16 | **Train** | **$81.77\%$** | **$81.75\%$** | High-precision rail transit detection |
| 17 | **Motorcycle** | **$69.15\%$** | **$69.70\%$** | Robust two-wheeler identification |
| 18 | **Bicycle** | **$78.12\%$** | **$78.35\%$** | Crisp frame and wheel spoke delineation |
| **All** | **Mean IoU (mIoU)** | **$79.69\%$** | **$\mathbf{80.00\%}$** | **Competitive with published SOTA** |

### 6.2 Single-Task Depth Baseline
* **AbsRel:** **$0.0944$** *(Target $< 0.12$)*
* **RMSE:** **$0.0646$**
* **MAE:** **$0.0319$**
* **$\delta < 1.25$ Accuracy:** **$90.54\%$** *(Target $> 85.0\%$)*
* **$\delta < 1.25^2$ Accuracy:** **$97.78\%$**
* **$\delta < 1.25^3$ Accuracy:** **$99.18\%$**

---

## 7. Qualitative Visualizations

Presentation figures generated at 200 DPI:

1. **Single-Scene 3-Panel Evaluation:**
   * Path: [`logs/segmentation/eval_sample.png`](logs/segmentation/eval_sample.png)
   * Displays `Input RGB Image (1024×2048)` $\to$ `Predicted Mask (Ours: 80.00% mIoU)` $\to$ `Ground Truth (gtFine)`.
   * Highlights sharp boundary delineation on small objects, pedestrian silhouettes, and occlusion handling.

2. **Multi-Scene Generalization Grid (3x3):**
   * Path: [`logs/segmentation/eval_sample_multi.png`](logs/segmentation/eval_sample_multi.png)
   * Evaluates the model across 3 distinct driving scenarios:
     1. High-density urban street with complex vehicle clutter.
     2. Pedestrian crosswalk with sidewalk barriers.
     3. Multi-lane intersection with complex building facades.

---

## 8. Multi-Task Learning (MTL) Research Strategy

### 8.1 Scientific Hypothesis
> *"A modern hierarchical Vision Transformer shared encoder can learn unified representations supporting both semantic segmentation and monocular depth estimation, maintaining accuracy parity with separate single-task models while reducing encoder parameter count and latency by nearly 50%."*

### 8.2 Loss Balancing & Task Interference
Joint optimization minimizes a combined objective:
$$\mathcal{L}_{\text{total}} = w_{\text{seg}} \mathcal{L}_{\text{seg}} + w_{\text{depth}} \mathcal{L}_{\text{depth}}$$

To prevent one task from dominating gradients, we investigate two paradigms:
1. **Static Loss Weighting:** Experimentally tuned fixed coefficients ($w_{\text{seg}}=1.0, w_{\text{depth}}=1.5$).
2. **Dynamic Uncertainty Weighting (Kendall et al., CVPR 2018):**
   Learning homoscedastic task uncertainties $s = \log(\sigma^2)$:
   $$\mathcal{L}_{\text{total}} = \frac{1}{2}\exp(-s_{\text{seg}})\mathcal{L}_{\text{seg}} + \exp(-s_{\text{depth}})\mathcal{L}_{\text{depth}} + \frac{1}{2}s_{\text{seg}} + \frac{1}{2}s_{\text{depth}}$$

### 8.3 Expected Scientific Benchmarking Table

| Model Architecture | Encoder Params | Total Params | Segmentation mIoU | Depth AbsRel | Inference FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Seg-Only** | $24.2\text{ M}$ | $27.5\text{ M}$ | **$79.69\%$** | — | $1.0\times$ |
| **Baseline 2: Depth-Only** | $24.2\text{ M}$ | $27.5\text{ M}$ | — | **$0.0944$** | $1.0\times$ |
| **Combined Independent** | $48.4\text{ M}$ *(Duplicated)* | $55.0\text{ M}$ | $79.69\%$ | $0.0944$ | $0.5\times$ |
| **Joint MTL (Static Weighting)** | **$24.2\text{ M}$** | **$30.8\text{ M}$** | $\approx 78.5\text{–}79.5\%$ | $\approx 0.095$ | $\approx \mathbf{1.8\times}$ |
| **Joint MTL (Kendall Uncertainty)** | **$24.2\text{ M}$** | **$30.8\text{ M}$** | $\mathbf{\approx 79.5\text{–}80.5\%}$ | $\mathbf{\approx 0.090}$ | $\approx \mathbf{1.8\times}$ |

---

## 9. Repository Structure

```text
Capstone/
├── checkpoints/
│   ├── segmentation/
│   │   ├── best_seg_model.pth          # Best segmentation model weights (80.00% mIoU)
│   │   └── last_seg_checkpoint.pth     # Full training state for resuming
│   └── depth/
│       └── best_depth_model.pth        # Best depth model weights
│
├── datasets/
│   └── cityscapes_dataset.py           # Native 1024x2048 dataset loader & augmentations
│
├── models/
│   ├── decoder/
│   │   └── progressive_depth_decoder.py # Progressive fusion depth head
│   ├── segformer_encoder.py            # SegFormer MiT-B2 hierarchical Transformer
│   ├── segformer_decoder.py            # SegFormer All-MLP segmentation head
│   ├── seg_model.py                    # Single-task segmentation wrapper
│   └── depth_model.py                  # Single-task depth wrapper
│
├── utils/
│   ├── depth_utils.py                  # Disparity <-> Metric depth conversion
│   ├── label_mapping.py                # 34 raw IDs -> 19 evaluation TrainIDs
│   ├── metrics.py                      # Academic global accumulator for mIoU & depth
│   └── visualize.py                    # Color mapping and rendering utilities
│
├── scripts/
│   ├── check_dataset.py                # Sanity checking data loader & shapes
│   ├── train_seg.py                    # Native 1024x2048 segmentation training
│   ├── evaluate_seg.py                 # Official 19-class benchmark evaluator
│   ├── train_depth.py                  # Native 1024x2048 depth training
│   └── evaluate_depth.py               # Depth evaluation (AbsRel, RMSE, delta)
│
├── logs/
│   ├── segmentation/
│   │   ├── eval_report.txt             # Saved official benchmark report
│   │   ├── eval_sample.png             # Presentation 3-panel figure
│   │   ├── eval_sample_multi.png       # Presentation 3x3 multi-scene figure
│   │   └── train_log.csv               # Epoch-by-epoch loss and mIoU log
│   └── depth/
│       ├── eval_report.txt             # Saved depth benchmark report
│       └── eval_sample.png             # Depth visual comparison
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Complete project documentation
```

---

## 10. Usage Guide

### 10.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/teju701/Capstone.git
cd Capstone

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 10.2 Evaluating Segmentation Baseline
To run full evaluation on the 500 validation images and regenerate reports and visuals:
```bash
python scripts/evaluate_seg.py
```

### 10.3 Training Segmentation Baseline
To train the baseline from scratch or resume from a checkpoint:
```bash
python scripts/train_seg.py
```

### 10.4 Evaluating Depth Baseline
```bash
python scripts/evaluate_depth.py
```

### 10.5 Training Depth Baseline
```bash
python scripts/train_depth.py
```

---

## Authors & Acknowledgments
* **Engineering Capstone Team**
* **Project:** Multi-Task Learning for Dense Scene Understanding in Autonomous Driving
* **Dataset:** Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
* **Backbone:** Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
* **Loss Balancing:** Kendall et al., *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, CVPR 2018.
