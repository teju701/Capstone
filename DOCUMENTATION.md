# Multi-Task Autonomous Driving Perception System
## Comprehensive Technical Documentation & Architecture Reference

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Autonomous Driving Challenge
In autonomous driving (e.g., Tesla Vision, Waymo, Cruise, Mobileye), vehicles must perceive both **semantic context** (identifying what each object is: road, car, pedestrian, traffic sign) and **geometric distance** (identifying where objects are in 3D physical space).

Traditionally, perception systems run two separate deep neural networks:
1. A **Semantic Segmentation Network** (e.g., SegFormer / DeepLabV3+)
2. A **Monocular Depth Estimation Network** (e.g., DPT / MiDaS)

### 1.2 The Bottleneck of Separate Models
Running two separate networks creates major compute and engineering drawbacks:
- **Redundant Computation**: Both networks spend $>70\%$ of their GPU FLOPs extracting the exact same low-level visual features (edges, textures, spatial gradients, horizon lines).
- **Memory & Latency Overhead**: Two heavy backbones require **$53.2\text{M}$ parameters** and double the inference time, causing thermal throttling on edge automotive hardware.
- **Asynchronous Latency**: Running two models asynchronously risks temporal misalignment between segmentation masks and depth predictions.

### 1.3 The Multi-Task Learning (MTL) Solution
Our architecture unifies both perception tasks into a **single, shared Vision Transformer (SegFormer MiT-B2)** backbone with task-specialized lightweight decoders:
- **Shared Parameters**: One hierarchical encoder processes the image once.
- **Parameter Savings**: **$-34.6\%$ parameter reduction** ($34.8\text{M}$ vs $53.2\text{M}$).
- **Inference Speed**: **$2\times$ throughput boost** ($9.7\text{ ms}$ on GPU / $\approx 700\text{ ms}$ on CPU).
- **Performance Retention**: **$79.81\%$ mIoU** on Cityscapes ($99.8\%$ baseline parity) and **$0.1129$ AbsRel** depth error.

---

## 2. Multi-Task Learning Architecture & Design Choices

```
                        Input Image: (B, 3, 1024, 2048)
                                       │
                                       ▼
             ┌──────────────────────────────────────────────────┐
             │    SegFormer MiT-B2 Shared Vision Transformer    │
             │       (Overlapped Patch Embeddings + SRA)        │
             └──────────────────────────────────────────────────┘
                 │            │             │            │
             Stage 1      Stage 2       Stage 3      Stage 4
             (H/4, W/4)   (H/8, W/8)    (H/16, W/16) (H/32, W/32)
              C1 = 64      C2 = 128      C3 = 320     C4 = 512
                 │            │             │            │
         ┌───────┴────────────┴─────────────┴────────────┴────────┐
         │                                                        │
         ▼                                                        ▼
┌───────────────────────────────┐     ┌─────────────────────────────────────────┐
│ Task 1: All-MLP Segmentation  │     │   Task 2: Progressive Fusion Depth      │
│            Decoder            │     │                Decoder                  │
│                               │     │                                         │
│ • Linear MLP Projections(256) │     │ • Bottom-up progressive fusion (4→1)    │
│ • Bilinear Upsample to H/4    │     │ • Residual skip-connections & conv fuse │
│ • Concat & Linear Fuse (256)  │     │ • 4× final bilinear upsample            │
│ • 1×1 Conv Classifier (19 cls)│     │ • 1×1 Conv + Sigmoid Output             │
└───────────────────────────────┘     └─────────────────────────────────────────┘
                 │                                         │
                 ▼                                         ▼
   19-Class Semantic Logits:                    Calibrated Metric Depth Map:
     (B, 19, 1024, 2048)                            (B, 1, 1024, 2048)
```

### 2.1 Shared Encoder: SegFormer MiT-B2 (Hierarchical Vision Transformer)
Unlike standard ViT (which outputs single-scale flat patch tokens with quadratic complexity), **SegFormer MiT-B2** provides four critical architectural innovations:

1. **Hierarchical Multi-Scale Features**:
   - Generates feature representations at 4 spatial resolutions:
     - **Stage 1**: $\frac{H}{4} \times \frac{W}{4}, C_1 = 64$ (High-frequency details: edges, poles, traffic lights).
     - **Stage 2**: $\frac{H}{8} \times \frac{W}{8}, C_2 = 128$ (Mid-level object boundaries: pedestrians, cyclists).
     - **Stage 3**: $\frac{H}{16} \times \frac{W}{16}, C_3 = 320$ (High-level semantic regions: cars, trucks, buses).
     - **Stage 4**: $\frac{H}{32} \times \frac{W}{32}, C_4 = 512$ (Global scene context: roads, sky, buildings).

2. **Overlapped Patch Merging**:
   - Standard ViT splits images into non-overlapping $16 \times 16$ patches, losing local pixel continuity across patch boundaries.
   - SegFormer uses $K=7, S=4, P=3$ overlapped convolution patch embedding, preserving continuous spatial structure essential for depth and boundary segmentation.

3. **Efficient Spatial Reduction Attention (SRA)**:
   - Reduces self-attention sequence length from $N$ to $N/R^2$ using reduction ratios $R \in [8, 4, 2, 1]$, enabling real-time native HD ($1024 \times 2048$) processing with minimal memory footprint.

4. **Mix-FFN with Positional-Encoding-Free Design**:
   - Replaces fixed positional embeddings with a $3 \times 3$ Depthwise Convolution inside the feed-forward network. This allows arbitrary input resolutions (e.g. $512 \times 1024$ fast mode vs $1024 \times 2048$ native HD) without interpolation distortion.

---

### 2.2 Task 1 Decoder: All-MLP Semantic Segmentation Head
- **Mechanism**:
  1. Each multi-scale stage $f_i$ is projected to a uniform channel dimension $C = 256$ via a lightweight MLP layer:
     $$M_i = \text{Linear}(C_i, 256)(f_i), \quad \forall i \in \{1, 2, 3, 4\}$$
  2. Features $M_2, M_3, M_4$ are bilinearly upsampled to $\frac{H}{4} \times \frac{W}{4}$ and concatenated along the channel axis:
     $$M = [M_1, \text{Upsample}(M_2), \text{Upsample}(M_3), \text{Upsample}(M_4)] \in \mathbb{R}^{B \times 1024 \times \frac{H}{4} \times \frac{W}{4}}$$
  3. A fusion MLP blends the multi-scale features: $F = \text{Linear}(1024, 256)(M)$.
  4. A $1 \times 1$ convolution outputs class logits for the 19 Cityscapes evaluation classes: $\hat{Y}_{seg} \in \mathbb{R}^{B \times 19 \times H \times W}$.

---

### 2.3 Task 2 Decoder: Progressive Fusion Depth Decoder
Monocular depth estimation requires sharp high-frequency edges for near obstacles combined with smooth low-frequency global consistency for distant road horizons.
- **Design Mechanism**:
  1. **Bottom-Up Progressive Aggregation**:
     - Begins from the lowest-resolution, highest-context stage $f_4$ ($H/32, 512\text{ channels}$).
     - Projects $f_4 \to 256$ channels and upsamples $2\times$.
     - Adds skip-connection $f_3$, followed by a $3 \times 3$ BatchNorm-ReLU refinement convolution.
  2. **Hierarchical Refinement**:
     - Recursively repeats this process for stages $f_2$ and $f_1$:
       $$x_{i-1} = \text{Refine}_{i-1}\Big(\text{Upsample}(x_i) + \text{Proj}_{i-1}(f_{i-1})\Big)$$
  3. **Head & Activation**:
     - Upsamples from $H/4 \to H$ using bilinear interpolation.
     - Projects through a $3 \times 3$ Conv ($256 \to 64$) and a $1 \times 1$ Conv ($64 \to 1$).
     - Applies Sigmoid activation: $\hat{y}_{depth} \in (0, 1)$.

---

## 3. Loss Strategy & Optimization Formulation

### 3.1 Individual Task Losses

#### A. Semantic Segmentation Loss ($\mathcal{L}_{seg}$)
Combines Cross-Entropy with multi-class Soft Dice Loss:
$$\mathcal{L}_{seg} = 0.7 \cdot \mathcal{L}_{CE} + 0.3 \cdot \mathcal{L}_{Dice}$$
- **$\mathcal{L}_{CE}$ with Label Smoothing ($\epsilon = 0.1$)**: Prevents overconfidence on dominant classes (Road, Sky) and regularizes boundary pixels.
- **$\mathcal{L}_{Dice}$**: Computes per-class intersection-over-union overlaps directly in probability space, preventing small critical classes (Pedestrian, Pole, Traffic Sign) from being drowned out by massive road/building areas:
  $$\mathcal{L}_{Dice} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_i p_{i, c} y_{i, c} + 1}{\sum_i p_{i, c} + \sum_i y_{i, c} + 1}$$

#### B. Depth Estimation Loss: Reverse Huber (BerHu Loss)
Standard $L_1$ or $L_2$ losses either under-penalize large errors or over-penalize outlier noise. The **Reverse Huber (BerHu)** loss behaves as $L_1$ for small residual errors and transitions to $L_2$ for large errors:
$$\mathcal{L}_{depth}(p, t) = \begin{cases} |p - t| & \text{if } |p - t| \le c \\ \frac{(p - t)^2 + c^2}{2c} & \text{if } |p - t| > c \end{cases}$$
where threshold $c = 0.2 \cdot \max(|p - t|)$. This guarantees smooth continuous road gradients while sharply penalizing vehicle and pedestrian depth boundaries.

---

### 3.2 Dynamic Task Weighting: Homoscedastic Task Uncertainty (Kendall et al.)

#### The Multi-Task Gradient Conflict Problem
If we simply minimize $\mathcal{L}_{total} = \mathcal{L}_{seg} + \mathcal{L}_{depth}$, training fails:
- At convergence, $\mathcal{L}_{seg} \approx 0.60$ while $\mathcal{L}_{depth} \approx 0.03$.
- The segmentation gradients are $20\times$ larger, dominating backpropagation and starving the depth decoder.

#### The Mathematical Formulation
We model task uncertainty by treating each task's output as a probabilistic distribution with learnable task-dependent homoscedastic uncertainty $\sigma_1, \sigma_2$:
$$\mathcal{L}_{total}(W, \sigma_1, \sigma_2) = \frac{1}{\sigma_1} \mathcal{L}_{seg} + \log \sigma_1 + \frac{1}{\sigma_2} \mathcal{L}_{depth} + \log \sigma_2$$

For numerical stability, we re-parameterize using log-variance $s = \log \sigma$:
$$\mathcal{L}_{total} = \exp(-s_1) \mathcal{L}_{seg} + s_1 + \exp(-s_2) \mathcal{L}_{depth} + s_2$$

#### How It Operates During Training
1. **Self-Balancing**: If task 2 (depth) has a small numerical loss, $\exp(-s_2)$ automatically scales up its weight $w_{depth} = \exp(-s_2)$ to balance gradient magnitudes.
2. **Regularization Anchor**: The penalty terms $+ s_1$ and $+ s_2$ prevent the model from trivially minimizing loss by setting $\sigma \to \infty$.
3. **Training Dynamics Observed Over 80 Epochs**:
   - **Epoch 1**: $w_{seg} = 1.000, \quad w_{depth} = 1.000$ (Equal initial balance).
   - **Epoch 20**: $w_{seg} = 1.552, \quad w_{depth} = 12.410$.
   - **Epoch 80 (Convergence)**: $w_{seg} \approx 1.594, \quad w_{depth} \approx 33.000$.

---

## 4. Benchmark Results & Verification

### 4.1 Overall Perception Performance (Cityscapes Validation Set — 500 Images)

| Task / Metric | Single-Task Baseline | Multi-Task Model (Ours) | Parity / Retention |
| :--- | :---: | :---: | :---: |
| **Segmentation mIoU (with TTA)** | **$80.00\%$** | **$79.81\%$** | **$99.8\%$ Baseline Parity** |
| **Depth AbsRel (lower is better)** | **$0.1003$** | **$0.1129$** | **$\approx 11\%$ Relative Error** |
| **Depth $\delta < 1.25$ (higher is better)** | **$89.56\%$** | **$87.11\%$** | **$87.11\%$ Within $25\%$ of LiDAR** |
| **Depth $\delta < 1.25^2$** | **$97.79\%$** | **$97.39\%$** | **$97.39\%$ Accuracy** |
| **Depth $\delta < 1.25^3$** | **$99.24\%$** | **$99.17\%$** | **$99.17\%$ Accuracy** |
| **Depth RMSE** | **$0.0678$** | **$0.0703$** | **$0.0703$** |
| **Total Parameters** | **$53.2\text{ M}$** | **$34.8\text{ M}$** | **$-34.6\%$ Savings** |
| **GPU Latency (RTX 4500 Ada)** | **$19.2\text{ ms}$** | **$9.7\text{ ms}$** | **$2\times$ Speedup** |

---

### 4.2 Per-Class Semantic Segmentation Breakdown (19 Cityscapes Classes)

| Class ID | Class Name | Category | MTL Model IoU | Baseline IoU | Change |
| :---: | :--- | :--- | :---: | :---: | :---: |
| `0` | **Road** | Flat | **$98.40\%$** | $98.37\%$ | $+0.03\%$ |
| `1` | **Sidewalk** | Flat | **$86.60\%$** | $86.53\%$ | $+0.07\%$ |
| `2` | **Building** | Construction | **$93.12\%$** | $93.22\%$ | $-0.10\%$ |
| `3` | **Wall** | Construction | **$59.55\%$** | $64.23\%$ | $-4.68\%$ |
| `4` | **Fence** | Construction | **$62.88\%$** | $61.60\%$ | $+1.28\%$ |
| `5` | **Pole** | Object | **$67.52\%$** | $66.71\%$ | $+0.81\%$ |
| `6` | **Traffic Light** | Object | **$71.91\%$** | $72.02\%$ | $-0.11\%$ |
| `7` | **Traffic Sign** | Object | **$80.27\%$** | $80.25\%$ | $+0.02\%$ |
| `8` | **Vegetation** | Nature | **$92.97\%$** | $93.00\%$ | $-0.03\%$ |
| `9` | **Terrain** | Nature | **$64.68\%$** | $64.75\%$ | $-0.07\%$ |
| `10` | **Sky** | Sky | **$95.34\%$** | $95.29\%$ | $+0.05\%$ |
| `11` | **Person** | Human | **$82.87\%$** | $83.10\%$ | $-0.23\%$ |
| `12` | **Rider** | Human | **$62.24\%$** | $62.97\%$ | $-0.73\%$ |
| `13` | **Car** | Vehicle | **$95.38\%$** | $95.25\%$ | $+0.13\%$ |
| `14` | **Truck** | Vehicle | **$83.94\%$** | $84.47\%$ | $-0.53\%$ |
| `15` | **Bus** | Vehicle | **$88.98\%$** | $88.35\%$ | $+0.63\%$ |
| `16` | **Train** | Vehicle | **$82.41\%$** | $81.75\%$ | $+0.66\%$ |
| `17` | **Motorcycle** | Vehicle | **$68.92\%$** | $69.70\%$ | $-0.78\%$ |
| `18` | **Bicycle** | Vehicle | **$78.34\%$** | $78.35\%$ | $-0.01\%$ |

---

## 5. Interactive Perception Cockpit & Deployment

### 5.1 System Architecture
The application is organized into a modular full-stack perception suite:
- **Backend**: Built with **FastAPI** (`app/server.py`). Dynamically detects device:
  $$\text{DEVICE} = \text{"cuda" if torch.cuda.is_available() else "cpu"}$$
- **Frontend**: Custom responsive dashboard using HTML5, modern CSS3 variables, and vanilla ES6 JavaScript (`app/static/`).
- **WebGL 3D Engine**: Client-side **Three.js** renderer executing at $60\text{ FPS}$ on browser GPU hardware.

---

### 5.2 Key Features of the Dashboard

1. **Triple Simultaneous Perception Gallery**:
   - Displays all three modalities side-by-side simultaneously:
     - **Left**: Raw RGB Camera Feed.
     - **Center**: 19-Class Semantic Mask.
     - **Right**: Calibrated Metric Depth Map.

2. **Physical Metric Depth Calibration**:
   - Converts network Sigmoid outputs directly into real-world meters:
     $$Z_{\text{meters}} = \text{depth\_sigmoid} \times 79.0 + 1.0 \quad (1.0\text{m} \le Z \le 80.0\text{m})$$
   - Uses **Google Turbo colormap** with robust percentile normalization ($2^{\text{nd}} - 98^{\text{th}}$ percentile), ensuring close obstacles ($<5\text{m}$) appear in bright Red/Orange, mid-range traffic ($5-25\text{m}$) in Yellow/Green, and distant background ($>30\text{m}$) in Deep Blue.

3. **Synchronized Cursor Hover Prober**:
   - Hovering over *any* pixel on *any* panel displays:
     - **Semantic Class**: Name & Category (e.g. `Pedestrian [Human]`, `Car [Vehicle]`).
     - **True Metric Distance**: Accurate distance in meters (e.g. `8.4 m`).
     - **Safety State**: `SAFE (>15m)`, `CAUTION (5-15m)`, or `CRITICAL DANGER (<5m)`.

4. **Automotive Collision Radar & Hazard Detection**:
   - Uses **Connected-Component Object Clustering** on dynamic classes (Pedestrians, Cyclists, Cars, Trucks) to count active physical hazards rather than raw pixel counts.
   - Computes:
     - Closest Obstacle distance ($m$).
     - Road Mid-Range driving distance ($m$).
     - Horizon distance ($m$).
     - Dynamic hazards within the $15\text{m}$ stopping envelope.

5. **1-Click Cross-Platform Launchers**:
   - Windows: Double-click [`run_demo.bat`](file:///c:/Users/Student4.DESKTOP-2052K3N/Desktop/Multi%20Tasking%20Learning/Capstone/run_demo.bat).
   - Cross-platform: Run `python run_demo.py`. Automatically spins up the server and opens `http://localhost:8000` in the default browser.

---

## 6. Personal Laptop (CPU-Only) Setup Instructions

When transitioning to a personal laptop without a dedicated GPU:

```bash
# 1. Clone the repository
git clone https://github.com/teju701/Capstone.git
cd Capstone

# 2. Create and activate a clean virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the trained model weights
# Place best_mtl_model.pth inside checkpoints/mtl/best_mtl_model.pth

# 5. Launch
run_demo.bat   # or python run_demo.py
```

---

## 7. Panel Defense Cheat Sheet: Anticipated Questions & Answers

### Q1: "Why use Multi-Task Learning instead of two separate models?"
> **Answer**: In autonomous driving, onboard compute is limited by power and thermals. Running two separate backbones costs $53.2\text{M}$ parameters and twice the FLOPs. Our MTL architecture shares the hierarchical SegFormer MiT-B2 encoder, saving $34.6\%$ parameters and cutting inference latency in half ($9.7\text{ms}$ on GPU) while maintaining $99.8\%$ baseline segmentation parity ($79.81\%$ vs $80.00\%$).

### Q2: "How did you prevent the segmentation loss from dominating the depth loss?"
> **Answer**: We implemented Kendall's Homoscedastic Uncertainty Weighting. Instead of manual trial-and-error loss weights, the model learns task log-variances ($s_{seg}, s_{depth}$) dynamically. Because depth loss has smaller numerical scale ($\approx 0.03$ vs $\approx 0.60$), the uncertainty formulation automatically scaled up the depth loss weight $w_{depth} = \exp(-s_{depth})$ from $1.0$ to $\approx 33.0$, ensuring balanced gradient propagation throughout backpropagation.

### Q3: "How does the model calculate depth in meters from a single RGB camera?"
> **Answer**: The network is trained on stereo disparity ground truth from Cityscapes. Disparity $d$ relates to metric depth via stereo baseline $B = 0.2093\text{m}$ and focal length $f = 2262.52\text{px}$: $Z = \frac{f \cdot B}{d}$. We linearly normalize depth in $[1.0\text{m}, 80.0\text{m}]$, and during live inference, the Sigmoid prediction $\hat{y}$ is calibrated via $Z_{\text{meters}} = \hat{y} \times 79.0 + 1.0$.
