# RADAR Codebase — Complete Explanation

> **RADAR** = **R**easoning and **A**rtifact **D**etection for deepfake **R**ecognition

This document walks you through every file in the project, in the order you'd need to understand them — from "what is this project?" all the way down to "what does line X do?". By the end you should understand the full architecture, data pipeline, training loop, and evaluation without any hand-waving.

---

## Table of Contents

1. [The Big Picture — What RADAR Actually Does](#1-the-big-picture)
2. [Project Structure After Cleanup](#2-project-structure)
3. [Setup & Environment (`setup_env.sh`, `requirements.txt`)](#3-setup--environment)
4. [Data Download & Extraction (`0_download_combined_dataset.sh`, `1_fast_extract.py`)](#4-data-download--extraction)
5. [Data Pipeline — Loading Images (`src/data/`)](#5-data-pipeline)
   - 5a. [How Images Are Found (`splits.py`)](#5a-how-images-are-found)
   - 5b. [How Images Become Tensors (`dataset.py`)](#5b-how-images-become-tensors)
   - 5c. [Training From ZIP Without Extracting (`zip_loader.py`, `zip_dataset.py`)](#5c-training-from-zip)
6. [The RADAR Model — Core Architecture (`src/method/`)](#6-the-radar-model)
   - 6a. [Package Init (`__init__.py`)](#6a-package-init)
   - 6b. [The Backbone — Any Architecture (`radar.py`)](#6b-the-backbone)
   - 6c. [Boundary Artifact Detection Module — BADM (`boundary.py`)](#6c-badm)
   - 6d. [Frequency Artifact Detection Module — AADM (`frequency.py`)](#6d-aadm)
   - 6e. [Evidence Refinement Module — ERM (`reasoning.py`)](#6e-erm)
   - 6f. [The Full Forward Pass (`radar.py` continued)](#6f-the-full-forward-pass)
   - 6g. [Loss Function (`loss.py`)](#6g-loss-function)
7. [Training Scripts](#7-training-scripts)
   - 7a. [Main Training (`2_train_model.py`)](#7a-main-training)
   - 7b. [Shared Training Engine (`src/experiments/train.py`)](#7b-shared-training-engine)
   - 7c. [Config-Driven Experiment Runner (`src/experiments/run.py`)](#7c-config-driven-runner)
8. [Ablation Studies (`3_train_ablation.py`, `run_ablations.sh`)](#8-ablation-studies)
9. [Results Comparison (`4_compare_ablations.py`)](#9-results-comparison)
10. [Attention Visualization (`5_visualize_attention.py`)](#10-attention-visualization)
11. [Baseline Models (`src/baselines/`)](#11-baseline-models)
12. [Analysis Utilities (`src/analysis/`)](#12-analysis-utilities)
13. [Utility Helpers (`src/utils/`)](#13-utility-helpers)
14. [Experiment Configs (`src/experiments/configs/`)](#14-experiment-configs)
15. [How Everything Connects — End-to-End Flow](#15-end-to-end-flow)

---

## 1. The Big Picture

**Problem:** Given a face image, decide if it's **real** or **AI-generated (deepfake)**.

**Insight:** Deepfake generators leave invisible artifacts that humans can't see but algorithms can detect:
- **Boundary artifacts** — unnatural edges around blended regions (hair/face boundary, jawline, etc.)
- **Frequency artifacts** — unusual patterns in the frequency domain (deepfakes often lose high-frequency detail or introduce periodic noise)

**RADAR's approach (3-module design):**

```
                              Input Image (224×224)
                                       │
                            ┌──────────┴──────────┐
                            │    ViT-Small Backbone │  ← pretrained on ImageNet
                            │    (frozen first 3 ep)│
                            └──────────┬──────────┘
                           ┌───────────┴───────────┐
                           │                       │
                    Patch Tokens              CLS Token
                    (196 × 384)                (384)
                           │                       │
                      ┌────┴────┐             ┌────┴────┐
                      │  BADM   │             │  AADM   │
                      │(boundary│             │(frequen-│
                      │detector)│             │cy detec-│
                      └────┬────┘             │tor)     │
                           │                  └────┬────┘
                     128-dim evidence        128-dim evidence
                           │                       │
                           └───────────┬───────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                     ERM (reasoning)         External Classifier
                     3 iterations of          (simple MLP on
                     cross-attention +         concatenated evidence)
                     GRU refinement
                          │                         │
                     logit_reasoning           logit_external
                          │                         │
                          └──────────┬──────────────┘
                                     │
                              Gating: α · reasoning + (1-α) · external
                              α = sigmoid(learnable_param)
                                     │
                              Final prediction
                              (0 = real, 1 = fake)
```

The key innovation is that instead of just throwing everything at one classifier, RADAR:
1. Forces **BADM** to specialize on boundary/edge artifacts
2. Forces **AADM** to specialize on frequency-domain artifacts
3. Uses **ERM** to iteratively reason over both pieces of evidence (like a detective reviewing clues multiple times)
4. Has a **gating mechanism** that learns whether to trust the reasoning path or the direct path more

---

## 2. Project Structure

After cleanup, here's every file and what it does:

```
RADAR/
├── README.md                          # Quick-start guide
├── CLAUDE.md                          # AI assistant context file
├── requirements.txt                   # Python dependencies
├── setup_env.sh                       # One-command environment setup
│
├── 0_download_combined_dataset.sh     # Download 227k-frame dataset from Kaggle
├── 1_fast_extract.py                  # Parallel ZIP extraction (optional)
├── 2_train_model.py                   # ** Main training script **
├── 3_train_ablation.py                # Baseline vs Baseline+RADAR for each backbone
├── 4_compare_ablations.py             # Compare ablation results in tables + LaTeX
├── 5_visualize_attention.py           # Generate attention visualization plots
├── run_ablations.sh                   # Run all 4 backbone ablation experiments
│
└── src/                               # Core library code
    ├── method/                        # The RADAR model itself
    │   ├── __init__.py                # Public API exports
    │   ├── radar.py                   # Main model class + ViT backbone
    │   ├── boundary.py                # BADM — edge/boundary detection
    │   ├── frequency.py               # AADM — FFT frequency detection
    │   ├── reasoning.py               # ERM — iterative evidence refinement
    │   └── loss.py                    # Multi-component loss function
    │
    ├── data/                          # Data loading pipeline
    │   ├── splits.py                  # Find images, split train/val/test
    │   ├── dataset.py                 # PyTorch Dataset + augmentations
    │   ├── zip_loader.py              # Parse splits from inside a ZIP
    │   └── zip_dataset.py             # PyTorch Dataset that reads from ZIP
    │
    ├── baselines/                     # Comparison models
    │   ├── __init__.py                # Exports all baselines
    │   ├── vit.py                     # ViT-Small (same backbone, no modules)
    │   ├── resnet.py                  # ResNet-50
    │   ├── xception.py                # Xception (popular in deepfake lit.)
    │   └── efficientnet.py            # EfficientNet-B0
    │
    ├── experiments/                   # Training engine + experiment runner
    │   ├── train.py                   # Shared train_epoch / evaluate / train_model
    │   ├── run.py                     # Config-driven experiment launcher
    │   └── configs/                   # YAML experiment configurations
    │       ├── radar.yaml
    │       ├── cross_domain.yaml
    │       └── wilddeepfake.yaml
    │
    ├── analysis/                      # Post-training analysis tools
    │   ├── stats.py                   # Bootstrap CI, DeLong test
    │   ├── aggregate.py               # Multi-seed result aggregation
    │   └── plots.py                   # Training curves, ROC, bar charts
    │
    └── utils/                         # Small helpers
        ├── __init__.py
        └── logging.py                 # Pretty-print helpers
```

---

## 3. Setup & Environment

### `setup_env.sh`

This is a bash script that creates a conda environment called `ml` and installs everything:

```
1. Sources conda so the `conda` command works
2. Checks if an 'ml' environment already exists — if yes, just activates it
3. If not, creates one with Python 3.11
4. Detects if you're on Apple Silicon (arm64) → installs PyTorch with CPU/MPS
   or Intel/Linux → installs PyTorch with CUDA 11.8
5. Installs everything from requirements.txt
```

### `requirements.txt`

The key dependencies and why they're needed:

| Package | Why |
|---------|-----|
| `torch`, `torchvision` | Deep learning framework — the entire model is built on PyTorch |
| `timm` | "PyTorch Image Models" — provides pretrained ViT-Small and other backbones |
| `albumentations` | Image augmentation library (faster than torchvision transforms) |
| `scikit-learn` | For AUC, accuracy metrics |
| `scipy` | For statistical tests (DeLong test) |
| `matplotlib`, `seaborn` | Visualization |
| `tqdm` | Progress bars |
| `kagglehub` | Dataset download |
| `tabulate` | Pretty-printing result tables |

---

## 4. Data Download & Extraction

### `0_download_combined_dataset.sh`

Downloads a public Kaggle dataset called "radar-deepfake-frames" (227k pre-extracted face frames):

**What it does step by step:**
1. Sets the output directory (default: `./data`)
2. Checks your OS and installs `wget` if missing
3. Downloads the ZIP file from Kaggle's public API (no authentication needed)
4. Tells you two ways to use the data:
   - **Instant mode**: Pass the ZIP path directly to the training script (no extraction!)
   - **Extract mode**: Extract first, then train from the directory

The dataset inside the ZIP has this structure:
```
combined/
├── train/
│   ├── real/    ← ~91k real face images
│   └── fake/    ← ~91k deepfake face images
├── val/
│   ├── real/    ← ~11k
│   └── fake/    ← ~11k
└── test/
    ├── real/    ← ~11k
    └── fake/    ← ~11k
```

### `1_fast_extract.py`

If you want to extract the ZIP (for faster subsequent loading), this script parallelizes it:

**How it works:**
1. Opens the ZIP and gets the list of all files
2. Splits the file list into chunks — one per CPU core
3. Uses Python's `multiprocessing.Pool` to extract chunks in parallel
4. Each worker opens its own copy of the ZIP file (ZIP files aren't thread-safe, but separate file handles in separate processes are fine)

**Why parallel?** A 227k-file ZIP on a single thread takes ~30 minutes. With 8 cores, ~4 minutes. With 96 cores (cloud GPU), ~30 seconds.

---

## 5. Data Pipeline

### 5a. How Images Are Found — `src/data/splits.py`

This file figures out WHERE your images are and splits them into train/val/test.

**Three key functions:**

#### `is_presplit_dataset(data_dir)`
Simply checks if `data_dir/train/` and `data_dir/val/` folders exist. If yes, the dataset is already split.

#### `load_presplit_data(data_dir, split)`
For pre-split datasets. Given `split="train"`, it:
1. Looks at `data_dir/train/real/` and `data_dir/train/fake/`
2. Globs for `*.jpg`, `*.png`, `*.jpeg` files
3. Assigns label `0` for real, `1` for fake
4. Returns two lists: `image_paths` and `labels`

#### `load_domain_data(data_dir, domain)` + `create_stratified_split(...)`
For single-domain datasets (e.g., `data_dir/stylegan/real/`, `data_dir/stylegan/fake/`):
1. `load_domain_data` collects all images from one domain folder
2. `create_stratified_split` shuffles them and splits 80/10/10 (configurable), keeping class balance by splitting real and fake indices separately then merging

**Important detail:** The split requires at least 100 images per class to avoid tiny datasets that would give unreliable results.

### 5b. How Images Become Tensors — `src/data/dataset.py`

#### The transforms

Two transform pipelines built with `albumentations`:

**Training transforms** (`get_train_transforms`):
```
Resize to 224×224
→ Random horizontal flip (50% chance)
→ Random shift/scale/rotate (30% chance, small amounts)
→ Color jitter (30% chance, subtle brightness/contrast changes)
→ Coarse dropout (20% chance, randomly mask 4 small patches)
→ Normalize to ImageNet mean/std
→ Convert to PyTorch tensor
```

**Why these specific augmentations?**
- Horizontal flip: Faces can be either way
- Shift/scale/rotate: Simulates slight misalignment
- Color jitter: Makes the model robust to lighting
- Coarse dropout: Regularization (forces model to not rely on one region)
- **NOT included**: GaussianBlur and ImageCompression — these would destroy the boundary and frequency artifacts that RADAR is specifically trying to detect!

**Validation/test transforms** (`get_val_transforms`):
```
Resize to 224×224 → Normalize → Convert to tensor
```
No augmentation — we want deterministic evaluation.

#### `DeepfakeDataset` class

A standard PyTorch `Dataset`:

- **`__init__`**: Stores image paths, labels, transform, and optional path to cached preprocessed features
- **`__getitem__(idx)`**:
  1. Opens the image at `self.image_paths[idx]` using PIL
  2. Converts to RGB numpy array
  3. Applies the transform (augmentations + normalization + to tensor)
  4. If preprocessed features exist on disk (Sobel edge maps), loads them too
  5. Returns `(image_tensor, label)` or `(image_tensor, label, extras_dict)`

The optional caching of Sobel edge maps (`_get_cache_path`) uses MD5 hashing of the image file to create unique cache filenames. This avoids recomputing Sobel filters every epoch, but in practice the training scripts don't use this — Sobel is computed on-the-fly inside the model.

### 5c. Training From ZIP — `src/data/zip_loader.py` + `src/data/zip_dataset.py`

These allow training **directly from the ZIP** without extracting to disk.

#### `zip_loader.py`

**`get_zip_splits(zip_path)`:**
1. Opens the ZIP, gets the full file listing
2. Filters for image files (`.jpg`, `.png`, `.jpeg`)
3. For each split (train/val/test), finds files matching the pattern `combined/{split}/real/` or `combined/{split}/fake/`
4. Returns a dict: `{"train": (paths, labels), "val": (paths, labels), "test": (paths, labels)}`

#### `zip_dataset.py` — `ZipDeepfakeDataset`

Like `DeepfakeDataset` but reads from inside a ZIP:

**Key challenge:** PyTorch DataLoader uses multiple worker processes. Each worker needs its own `ZipFile` handle (they aren't safely shared between processes).

**Solution:** The `_get_zip()` method tracks which worker is currently using the dataset. When a new worker touches it for the first time, it opens a fresh `ZipFile` handle. This is tracked via `_worker_id`.

**`__getitem__(idx)`:**
1. Gets (or opens) the ZIP handle for the current worker
2. Reads the raw bytes of the image from inside the ZIP
3. Opens the bytes with PIL (using `io.BytesIO`)
4. Applies transforms, returns `(tensor, label)`

---

## 6. The RADAR Model

### 6a. Package Init — `src/method/__init__.py`

Exports the public API:
```python
RADAR, RADARConfig        # Main model + its config dataclass
FrequencyArtifactDetector  # AADM
BoundaryArtifactDetector   # BADM
EvidenceRefinementModule   # ERM
RADARLoss, LossConfig      # Loss function
```

### 6b. The Backbone — ViT (`src/method/radar.py`)

#### `RADARConfig` (dataclass)

All hyperparameters in one place:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `img_size` | 224 | Input image size |
| `patch_size` | 16 | ViT splits image into 16×16 patches |
| `embed_dim` | 384 | ViT-Small's hidden dimension |
| `evidence_dim` | 128 | Size of BADM/AADM output vectors |
| `reasoning_iterations` | 3 | How many times ERM refines its answer |
| `reasoning_heads` | 4 | Number of attention heads in ERM |
| `fft_size` | 112 | Resolution for FFT computation |
| `dropout` | 0.1 | Dropout rate |
| `gating_init` | 0.5 | Initial balance between reasoning/external |
| `freeze_backbone_epochs` | 3 | Epochs before ViT starts training |

#### `RADAR` class — `__init__`

```python
self.encoder = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
```

This loads a **ViT-Small** pretrained on ImageNet. `num_classes=0` removes the classification head — we just want features.

**ViT-Small specs:** 22M parameters, splits 224×224 image into 14×14 = 196 patches of 16×16 pixels each, produces 384-dimensional features.

When you call `encoder.forward_features(x)`, it returns a tensor of shape `(batch, 197, 384)`:
- Position 0: the **CLS token** — a single 384-dim vector summarizing the whole image
- Positions 1–196: **patch tokens** — one 384-dim vector per image patch

Then it creates:
- `self.badm` — Boundary module (gets patch tokens)
- `self.aadm` — Frequency module (gets CLS token)
- `self.reasoning` — ERM (gets evidence from both)
- `self.external_classifier_dual` — MLP that takes concatenated evidence (256 → 128 → 1)
- `self.external_classifier_single` — MLP for when only one module is active (128 → 64 → 1)
- `self._gating_logit` — A single learnable scalar, passed through sigmoid to get α ∈ (0, 1)

**Weight initialization** (`_init_weights`): Linear layers get Xavier uniform init, Conv2d layers get Kaiming normal init. This helps training start from a reasonable point.

#### Backbone freezing/unfreezing

```python
def freeze_backbone(self):    # Set all ViT params to requires_grad=False
def unfreeze_backbone(self):  # Set them back to True
def get_backbone_params(self): # Return ViT parameters (for differential LR)
def get_head_params(self):     # Return everything except ViT (BADM, AADM, ERM, classifiers, gating)
```

**Why freeze?** The ViT is pretrained and already good at extracting visual features. If you train everything from scratch, the ViT's gradients dominate and BADM/AADM never learn anything useful. By freezing the ViT for 3 epochs, BADM/AADM are forced to learn from the ViT's fixed (good) features. After unfreezing, the ViT fine-tunes gently (at 10× lower learning rate).

### 6c. BADM — Boundary Artifact Detection Module (`src/method/boundary.py`)

**Purpose:** Detect forged boundary artifacts (unnatural edges where face parts are blended).

#### Building blocks

**`ConvBlock`**: Conv2d → BatchNorm → ReLU. The basic unit of every CNN.

**`ResidualBlock`**: Two ConvBlocks with a skip connection (input is added back to output). This solves the vanishing gradient problem — gradients can flow through the skip connection even if the conv layers are difficult to train.

**`SpatialAttention`**: 
1. Computes the mean and max across channels at each spatial position
2. Concatenates them → 2-channel tensor
3. Passes through a 7×7 conv → sigmoid
4. Multiplies the attention map with the original features

This makes the model focus on the most informative spatial regions (where edges are most suspicious).

#### `BoundaryArtifactDetector.forward(image, patch_features, sobel_cached)`

**Step 1 — Compute Sobel edge map:**
```
RGB image → Grayscale (using standard luminance weights 0.299R + 0.587G + 0.114B)
→ Sobel X filter (horizontal edges): [[-1,0,1],[-2,0,2],[-1,0,1]]
→ Sobel Y filter (vertical edges): [[-1,-2,-1],[0,0,0],[1,2,1]]
→ Edge magnitude = sqrt(grad_x² + grad_y²)
→ Normalize to [0, 1]
```

The Sobel filter is a classical edge detector. It highlights every boundary in the image. Deepfake artifacts show up as abnormally strong or strangely shaped edges.

**Step 2 — Encode edges through CNN:**
```
(B, 1, 224, 224) edge map
→ ConvBlock(1→32, 7×7, stride 2)     → (B, 32, 112, 112)
→ ConvBlock(32→64, 3×3, stride 2)    → (B, 64, 56, 56)
→ ResidualBlock(64)                    → (B, 64, 56, 56)
→ SpatialAttention(64)                → (B, 64, 56, 56)  [attention-weighted]
→ ConvBlock(64→128, 3×3, stride 2)   → (B, 128, 28, 28)
→ ResidualBlock(128)                   → (B, 128, 28, 28)
→ ConvBlock(128→256, 3×3, stride 2)  → (B, 256, 14, 14)
→ AdaptiveAvgPool2d(1)                → (B, 256, 1, 1)
→ Flatten                             → (B, 256)
```

**Step 3 — Process ViT patch features:**
```
patch_features: (B, 196, 384) — one vector per image patch
→ Mean pool across patches:  (B, 384)
→ Max pool across patches:   (B, 384)
→ Concatenate:               (B, 768)
→ Linear(768→384) → ReLU → Dropout → Linear(384→192) → ReLU
→ Result:                    (B, 192)
```

**Why mean AND max?** Mean captures the average patch information, max captures the most activated patch. Together they provide a richer summary.

**Step 4 — Fuse and classify:**
```
Concatenate edge features (256) + patch features (192) → (B, 448)
→ Linear(448→256) → ReLU → Dropout → Linear(256→128)
→ evidence: (B, 128)
→ Linear(128→1) → logit
```

**Output:** `{"evidence": (B, 128), "logit": (B, 1), "score": sigmoid(logit)}`

The 128-dim evidence vector is the BADM's "summary of boundary artifacts found." It gets passed to the ERM for reasoning.

### 6d. AADM — Frequency Artifact Detection Module (`src/method/frequency.py`)

**Purpose:** Detect frequency-domain artifacts (compression patterns, spectral anomalies).

#### `compute_frequency_spectrum(image, target_size=112)`

This is the core signal processing function:

**Step 1 — Convert to grayscale and resize to 112×112:**
Frequency analysis works on grayscale. We resize to a smaller resolution because FFT is O(n²) and we don't need full resolution for frequency patterns.

**Step 2 — 2D FFT:**
```python
fft = torch.fft.fft2(gray)           # Complex Fourier transform
fft_shifted = torch.fft.fftshift(fft) # Move zero-frequency to center
magnitude = torch.abs(fft_shifted)    # Get magnitude (discard phase)
```

The FFT converts a spatial image into the frequency domain. Each pixel in the FFT output represents how much of a particular frequency exists in the image. Low frequencies (smooth gradients) are at the center, high frequencies (sharp edges, noise) are at the edges.

**Step 3 — Multi-scale frequency band extraction:**

Creates 3 high-pass filters at different cutoff frequencies:
```
Band 1: cutoff = H/16 (keeps only very high frequencies — fine noise)
Band 2: cutoff = H/8  (keeps high frequencies — edges, textures)  
Band 3: cutoff = H/4  (keeps medium-high frequencies — structures)
```

For each band:
1. Create a distance map from center
2. Apply soft high-pass: `sigmoid((distance - cutoff) / 10)` — smoothly passes frequencies above the cutoff
3. Multiply with the FFT magnitude
4. Apply `log1p` (log(1+x)) to compress the dynamic range
5. Normalize to [0, 1]

**Result:** 3-channel tensor `(B, 3, 112, 112)` — like an RGB image but where each "color" channel is a different frequency band.

**Why 3 bands?** Different deepfake methods leave artifacts at different frequency scales. GAN-generated faces often have high-frequency checkerboard patterns. Compression artifacts appear at medium frequencies. By looking at multiple bands, the model can detect a wider variety of forgery types.

#### `FrequencyArtifactDetector.forward(raw_image, cls_token, freq_cached)`

**Step 1 — Get frequency spectrum:**
Either uses pre-computed cache or calls `compute_frequency_spectrum`.

**Step 2 — Encode frequency features through CNN:**
```
(B, 3, 112, 112) frequency spectrum
→ ConvBlock(3→48, 7×7, stride 2)     → (B, 48, 56, 56)
→ ConvBlock(48→96, 3×3, stride 2)    → (B, 96, 28, 28)
→ ResidualBlock(96)                    → (B, 96, 28, 28)
→ ConvBlock(96→192, 3×3, stride 2)   → (B, 192, 14, 14)
→ ResidualBlock(192)                   → (B, 192, 14, 14)
→ ConvBlock(192→256, 3×3, stride 2)  → (B, 256, 7, 7)
→ AdaptiveAvgPool2d(1)                → (B, 256, 1, 1)
→ Flatten                             → (B, 256)
```

**Step 3 — Process ViT CLS token:**
```
cls_token: (B, 384)
→ Linear(384→192) → ReLU → Dropout
→ Result: (B, 192)
```

**Why CLS token instead of patch tokens?** The CLS token is a global summary of the image. Frequency artifacts are global in nature (they affect the whole image's spectrum), so a global feature makes sense. BADM uses patch tokens because boundary artifacts are spatial/local.

**Step 4 — Fuse and classify:**
Same pattern as BADM: concatenate (256+192=448) → MLP → 128-dim evidence → linear → logit.

### 6e. ERM — Evidence Refinement Module (`src/method/reasoning.py`)

**Purpose:** Iteratively refine the model's understanding by attending to both BADM and AADM evidence multiple times, like a detective reviewing clues.

#### `EvidenceCrossAttention`

Standard multi-head cross-attention:

```
Input: query (B, 128), evidence (B, 2, 128)  [2 evidence sources]

1. Project query → Q: (B, 1, 4_heads, 32_dim_per_head)
2. Project evidence → K, V: (B, 2, 4, 32)
3. Attention scores: Q @ K^T / sqrt(32)  → (B, 4, 1, 2)
4. Softmax over the 2 evidence sources
5. Weighted sum: attn @ V → (B, 4, 1, 32)
6. Reshape + output projection → (B, 128)

Also returns attention weights (B, 2): how much weight each evidence source got
```

**Intuition:** The query asks "given what I currently believe, which evidence source should I pay more attention to?" If boundary evidence is very informative for this image, it gets higher attention. The next iteration can then shift focus.

#### `EvidenceRefinementModule.forward(*evidence_list)`

**Step 1 — Initialize hidden state:**
- If 2 sources: concatenate both evidence vectors (256-dim) → project to 128-dim
- If 1 source: project single evidence (128-dim) → 128-dim

**Step 2 — Iterative refinement (3 times):**

For each iteration `t`:
```
1. Cross-attention: h attends to evidence_stack → context vector
   (Also records attention weights → attention_history[t])

2. GRU update: h = GRU(context, h)
   (GRU = Gated Recurrent Unit — selectively updates the hidden state,
    keeping useful information and discarding noise)

3. LayerNorm(h)

4. Feed-Forward Network: h = h + FFN(h)
   where FFN = Linear(128→256) → GELU → Dropout → Linear(256→128) → Dropout
   (This is a residual connection with a 2-layer MLP, standard in transformers)

5. Predict: logit = Linear(h) → scalar
   (Each iteration makes its own prediction)
```

**Step 3 — Compute convergence delta:**
```
convergence_delta = |prob_iter3 - prob_iter2|
```
This measures how much the prediction changed in the last iteration. If it's near 0, the model has "converged" — it's confident in its answer. During training, you want this to be 0.001–0.005 (small but not zero).

**Output:**
```python
{
    "final_logit": logit from last iteration,
    "iteration_logits": all 3 logits (for deep supervision loss),
    "iteration_probs": sigmoid of all 3 logits,
    "attention_history": (B, 3, 2) — which evidence source got more attention at each step,
    "convergence_delta": scalar — prediction stability measure
}
```

### 6f. The Full Forward Pass (`src/method/radar.py` — `RADAR.forward`)

Putting it all together:

```python
def forward(self, x, freq_cached=None, sobel_cached=None, use_badm=True, use_aadm=True):
```

1. **Extract backbone features (ViT or CNN):**
   ```
   cls_token, patch_features = self._extract_features(x)
   # ViT:  cls_token=(B, 384), patch_features=(B, 196, 384)
   # CNN:  cls_token=(B, 2048), patch_features=(B, 49, 2048)
   ```

2. **Run detection modules (if enabled):**
   ```
   badm_out = self.badm(x, patch_features, sobel_cached) if use_badm else None
   aadm_out = self.aadm(x, cls_token, freq_cached) if use_aadm else None
   ```
   The `use_badm`/`use_aadm` flags allow disabling modules for ablation studies.

3. **Run reasoning:**
   ```
   evidence_list = [badm_evidence, aadm_evidence]  # whatever is active
   reasoning_out = self.reasoning(*evidence_list)
   ```

4. **Run external classifier (direct path):**
   ```
   If 2 sources: concat evidence → external_classifier_dual → logit
   If 1 source: evidence → external_classifier_single → logit
   ```

5. **Gate the two paths:**
   ```
   α = sigmoid(self._gating_logit)  # learnable, starts at 0.5
   final_logit = α * reasoning_logit + (1 - α) * external_logit
   ```

6. **Return everything** in a dictionary (21 keys!) so the loss function and monitoring code can access whatever they need.

### 6g. Loss Function (`src/method/loss.py`)

#### `LossConfig`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `lambda_main` | 1.0 | Weight for main classification loss |
| `lambda_branch` | 0.3 | Weight for individual BADM/AADM losses |
| `lambda_orthogonal` | 0.01 | Weight for forcing BADM ≠ AADM evidence |
| `lambda_deep_supervision` | 0.05 | Weight for per-iteration losses |
| `label_smoothing` | 0.05 | Softens labels: 1→0.975, 0→0.025 |
| `orthogonality_margin` | 0.1 | How different evidence vectors must be |

#### `RADARLoss.forward(outputs, labels, use_badm, use_aadm)`

**Component 1 — Main loss (λ=1.0):**
```
BCE(final_logit, smoothed_labels)
```
Standard binary cross-entropy on the final gated prediction. Label smoothing prevents overconfidence.

**Component 2 — Branch loss (λ=0.3):**
```
0.5 * (BCE(badm_logit, labels) + BCE(aadm_logit, labels))
```
Each module's individual classifier should also predict correctly. This provides direct gradient signal to each module (instead of only getting gradients through the reasoning path).

**Component 3 — Orthogonality loss (λ=0.01):**
```
1. L2-normalize BADM and AADM evidence vectors
2. Compute cosine similarity between them
3. loss = ReLU(|cosine_similarity| - 0.1)
```
**What this does:** Forces BADM and AADM to learn DIFFERENT features. Without this, both modules might learn the same thing (redundant). The margin of 0.1 means: "it's okay if they're slightly correlated (cosine < 0.1), but penalize anything beyond that."

**Intuition:** If cosine similarity = 1.0, they're identical → loss is high. If cosine = 0.0, they're orthogonal (completely different information) → loss is 0. We want them to capture complementary information.

**Component 4 — Deep supervision loss (λ=0.05):**
```
For each of the 3 ERM iterations:
  loss_i = BCE(iteration_i_logit, labels)
  weight_i = linearly increasing (later iterations matter more)
Total = weighted sum of all iteration losses
```
**Why?** Without this, only the final iteration gets direct gradient signal. The earlier iterations might not learn anything useful. Deep supervision ensures every iteration makes a reasonable prediction, with later iterations being penalized more heavily (they should be more accurate).

**Final total:**
```
total = 1.0 * main + 0.3 * branch + 0.01 * orthogonal + 0.05 * deep_supervision
```

---

## 7. Training Scripts

### 7a. Main Training — `2_train_model.py`

This is **the** script you run. It orchestrates everything.

#### `main()` flow:

1. **Parse arguments** — data_dir, batch_size, epochs, LR, seed, etc.

2. **Auto-detect GPU capabilities:**
   ```python
   if torch.cuda.is_available():
       gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
       optimal_batch = 128 if gpu_mem > 40 else 64  # More VRAM → bigger batch
       optimal_workers = min(8, cpu_count - 1)
   ```

3. **Build config dictionary** — all hyperparameters in one dict.

4. **Set seed** — For reproducibility: Python random, numpy, torch, CUDA all seeded.

5. **Save config** — Dumps to `outputs/<experiment_name>/config.yaml` so you always know what settings produced what results.

6. **Load data** — Detects whether input is a ZIP or directory, pre-split or single-domain. Creates `DeepfakeDataset` or `ZipDeepfakeDataset` accordingly.

7. **Create DataLoaders:**
   ```python
   train_loader = DataLoader(
       ..., shuffle=True,       # Random order each epoch
       pin_memory=True,         # Speed up CPU→GPU transfer
       drop_last=True,          # Drop incomplete final batch
       persistent_workers=True, # Keep workers alive between epochs
       prefetch_factor=4        # Pre-load 4 batches per worker
   )
   ```

8. **Create model** — `RADAR(config).to(device)`

9. **Call `train_model()`** — the training loop (defined in this file, not the one in `src/experiments/train.py`).

10. **Evaluate on test set** — Load best checkpoint, run evaluation, save metrics.

#### `train_model()` — the training loop

```
For each epoch:
  1. If epoch == freeze_epochs: unfreeze backbone
  2. Run train_epoch() → get losses
  3. Run evaluate() → get val AUC, accuracy
  4. Print summary
  5. If val AUC improved: save checkpoint, reset patience
  6. If patience exceeded: early stop
```

**Optimizer setup:**
```python
param_groups = [
    {"params": head_params, "lr": 0.0005},      # BADM, AADM, ERM, classifiers
    {"params": backbone_params, "lr": 0.00005},  # ViT (10× lower!)
]
optimizer = AdamW(param_groups, weight_decay=0.05)
```

**Why differential LR?** The ViT is already pretrained — it only needs gentle fine-tuning. The new modules (BADM/AADM/ERM) start from random initialization and need larger updates.

**Scheduler:** `OneCycleLR` with cosine annealing:
- Starts low, warms up to max LR over 10% of training
- Then gradually decreases following a cosine curve
- This "super-convergence" schedule often trains faster than constant LR

**AMP (Automatic Mixed Precision):** Uses `GradScaler` and `autocast` to train in FP16 where possible. This nearly doubles speed on modern GPUs with minimal accuracy impact.

### 7b. Shared Training Engine — `src/experiments/train.py`

This provides reusable `train_epoch()`, `evaluate()`, and `train_model()` functions used by multiple scripts.

#### `train_epoch()`

```python
for batch_idx, batch_data in enumerate(loader):
    # 1. Move data to GPU
    images = images.to(device, non_blocking=True)
    
    # 2. Forward pass with mixed precision
    with autocast(enabled=(device=="cuda")):
        outputs = model(images, ...)
        losses = loss_fn(outputs, labels, ...)
        loss = losses["total"] / gradient_accumulation_steps
    
    # 3. Skip if NaN/Inf
    if not torch.isfinite(loss):
        skipped_batches += 1
        continue
    
    # 4. Backward pass
    scaler.scale(loss).backward()
    
    # 5. Gradient accumulation — only step every N batches
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)  # Prevent explosion
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
```

**Gradient accumulation:** If batch_size=64 and accumulation_steps=2, the effective batch size is 128. This lets you train with larger effective batches on limited GPU memory.

**Gradient clipping at 1.0:** Prevents exploding gradients that could destabilize training.

#### `evaluate()`

Runs the model in inference mode (`@torch.inference_mode()`) on the validation/test set:
1. Collects all predictions and labels
2. Computes AUC, accuracy, reasoning-specific metrics, gating alpha, convergence delta
3. Returns everything in a dict

### 7c. Config-Driven Experiment Runner — `src/experiments/run.py`

An alternative to `2_train_model.py` that reads all settings from a YAML file:

```bash
python src/experiments/run.py --config src/experiments/configs/radar.yaml
```

Follows the same flow but uses the shared `train_model()` from `train.py` instead of defining its own loop.

---

## 8. Ablation Studies

### The Approach: Baseline vs Baseline + RADAR

The ablation study answers the question: **"Do RADAR's modules (BADM + AADM + ERM) actually improve detection, or is the backbone doing all the work?"**

For each of 4 backbone architectures, we run two experiments:
1. **Baseline** — The backbone with a simple linear classification head (`num_classes=1`)
2. **Baseline + RADAR** — The same backbone wrapped with BADM, AADM, and ERM

Both are trained for the same number of epochs (default 15) on the same data. If RADAR consistently improves AUC across **all** backbones, that's strong evidence that the modules provide real value — not just overfitting to one architecture.

### `3_train_ablation.py`

```bash
# Run all 4 backbones (ViT, ResNet, Xception, EfficientNet)
python 3_train_ablation.py --data_dir ./data

# Run a single backbone
python 3_train_ablation.py --data_dir ./data --backbone resnet50

# Train from ZIP directly, 15 epochs
python 3_train_ablation.py --data_dir ./data/radar-deepfake-frames.zip --num_epochs 15
```

**The 4 backbones tested:**
| `timm` Name | Display Name | Why Included |
|-------------|--------------|-------------|
| `vit_small_patch16_224` | ViT-Small | Transformer-based, RADAR's default backbone |
| `resnet50` | ResNet-50 | Classic CNN, widely used baseline |
| `xception` | Xception | Originally designed for deepfake detection |
| `tf_efficientnet_b0` | EfficientNet-B0 | Efficiency-focused, smallest model |

**How `train_baseline()` works:**
1. Creates a `timm` model with `num_classes=1` (adds a linear head automatically)
2. Trains with `BCEWithLogitsLoss` (standard binary classification)
3. Uses AdamW + OneCycleLR scheduler
4. Tracks best validation AUC and saves that checkpoint

**How `train_radar()` works:**
1. Creates `RADAR(RADARConfig(backbone=backbone_name))` — wraps the same backbone with all RADAR modules
2. Trains with the full `RADARLoss` (main + branch + orthogonality + deep supervision)
3. Freezes backbone for the first 3 epochs (same as main training)
4. Uses differential learning rate (backbone at 10× lower LR)
5. Tracks best validation AUC and saves that checkpoint

**Result storage:** After each backbone completes, results are incrementally saved to `outputs/ablations/ablation_results.json`. This means if training crashes mid-way (e.g., during backbone 3 of 4), you don't lose the first two results.

The JSON contains, for each backbone:
- `display_name` — Human-readable backbone name
- `baseline` — `{test_auc, test_acc, best_val_auc, params, train_time_sec}`
- `radar` — Same metrics for the RADAR-enhanced version

### `run_ablations.sh`

One-command runner:
```bash
./run_ablations.sh ./data 15
```

1. Runs `3_train_ablation.py` with the given data dir and epoch count
2. Then runs `4_compare_ablations.py` to print formatted results tables

---

## 9. Results Comparison — `4_compare_ablations.py`

Reads `ablation_results.json` from the ablations output directory and generates 5 formatted outputs:

1. **Main comparison table (Table 1):**
   ```
   Backbone             Baseline AUC    + RADAR AUC      Δ AUC   Baseline Acc    + RADAR Acc
   ViT-Small                  91.20%          94.50%     +3.30%         85.40%          89.20%
   ResNet-50                  89.80%          93.10%     +3.30%         83.60%          87.80%
   Xception                   90.50%          93.80%     +3.30%         84.20%          88.50%
   EfficientNet-B0            87.30%          91.60%     +4.30%         81.10%          85.90%
   ```

2. **Parameter count table (Table 2):** Shows total params for baseline vs RADAR, overhead count and percentage.

3. **Training time table (Table 3):** Wall-clock seconds for baseline vs RADAR.

4. **Key findings (auto-computed):**
   - Average AUC improvement across all backbones
   - Best and smallest improvements
   - Whether RADAR improved **all** backbones (✓) or missed some (⚠)

5. **LaTeX table:** Ready to paste into a paper, with `\textbf` on RADAR scores.

---

## 10. Attention Visualization — `5_visualize_attention.py`

Generates plots showing what RADAR's reasoning module is doing internally.

### `visualize_attention_evolution()`

For individual samples:
- Plots how BADM vs AADM attention weights change across 3 reasoning iterations
- Shows gating alpha (reasoning vs external path weight)
- Labels with ground truth and prediction

**Example insight:** For a deepfake with obvious boundary artifacts, you might see BADM attention start high and stay high across iterations. For a deepfake with subtle compression artifacts, AADM attention might increase over iterations as the model "discovers" the evidence.

### `visualize_attention_statistics()`

Aggregate statistics across the test set:
- Histogram of final BADM attention (split by real vs fake)
- Histogram of final AADM attention (split by real vs fake)
- Histogram of gating alpha (split by real vs fake)
- Scatter plot of BADM vs AADM attention

**What to look for:**
- If real and fake distributions are separated → the model uses this evidence well
- If they overlap completely → that module isn't contributing

---

## 11. Baseline Models (`src/baselines/`)

Four comparison models, all following the same interface:

```python
def forward(self, x, freq_cached=None, sobel_cached=None) -> {"logit": ..., "prob": ...}
```

| Model | Source | Params | Notes |
|-------|--------|--------|-------|
| `ViTBaseline` | `timm` | ~22M | Same ViT-Small backbone as RADAR, but with a simple classification head. This is the "what if we just used ViT alone?" baseline. |
| `ResNetBaseline` | `torchvision` | ~25M | Classic CNN. Popular in computer vision. Uses ImageNet-pretrained ResNet-50. |
| `XceptionBaseline` | `timm` | ~23M | Originally designed for deepfake detection by Rossler et al. Uses depthwise separable convolutions. |
| `EfficientNetBaseline` | `timm` | ~5M | Efficiency-focused architecture. Smallest model. |

All accept (and ignore) `freq_cached` and `sobel_cached` parameters so they have the same call signature as RADAR — this makes it easy to swap them in the same evaluation pipeline.

**Dual role in ablation studies:** These same backbone architectures are also used **inside RADAR** via `RADARConfig(backbone=...)`. The ablation study in `3_train_ablation.py` compares each baseline model (standalone) against the same backbone wrapped with RADAR modules, demonstrating that BADM + AADM + ERM improve any backbone — not just ViT.

---

## 12. Analysis Utilities (`src/analysis/`)

### `stats.py`

**`bootstrap_auc_ci(labels, probs, n_bootstrap=2000)`:**
1. Resample the test set with replacement 2000 times
2. Compute AUC on each resample
3. Return the 2.5th and 97.5th percentile as a 95% confidence interval

**Why?** AUC is a single number. The confidence interval tells you "the true AUC is between X and Y with 95% confidence." Essential for papers.

**`delong_test(labels, probs1, probs2)`:**
A statistical test to determine if two models' AUCs are significantly different. Returns a z-statistic and p-value. If p < 0.05, the difference is statistically significant.

### `aggregate.py`

For running experiments with multiple random seeds:
- `load_experiment_results()` — loads metrics.json + config.yaml from one run
- `aggregate_multi_seed_results()` — computes mean, std, min, max across seeds
- `create_results_table()` — formats into a pandas DataFrame

### `plots.py`

Three visualization functions:
- **`plot_training_curves()`** — Loss and AUC over epochs for multiple experiments
- **`plot_roc_curves()`** — Receiver Operating Characteristic curves (TPR vs FPR)
- **`plot_ablation_results()`** — Bar chart of AUC for each ablation configuration

All save to PNG at 300 DPI (publication quality).

---

## 13. Utility Helpers (`src/utils/`)

### `logging.py`

Simple pretty-printing functions for consistent console output:
- `print_section("title")` — prints a `====` bordered section header
- `print_subsection("title")` — prints a `----` bordered sub-header
- `print_result({"AUC": 0.95, ...})` — formats key-value pairs nicely
- `print_complete("Done!", summary)` — prints a completion message with ✓

---

## 14. Experiment Configs (`src/experiments/configs/`)

YAML files for different experimental setups:

### `radar.yaml` — Default training config
- Evidence dim: 64 (with notes on sensitivity analysis: 32, 64, 128)
- 50 epochs, batch 128, LR 0.001
- Gradient accumulation: 4 steps (effective batch = 512)

### `cross_domain.yaml` — Generalization test
- Train on WildDeepfake, test on FaceForensics++
- Tests if the model transfers across different forgery methods
- Lower LR (0.0005), stronger label smoothing (0.1)

### `wilddeepfake.yaml` — Single dataset
- Train and test on WildDeepfake only
- Smaller batch (64), gradient accumulation for effective 128

---

## 15. How Everything Connects — End-to-End Flow

Here's the complete journey from "I have images" to "I have results":

```
Step 1: Download data
  ./0_download_combined_dataset.sh
  → Downloads radar-deepfake-frames.zip to ./data/

Step 2: (Optional) Extract
  python 1_fast_extract.py
  → Creates ./data/train/real/, ./data/train/fake/, etc.

Step 3: Train
  python 2_train_model.py --data_dir ./data
  │
  ├─ splits.py detects pre-split structure
  ├─ dataset.py creates DeepfakeDataset with augmentations
  ├─ DataLoader wraps it with batching + parallel loading
  │
  ├─ RADAR model created:
  │   ├─ ViT-Small loaded from timm (pretrained, frozen)
  │   ├─ BADM initialized (random weights)
  │   ├─ AADM initialized (random weights)
  │   └─ ERM initialized (random weights)
  │
  ├─ Training loop (30 epochs):
  │   │
  │   ├─ Epochs 1-3 (backbone frozen):
  │   │   For each batch:
  │   │     Image → ViT features (no gradient)
  │   │     → BADM processes edges + patches → 128-dim evidence
  │   │     → AADM processes FFT + CLS → 128-dim evidence
  │   │     → ERM refines evidence (3 iterations) → reasoning logit
  │   │     → External MLP → external logit
  │   │     → Gate: α·reasoning + (1-α)·external → final logit
  │   │     → Loss: main + branch + orthogonal + deep_supervision
  │   │     → Backward through BADM/AADM/ERM only
  │   │
  │   ├─ Epoch 4+: (backbone unfrozen, LR×0.1)
  │   │   Same as above but gradients flow into ViT too
  │   │
  │   └─ Each epoch: check val AUC → save if best → early stop if stuck
  │
  └─ Final: load best checkpoint → evaluate on test set → save metrics.json

Step 4: Ablations
  ./run_ablations.sh ./data 15
  │
  ├─ For each backbone (ViT, ResNet, Xception, EfficientNet):
  │   ├─ Train baseline (backbone + linear head) for 15 epochs
  │   ├─ Train RADAR (backbone + BADM + AADM + ERM) for 15 epochs
  │   └─ Evaluate both on test set → save to ablation_results.json
  │
  └─ Results saved incrementally (crash-safe)

Step 5: Compare
  python 4_compare_ablations.py
  → Reads ablation_results.json → prints comparison tables + LaTeX
  → Shows Δ AUC for each backbone → confirms RADAR helps all of them

Step 6: Visualize
  python 5_visualize_attention.py --checkpoint outputs/radar_combined/best.pth
  → Generates attention evolution plots + statistics
```

---

## Key Concepts to Remember

1. **The model has two "paths" to a final answer**: The reasoning path (ERM doing iterative cross-attention) and the external path (simple MLP). The gating parameter α learns which to trust more.

2. **Three iterations in ERM** is the default. Each iteration attends to both evidence sources, updates its hidden state, and makes a prediction. Later iterations should be more refined.

3. **Orthogonality loss is crucial** — without it, BADM and AADM learn the same features and become redundant. With it, they're forced to specialize.

4. **RADAR is backbone-agnostic** — thanks to `_extract_features()`, any `timm` model (ViT, ResNet, Xception, EfficientNet, etc.) can serve as the backbone. The ablation study proves this works across 4 very different architectures.

5. **Backbone freezing for 3 epochs** prevents the pretrained backbone from changing before the new modules have learned anything. This is a common transfer learning strategy.

6. **The entire model returns dictionaries** everywhere — model outputs, loss components, evaluation metrics. This makes it very flexible for monitoring and ablation.

7. **Deep supervision** means every ERM iteration is supervised, not just the final one. This helps train the earlier iterations to be useful instead of random.
