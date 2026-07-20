# Multimodal Satellite Image Matching (Sentinel-1 ↔ Sentinel-2)

This project implements a robust machine learning pipeline for cross-modal image matching and verification between **Sentinel-1 (SAR / Radar)** and **Sentinel-2 (Optical / RGB)** satellite imagery. 

Due to the fundamental physical differences between active radar sensors (which capture surface roughness and geometry) and optical sensors (which capture spectral reflectance), classical pixel-to-pixel matching fails. This solution utilizes a **Siamese Neural Network architecture with Triplet Loss** to map both modalities into a shared embedding space where semantic similarity can be robustly measured.

---

## 🚀 Key Features
* **Cross-Modal Embedding Alignment:** Maps noisy SAR patches and optical RGB patches into a unified 128-dimensional metric sphere.
* **Robust Dataset Pipeline:** Dynamic folder parsing with automatic Windows/Linux path normalization and strict pair indexing.
* **Hybrid Keypoint Matching:** Combines learned global embeddings with geometric RANSAC and SIFT filtering to find stable structural landmarks across modalities.
* **Resource-Friendly Execution:** Built-in safeguards against `ZeroDivisionError` and dynamic batch sizing for stable CPU/GPU training.

---

## 📐 Solution Architecture & Mathematical Approach

### 1. Siamese Network Topology
The system utilizes a shared-weight backbone based on **ResNet-18** (with the final classification layer removed) acting as a deep feature extractor. The extracted features are passed through a projection head:
$$\text{Linear}(512 \to 256) \to \text{ReLU}() \to \text{Linear}(256 \to 128)$$

To prevent embedding collapse and stabilize gradient propagation, the output vectors are forced onto a unit hypersphere via **$L_2$ Normalization**:
$$\mathbf{e} = \frac{f(\mathbf{x})}{\Vert{}f(\mathbf{x})\Vert{}_2}$$

### 2. Metric Learning via Triplet Loss
During training, the model processes triplets consisting of an **Anchor ($\mathbf{a}$)** (Sentinel-1 SAR patch), a **Positive ($\mathbf{p}$)** (corresponding Sentinel-2 Optical patch of the same location), and a **Negative ($\mathbf{n}$)** (random optical patch from a completely different landscape).

The network minimizes the Triplet Margin Loss:
$$\mathcal{L}(a, p, n) = \max \left( d(\mathbf{e}_a, \mathbf{e}_p) - d(\mathbf{e}_a, \mathbf{e}_n) + \alpha, 0 \right)$$
* Where $d(\mathbf{x}, \mathbf{y}) = \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_2$ represents the Euclidean distance.
* The margin $\alpha$ is set to `0.3`, ensuring the model aggressively pushes dissimilar pairs apart while pulling cross-modal representations of the same location together.

---

## 📂 Project Structure

```text
Quantum_task2/
├── dataset_preparation.ipynb  # Downloads dataset via kagglehub and builds index
├── train.py                  # CLI script for model training with Triplet Loss
├── inference.py              # CLI script for testing matching on image pairs
├── demo.ipynb                # Interactive Jupyter Notebook for visualization
├── requirements.txt          # Project dependencies
└── model_weights.pth         # Saved state dictionary of the trained model