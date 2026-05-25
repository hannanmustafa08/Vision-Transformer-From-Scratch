# Transformer & Vision Transformer — From Scratch

**Course:** CS437 Deep Learning — Spring 2026, Lahore University of Management Sciences (LUMS)

A ground-up implementation of the Transformer architecture (Vaswani et al., 2017 — 
"Attention Is All You Need") and its adaptation to computer vision as a Vision Transformer 
(ViT), applied to CIFAR-10 image classification with systematic attention mechanism experiments.

---

## 🧠 Why This Matters

Transformers have replaced RNNs as the dominant architecture for sequential data and, 
since ViT (Dosovitskiy et al., 2020), increasingly for vision too. Understanding them 
from the ground up — without library abstractions — builds the intuition needed to 
modify, debug, and extend attention-based architectures in practice.

---

## 📦 What's Inside

### Part 1 — Transformer from Scratch

Built the full original Transformer architecture component by component:

**Input Layer:**
- Token embedding layer converting discrete word indices to dense vectors
- Sinusoidal positional encoding (PE) that injects sequence-order information without 
  learnable parameters: PE(pos, 2i) = sin(pos/10000^(2i/d_model))

**Encoder Stack (N layers):**
- **Multi-Head Self-Attention:** Implemented scaled dot-product attention from scratch — 
  Q, K, V projections, scaling by √d_k, softmax normalisation, and output projection — 
  then parallelised across h=8 heads for capturing diverse relationship types
- **Position-wise Feedforward Network (FFN):** Two-layer MLP with ReLU applied 
  independently at each position
- **Residual Connections + Layer Normalisation:** Applied post-attention and post-FFN 
  for stable gradient flow through deep stacks

**Decoder Stack (N layers):**
- **Masked Multi-Head Self-Attention:** Causal masking prevents the decoder from 
  attending to future tokens during autoregressive generation
- **Cross-Attention:** Decoder queries attend to encoder keys/values, grounding 
  generation in the encoded source representation
- Final linear projection + softmax over vocabulary for next-token prediction

**Stack:** PyTorch · Custom Multi-Head Attention · Positional Encoding · Layer Norm · Residual Connections

---

### Part 2 — Vision Transformer (ViT) on CIFAR-10

Adapted the Transformer encoder to image classification via the ViT framework, 
then conducted systematic attention mechanism experiments.

**Dataset:** CIFAR-10 — 20,000-image training subset, full 10,000-image validation set  
**Training config:** 15 epochs · Adam (lr=3e-4) · CosineAnnealingLR · CrossEntropyLoss · 
Gradient clipping (max norm=1.0)

**ViT Architecture:**
- Images (32×32×3) are split into fixed-size patches (e.g., 4×4), linearly embedded 
  into a sequence of patch tokens — treating an image like a sentence of visual words
- A learnable [CLS] token aggregates global information; positional embeddings are 
  added to preserve spatial structure
- Standard Transformer encoder stack processes the token sequence
- [CLS] token representation fed to a classification head for final predictions

**Attention Mechanism Experiments:**

Systematically modified the core attention mechanism and measured downstream impact 
on CIFAR-10 classification accuracy:

- **Baseline ViT:** Standard scaled dot-product attention
- **Number of heads ablation:** Varied h ∈ {1, 2, 4, 8} — multi-head attention 
  outperforms single-head by capturing complementary feature relationships in parallel
- **Attention temperature scaling:** Modified the 1/√d_k scaling factor — sharper 
  attention (lower temperature) improves classification on cleaner data; softer 
  attention generalises better under distribution shift
- **Attention dropout:** Regularising attention weights via dropout reduces overfitting 
  on the 20K training subset, with measurable validation accuracy improvement

**Stack:** PyTorch · Vision Transformer · Patch Embeddings · Multi-Head Self-Attention · 
CIFAR-10 · CosineAnnealingLR

---

## 🛠️ Tech Stack

**Framework:** PyTorch  
**Language:** Python  
**Dataset:** CIFAR-10  
**Techniques:** Multi-Head Self-Attention · Positional Encoding · Masked Attention · 
Cross-Attention · Patch Embeddings · ViT · Attention Ablation Studies

---

## 💻 How to Run

1. Clone the repository
2. Install dependencies: `pip install torch torchvision numpy matplotlib`
3. CIFAR-10 downloads automatically via torchvision on first run
4. Run notebooks sequentially
5. A CUDA-enabled GPU is strongly recommended for the ViT training experiments

---
