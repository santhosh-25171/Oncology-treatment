# Stage 02 Deep Learning — Model Architecture & Engineering Report

## Status: Completed & Verified

This report provides comprehensive technical documentation of the deep learning architectures implemented and trained in Stage 02 for Personalized Precision Oncology.

---

## 1. Spatial Vision Architecture: `BaselineCNN`

### 1.1 Objective & Clinical Task
To classify $224 \times 224$ histopathology tissue tiles into 6 diagnostic morphological categories:
1. `normal`: Healthy glandular and stromal architecture
2. `benign`: Non-invasive, hyperplastic tissue
3. `malignant`: Infiltrating neoplastic cells with nuclear atypia
4. `tumor_margin`: Border zone interfacing neoplastic and non-neoplastic tissue
5. `necrotic`: Non-viable, ischemic tissue debris
6. `inflammatory`: Dense immune infiltrate (lymphocytes/neutrophils)

### 1.2 Mathematical Formulation & Layer Details
The network consists of three hierarchical convolutional blocks followed by global average pooling and regularized classification:

$$\text{Input: } X \in \mathbb{R}^{B \times 3 \times 224 \times 224}$$

1. **Block 1 (Low-level edge & stain filters)**:
   $$\text{Conv2D}(3 \to 32, k=3, p=1) \to \text{BatchNorm2d}(32) \to \text{ReLU} \to \text{MaxPool2d}(2) \implies \mathbb{R}^{B \times 32 \times 112 \times 112}$$
2. **Block 2 (Mid-level cellular morphology & texture)**:
   $$\text{Conv2D}(32 \to 64, k=3, p=1) \to \text{BatchNorm2d}(64) \to \text{ReLU} \to \text{MaxPool2d}(2) \implies \mathbb{R}^{B \times 64 \times 56 \times 56}$$
3. **Block 3 (High-level semantic & structural tissue patterns)**:
   $$\text{Conv2D}(64 \to 128, k=3, p=1) \to \text{BatchNorm2d}(128) \to \text{ReLU} \to \text{MaxPool2d}(2) \implies \mathbb{R}^{B \times 128 \times 28 \times 28}$$
4. **Global Average Pooling (GAP)**:
   $$\text{AdaptiveAvgPool2d}(1) \implies \mathbb{R}^{B \times 128 \times 1 \times 1} \xrightarrow{\text{flatten}} \mathbb{R}^{B \times 128}$$
   *Replaces parameter-heavy dense layers, providing spatial translation invariance and drastically reducing overfitting.*
5. **Regularization & Output**:
   $$\text{Dropout}(p=0.5) \to \text{Linear}(128 \to 6) \implies \mathbb{R}^{B \times 6}$$

- **Total Trainable Parameters**: 94,470
- **Loss Function**: Weighted Cross-Entropy Loss (inversely proportional to class frequencies)
- **Optimizer**: Adam ($\eta = 10^{-3}$)

---

## 2. Temporal Sequence Architecture: `LSTMProgressionModel`

### 2.1 Objective & Clinical Task
To model longitudinal clinical biomarkers and tumor measurements across variable visit schedules to predict 90-day disease progression (`progression_90d`: 0 = Non-progression, 1 = Progression).

### 2.2 Layer Details & Sequence Packing
- **Input Dimension**: $F = 30$ continuous and one-hot encoded longitudinal features.
- **Sequence Handling**: Uses `pack_padded_sequence` with dynamic visit lengths to prevent uninformative computation over padded timesteps.
- **Recurrent Core**: 2-layer stacked LSTM ($h = 64$ hidden units, dropout $0.2$ between recurrent layers).
- **Classification Head**: Dropout($p=0.3$) on final hidden state $h_T \in \mathbb{R}^{B \times 64} \to \text{Linear}(64 \to 2)$.

---

## 3. Temporal Self-Attention Architecture: `TransformerProgressionModel`

### 3.1 Objective & Clinical Task
To capture non-adjacent temporal interactions across longitudinal patient visits via multi-head self-attention, dynamically learning which historical visits most influence 90-day progression.

### 3.2 Layer Details & Attention Mechanics
1. **Feature Projection**:
   $$\text{Linear}(30 \to 64) \implies \mathbb{R}^{B \times L \times 64}$$
2. **Sinusoidal Positional Encoding**:
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
3. **Transformer Encoder**:
   - 2 stacked `TransformerEncoderLayer` modules.
   - Multi-Head Attention: 4 heads ($d_k = 16$ per head).
   - Feedforward network: $d_{\text{ff}} = 128$ with ReLU activation and dropout $0.2$.
   - Attention padding mask: enforces that attention weights are zero for padded visit steps.
4. **Masked Mean Pooling**:
   $$\bar{h} = \frac{\sum_{t=1}^{L} h_t \cdot \mathbb{I}(\text{valid})}{\max(L_{\text{valid}}, 1)} \in \mathbb{R}^{B \times 64}$$
   *Guarantees sequence representation is invariant to arbitrary padding length.*
5. **Classifier**:
   $$\text{Dropout}(p=0.3) \to \text{Linear}(64 \to 2)$$

---

## 4. Multimodal Fusion Architecture: `MultimodalFusionModel`

### 4.1 Objective
Combines spatial cellular embeddings from biopsy images with longitudinal biomarker trajectories to form a unified multimodal patient representation.

### 4.2 Architecture Flow
1. **Image Branch**: Frozen `BaselineCNN` processes all tiles for patient $p \to$ GAP ($128\text{d}$) $\to$ Tile mean-pooling $\to \text{Linear}(128 \to 64) \implies \mathbf{e}_{\text{img}} \in \mathbb{R}^{64}$.
2. **Temporal Branch**: Frozen `TransformerProgressionModel` processes longitudinal biomarker sequence $\to$ Masked mean pooling $\to \text{Linear}(64 \to 64) \implies \mathbf{e}_{\text{temp}} \in \mathbb{R}^{64}$.
3. **Fusion Head**:
   $$\mathbf{z} = [\mathbf{e}_{\text{img}} \,\|\, \mathbf{e}_{\text{temp}}] \in \mathbb{R}^{128}$$
   $$\text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(64 \to 2)$$

---

## 5. Model Weights & Checkpoint Artifacts
- CNN Checkpoint: `artifacts/models/cnn_best.pt`
- LSTM Checkpoint: `artifacts/models/lstm_best.pt`
- Transformer Checkpoint: `artifacts/models/transformer_best.pt`
- Multimodal Fusion Checkpoint: `artifacts/models/fusion/multimodal_fusion_best.pt`
