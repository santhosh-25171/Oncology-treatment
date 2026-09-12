# Stage 02 Deep Learning — Explainability & Visual Saliency Report

## Status: Completed & Verified

This report documents the explainable AI (XAI) framework developed for Stage 02 Deep Learning histopathology image analysis using Gradient-weighted Class Activation Mapping (Grad-CAM).

---

## 1. Objective of Deep Learning Explainability

In computational pathology and oncology decision support, deep learning models cannot operate as uninterpretable "black boxes." Pathologists and clinicians require visual confirmation that the convolutional neural network is basing its predictions on clinically relevant morphological features—such as cellular atypia, architectural disorganization, stromal margin infiltration, or ischemic necrosis—rather than background lighting artifacts or staining inconsistencies.

---

## 2. Grad-CAM Methodology & Architecture

### 2.1 Theoretical Formulation
Grad-CAM computes the gradients of the target class score $y^c$ with respect to the feature activation map $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y^c}{\partial A_{i,j}^k}$$

where:
- $Z = H \times W$ is the spatial area of the feature map (Global Average Pooling).
- $\alpha_k^c$ captures the importance of feature map $k$ for target class $c$.

The final class-discriminative saliency map is computed via a rectified linear combination:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

Applying **$\text{ReLU}$** ensures the heatmap only highlights features that have a positive influence on the target class score, filtering out features that contribute to other categories.

### 2.2 Target Layer Selection
- **Target Layer**: `BaselineCNN.block3.0` (final `Conv2d(64 \to 128, kernel_size=3)` before GAP).
- **Spatial Resolution**: $28 \times 28 \times 128$ feature tensor.
- **Upsampling**: Bilinear interpolation from $28 \times 28$ to the original input resolution of $224 \times 224$.

---

## 3. Visual Saliency Observations Across Tissue Classes

| Tissue Class | Expected Morphological Focus | Observed Grad-CAM Activation |
| :--- | :--- | :--- |
| **Malignant** | High nuclear-to-cytoplasmic ratio, pleomorphism, hyperchromasia | Intense focused activation over dense atypical nuclear clusters. |
| **Benign** | Well-differentiated glandular architecture, uniform nuclei | Diffuse, moderate activation across organized glandular structures. |
| **Normal** | Regular quiescent stroma and glandular lumen | Low, evenly distributed background baseline activation. |
| **Tumor Margin** | Interface boundary between invasive cells and stroma | High directional saliency along the invasive margin interface. |
| **Necrotic** | Anuclear eosinophilic debris, loss of cellular borders | Prominent localization on amorphous cellular breakdown zones. |
| **Inflammatory** | Infiltration of small, dense round mononuclear cells | Concentrated focal peaks over lymphocytic aggregate patches. |

---

## 4. Clinical Significance & Safety Protocols

1. **False Positive Prevention**: When the model flags a tile as `malignant`, the oncologist can inspect the Grad-CAM overlay to verify that the activation peak coincides with actual neoplastic cellular morphology.
2. **Quality Assurance**: If a heatmap highlights an empty background or a slide preparation artifact (stain bubble), clinicians can immediately discard the inference as unreliable.
3. **Clinical Decision Support Boundary**: Grad-CAM heatmaps are intended as a visual aid to assist pathologists and must never replace histological examination of whole-slide biopsies.

---

## 5. Generated Artifacts & Visualizations
- Implementation: `stage2_dl/src/explainability/gradcam.py`
- Execution Script: `stage2_dl/scripts/generate_gradcam.py`
- Visual Artifacts: `stage2_dl/artifacts/results/gradcam/`
  - Per-class heatmaps: `gradcam_malignant.png`, `gradcam_normal.png`, etc.
  - Multi-class summary panel: `gradcam_all_classes_summary.png`
  - Quantitative report: `gradcam_results.json`
