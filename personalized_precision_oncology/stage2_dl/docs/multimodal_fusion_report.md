# Stage 02 Deep Learning — Multimodal Fusion Report

## Objective
Multimodal fusion integrates distinct data modalities to improve predictive capability. This enhancement combines spatial (CNN) and temporal (Transformer) features into a unified patient representation.

## Why fusion?
- **CNN**: "What patterns are present in the tissue image?"
- **Transformer**: "How are the patient's biomarkers changing over time?"
- **Fusion**: "What does the patient's tissue look like AND how is the disease evolving?"

## Architecture
- Image tiles per patient passed through frozen CNN → Mean pool → 128d → Linear proj to 64d
- Temporal sequence passed through frozen Transformer → Masked mean pool → 64d
- Concatenation (128d) → ReLU → Linear → 2 classes (Progression)

## Patient alignment
`patient_id` explicitly aligned the image subsets and temporal sequences into single training instances.

## Evaluation
| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| CNN-only | 0.0000 | 0.0000 | 0.0000 |
| Transformer-only | 1.0000 | 1.0000 | 1.0000 |
| **Fusion** | **1.0000** | **1.0000** | **1.0000** |

## Limitations
- Synthetic dataset
- Synthetic histopathology-style images
- Synthetic temporal biomarker data
- No real clinical WSI
- No real CT/MRI
- No clinical validation
- Perfect or near-perfect synthetic metrics may indicate an artificially easy benchmark
- Results cannot be generalized to human patients
