# Stage 02 Deep Learning — LSTM Temporal Modeling Report

## Purpose
Following the baseline CNN spatial analysis, the LSTM model handles the sequential, chronological dependencies of patient biomarkers and tumor volume measurements. It maps dynamic longitudinal data to predict a future state.

## Input
- **Features**: 30 normalized features including tumor volumes, multi-gene markers (ctDNA, CEA, LDH), and treatment status encodings.
- **Sequence construction**: Patient observations were ordered chronologically.

## Variable sequence lengths
Sequences vary dynamically (e.g., 25-34 observations). We deployed zero-padding alongside `pack_padded_sequence` so the LSTM strictly processes meaningful timesteps and ignores artifacts introduced by padding.

## Architecture
- Input features (30) → LSTM (2 layers, 64 hidden, batch_first, dropout=0.2)
- Final hidden representation extracted effectively corresponding to the exact sequence length (un-padded).
- Dropout (0.3) → Linear (64 -> 2)
- Predicting `progression_90d` (0 = Stable/Responsive, 1 = Progression).

## Why LSTM
LSTMs intrinsically persist hidden states across time, capturing critical rates of change (e.g., aggressive tumor growth spurts or rebounding ctDNA levels) that are otherwise lost in non-sequential algorithms.

## Training Configuration
- Optimizer: Adam (lr=0.001)
- Loss: Weighted CrossEntropyLoss to address class imbalance.
- Batch size: 32
- Best Epoch: 1

## Evaluation Results (Test Set)
- **Accuracy**: 1.0000
- **Precision**: 1.0000
- **Recall**: 1.0000
- **Macro-F1**: 1.0000
- **ROC-AUC**: 1.0000

![Confusion Matrix](../artifacts/results/lstm/confusion_matrix.png)

## Limitations
**This LSTM is a research/educational prototype trained on synthetic clinical data.** It is categorically NOT clinically validated and cannot safely diagnose or predict actual cancer progression. The simulated data may possess artifacts or simplistic patterns that an LSTM exploits, which do not translate to real human biology.
