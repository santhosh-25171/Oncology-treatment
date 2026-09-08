# DL Engineer Module

**Role-wise Organization Layer**

To strictly adhere to the project's requirement of prioritizing a stable working pipeline over heavy refactoring (`WORKING PIPELINE > ROLE-WISE ORGANIZATION`), the actual executable python files remain safely locked within their native `src/` and `scripts/` directories to prevent breaking cross-module imports and Pytest suites.

This documentation serves as the central map and organizational layer for all Deep Learning responsibilities within Stage 02.

---

## 1. CNN (Spatial / Image Understanding)
Processes 224x224 histopathology tiles and extracts a 128-dimensional spatial embedding per image.
* **Working Implementation**: `scripts/run_baseline_cnn.py` (contains the `BaselineCNN` architecture and training loop)
* **Checkpoint**: `artifacts/models/cnn_best.pt`
* **Results & Metrics**: `artifacts/results/cnn/`
* **Performance Note**: Evaluated on synthetic image tiles; achieves perfect convergence due to synthetic benchmark simplicity.

## 2. LSTM (Temporal Sequence Modeling)
Processes variable-length longitudinal clinical/biomarker arrays via recurrent padding matrices.
* **Architecture**: `src/models/lstm.py` (`LSTMProgressionModel`)
* **Training Orchestrator**: `scripts/train_lstm.py`
* **Checkpoint**: `artifacts/models/lstm_best.pt`
* **Results & Metrics**: `artifacts/results/lstm/`

## 3. Transformer (Temporal Sequence Modeling)
Processes temporal data using multi-head self-attention and dynamic source key padding masks.
* **Architecture**: `src/models/transformer.py` (`TransformerProgressionModel`)
* **Training Orchestrator**: `scripts/train_transformer.py`
* **Checkpoint**: `artifacts/models/transformer_best.pt`
* **Results & Metrics**: `artifacts/results/transformer/`

## 4. LSTM vs Transformer (Model Comparison)
Evaluates and benchmarks both temporal networks synchronously on the exact same patient validation split.
* **Execution Script**: `scripts/compare_temporal_models.py`
* **Results & Benchmarks**: `artifacts/results/comparison/`
* **Performance Note**: Both models achieved perfect accuracy due to deterministic synthetic data logic. The Transformer yielded a moderately faster inference calculation.

## 5. Multimodal Fusion (CNN + Transformer)
Combines the 128d frozen Spatial CNN vector with the 64d frozen Temporal Transformer vector via late-stage concatenation on a strictly patient-aligned basis.
* **Architecture**: `src/models/fusion.py` (`MultimodalFusionModel`)
* **Training Orchestrator**: `scripts/train_fusion.py`
* **Dataset Logic**: `src/data/fusion_dataset.py`
* **Checkpoint**: `artifacts/models/fusion/multimodal_fusion_best.pt`
* **Results & Metrics**: `artifacts/results/fusion/`
* **Performance Note**: Fusion matched the best single modality (1.0000 Macro-F1) but did not provide measurable improvement on this purely synthetic benchmark.

---

### Important Disclaimer
The dataset powering these models is entirely synthetic and educational. Results are not clinically validated and do not translate to actual oncological workflows.
