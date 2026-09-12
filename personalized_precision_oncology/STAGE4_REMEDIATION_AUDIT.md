# STAGE 4 REMEDIATION AUDIT
## Comprehensive Repository Inventory & Code Location Analysis

**Project**: Personalized Precision Medicine for Oncology Treatment Optimization  
**Author**: Stage 4 SLM Engineer + Evaluation Engineer  
**Date**: September 2026  
**Status**: Pre-Remediation Baseline Verified (Zero Code Modified)

---

## 1. Complete Repository Structure & Asset Map

### 1.1 Stage 1 — Classical ML Clinical Risk Pipeline
- **Core Prediction Engine**: [`stage1_ml/prediction/prediction.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage1_ml/prediction/prediction.py) (`OncologyPredictionPipeline`)
  - Predicts mortality risk (`risk_probability`, `risk_category`), therapy response (`Responder` vs `Non-Responder`), and adverse toxicity probability.
- **Model Training & Artifacts**: [`stage1_ml/training/train_models.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage1_ml/training/train_models.py), [`stage1_ml/models/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage1_ml/models/) (CalibratedClassifierCV, RandomForest, scalers, encoders).
- **Feature Engineering**: [`stage1_ml/features/feature_pipeline.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage1_ml/features/feature_pipeline.py).
- **Automated Test Suite**: [`stage1_ml/tests/test_stage1.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage1_ml/tests/test_stage1.py) (3 tests passing).

### 1.2 Stage 2 — Deep Learning Multimodal Progression Pipeline
- **Core Prediction Engine**: [`stage2_dl/prediction/trajectory_predictor.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage2_dl/prediction/trajectory_predictor.py) (`MultimodalTrajectoryPredictor`)
  - Predicts longitudinal disease progression trajectory (`progression_probability`, `prediction`), fusion embedding, and histopathological classification.
- **Neural Model Architectures**:
  - ResNet CNN (Histopathology): [`stage2_dl/models/cnn/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage2_dl/models/cnn/)
  - BiLSTM (Temporal sequence): [`stage2_dl/models/lstm/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage2_dl/models/lstm/)
  - Transformer (Multi-head self-attention): [`stage2_dl/models/transformer/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage2_dl/models/transformer/)
  - Multimodal Fusion: [`stage2_dl/models/fusion/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage2_dl/models/fusion/)
- **Model Weights**: [`stage2_dl/artifacts/models/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage2_dl/artifacts/models/) (`best_fusion_model.pt`, `best_transformer_model.pt`, `best_cnn_model.pt`, `best_lstm_model.pt`).
- **Automated Test Suite**: [`stage2_dl/tests/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage2_dl/tests/) (7 files, 32 tests passing).

### 1.3 Stage 3 — Clinical NLP & Urgency Triage Pipeline
- **Core Prediction Engine**: [`stage3_nlp/models/predict.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage3_nlp/models/predict.py) (`ClinicalNLPService`)
  - Classifies clinical triage urgency (`Emergency`, `Urgent`, `Routine`) and extracts oncology entities (genes, stages, drugs).
- **Urgency & NER Models**: [`stage3_nlp/models/classification/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage3_nlp/models/classification/) (`tfidf_vectorizer.pkl`, `urgency_classifier.pkl`).
- **Clinical Preprocessing & Annotation**: [`stage3_nlp/preprocessing/clean_text.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage3_nlp/preprocessing/clean_text.py), [`stage3_nlp/annotation/annotate_ner.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage3_nlp/annotation/annotate_ner.py).
- **Automated Test Suite**: [`stage3_nlp/tests/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage3_nlp/tests/) (5 files, 27 tests passing).

### 1.4 Stage 3 Audio / Whisper Pipeline
- **Core Transcriber**: [`stage3_nlp/audio/transcriber.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage3_nlp/audio/transcriber.py) (`AudioTranscriber`)
  - Transcribes clinician audio notes via local OpenAI Whisper, outputting formatted text directly into the Stage 3 clinical cleaning and urgency classification pipeline.
- **Integration Test**: [`integration/tests/test_audio_nlp_pipeline.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/integration/tests/test_audio_nlp_pipeline.py)
  - Asserts that audio input flows through: `Audio -> Whisper -> Transcript -> Stage 3 NLP -> Urgency + NER` without creating a fragmented second NLP pipeline.

### 1.5 Stage 4 — SLM Dataset Preprocessing & Splits
- **Dataset Preprocessing Script**: [`stage4_slm/data/prepare_dataset.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/data/prepare_dataset.py)
  - Processes raw CSV, cleans narratives, builds composite multi-stage prompts, and performs patient-stratified GroupShuffleSplit (`seed=42`).
- **Verified Splits on Disk**:
  - `train.csv`: 7,896 records (2,560 patients)
  - `validation.csv`: 1,010 records (320 patients)
  - `test.csv`: 950 records (320 patients)
- **Domain Dictionary**: [`stage4_slm/domain/oncology_dictionary.json`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/domain/oncology_dictionary.json) (123 verified terms).

### 1.6 Stage 4 — Model Weights, Checkpoints, & LoRA Adapter
- **Base Architecture**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Directory**: [`stage4_slm/models/qwen2.5_0.5b/adapter/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/models/qwen2.5_0.5b/adapter/)
  - `adapter_model.safetensors` (35.2 MB)
  - `adapter_config.json` ($r=16, \alpha=32$, 7 target modules)
- **Tokenizer Directory**: [`stage4_slm/models/qwen2.5_0.5b/tokenizer/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/models/qwen2.5_0.5b/tokenizer/)

### 1.7 Integration Layer & Test Suites
- **FastAPI Service**: [`integration/api/main.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/integration/api/main.py)
  - Hosts Stage 1 (`/predict/clinical`), Stage 2 (`/predict/trajectory`), Stage 3 (`/predict/nlp`).
- **Unified Streamlit Dashboard**: [`integration/dashboard/unified_dashboard.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/integration/dashboard/unified_dashboard.py).
- **Integration Tests**: [`integration/tests/`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/integration/tests/) (46 tests passing).

---

## 2. Exact Code Locations Identified for Remediation

### 2.1 Issue 2: Training Sample Limitation

| File Path | Line Range | Function / Structure | Current Code / Limitation | Remediation Required |
| :--- | :--- | :--- | :--- | :--- |
| [`stage4_slm/training/training_config.yaml`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/training/training_config.yaml) | Lines 42, 51–52 | `training` block | `max_steps: 10`<br>`max_train_samples: 100`<br>`max_val_samples: 20` | Remove artificial sample limits; set full dataset paths (`train.csv`, `val.csv`), batch size, and gradient accumulation. |
| [`stage4_slm/training/train_slm.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/training/train_slm.py) | Lines 78–79 | `OncologySynthesisDataset.__init__` | `if max_samples is not None and max_samples > 0: self.df = self.df.head(max_samples)` | Allow `max_samples = None` to load complete 7,896 records from CSV. |
| [`stage4_slm/training/train_slm.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/training/train_slm.py) | Lines 181–189 | `Stage4SLMTrainer.__init__` | Ingests `max_train_samples = 100` and `max_val_samples = 20` | Ingest full partitions (`train_dataset` = 7,896, `val_dataset` = 1,010). |
| [`stage4_slm/training/train_slm.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/training/train_slm.py) | Lines 233–250 | `Stage4SLMTrainer.train` | Slices validation loader to only 5 samples (`val_subset = Subset(val_dataset, list(range(min(5, len(val_dataset))))`) | Evaluate validation loss across complete or representative validation cohort. |

### 2.2 Issue 3: Test Sample Limitation

| File Path | Line Range | Function / Structure | Current Code / Limitation | Remediation Required |
| :--- | :--- | :--- | :--- | :--- |
| [`stage4_slm/inference/run_inference.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/inference/run_inference.py) | Lines 152, 156 | `evaluate_test_sample` | `def evaluate_test_sample(self, num_samples: int = 10): ... sample_df = df_test.head(num_samples)` | Remove hardcoded 10-sample slice; support full 950 test records. |
| [`stage4_slm/evaluation/run_evaluation.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/evaluation/run_evaluation.py) | Lines 247–248 | `FullTestEvaluator.run_full_evaluation` | `num_eval = min(max_samples, len(self.df_test)) if max_samples else len(self.df_test)` | Defaults to `max_samples` when passed via CLI. |
| [`stage4_slm/evaluation/run_evaluation.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/evaluation/run_evaluation.py) | Lines 256–258 | `FullTestEvaluator.run_full_evaluation` | `p_id = row["patient_id"]`<br>`if p_id in evaluated_ids: continue` | **CRITICAL BUG**: Tracks evaluated samples by `patient_id` rather than row index. Because patients have multiple longitudinal encounters, rows sharing a `patient_id` are skipped. Must track by row index to evaluate all 950 rows! |

### 2.3 Issue 1: CPU Latency Measurement & Inference Generation

| File Path | Line Range | Function / Structure | Current Code / Limitation | Remediation Required |
| :--- | :--- | :--- | :--- | :--- |
| [`stage4_slm/inference/run_inference.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/inference/run_inference.py) | Lines 125–139 | `generate_summary` | Uses unprofiled `model.generate()`, non-fused adapter forward path, and standard string regex. | Implement 7-stage latency profiling; leverage `merge_and_unload()`; test thread configurations and `torch.inference_mode()`. |
| [`stage4_slm/evaluation/run_evaluation.py`](file:///c:/Users/santh/OneDrive%20-%20Rathinam%20Group%20Of%20Institutions/Desktop/Oncology%20patient%20prediction/Oncology-treatment/personalized_precision_oncology/stage4_slm/evaluation/run_evaluation.py) | Lines 106–117 | `evaluate_record` | Single timing block `t0 = time.time() ... dur = time.time() - t0` lumps prefill and decode together. | Decompose timing into prompt prefill vs autoregressive decode tokens. |

---

## 3. Preservation & Checkpoint Strategy

1. **Adapter Checkpoint Backup**: Before running any retraining script, copy `stage4_slm/models/qwen2.5_0.5b/adapter/` to `stage4_slm/models/qwen2.5_0.5b/adapter_backup_100samples/`.
2. **Upstream Pipelines Frozen**: Zero modifications to `stage1_ml/`, `stage2_dl/`, `stage3_nlp/`, and `integration/`.
3. **Data Split Preservation**: `train.csv`, `validation.csv`, and `test.csv` will remain strictly untouched.
