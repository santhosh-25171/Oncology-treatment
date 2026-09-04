# Stage 2 Deep Learning - Data Engineering Report

## 1. Dataset Selection
We have selected two distinct datasets to support the Stage 2 Deep Learning capabilities (Convolutional Neural Networks and Sequence Models):
1. **BreastMNIST (Image Dataset)**: A publicly available biomedical image dataset representing breast ultrasound scans.
2. **Synthetic Longitudinal Oncology Dataset (Temporal Dataset)**: A simulated sequential biomarker dataset explicitly labeled as **SYNTHETIC** and **NOT CLINICALLY VALIDATED**.

## 2. Image Dataset Engineering
- **Source**: MedMNIST v2 collection.
- **Original Resolution**: 28x28 grayscale.
- **CNN Preprocessing Resolution**: 128x128.
- **IMPORTANT**: **The BreastMNIST dataset remains a low-resolution public benchmark dataset. Resizing from 28x28 to 128x128 (via bicubic interpolation) strictly improves structural CNN input compatibility but does NOT recover or invent additional medical information.**
- **Normalization**: Independent image scaling to `[0, 1]` limits. No train-set statistics leak into the validation sets via image bounds.

## 3. Synthetic Temporal Sequence Engineering
Because no suitable public longitudinal dataset met the criteria natively, temporal records were synthetically generated (`prepare_stage2_data.py`).
- **Realism Improvements**: The simulation incorporates patient-to-patient baseline variance (log-normal distribution), trajectory noise (random walks preventing monotonicity), differing rates of biomarker change, and massive trajectory overlap between responders and non-responders.
- **Missingness**: Roughly 30% overall missingness is present, originating from varied sequence lengths (padded with NaN) and explicitly injected missing time-point observations.

## 4. Synthetic Label Generation
- **Target Logic**: The `outcome` label ("Responder" vs "Non-Responder") is generated *first* for a patient using a fixed random seed (approx. 40% responders). The label dictates the *general statistical drift* of the patient's biomarkers, but massive noise and random walks ensure the target is **never directly leaked** into a single feature.
- **Predictability**: The outcome cannot be trivially predicted from a single snapshot; a temporal model (LSTM/Transformer) will be required to learn the complex trajectory patterns.

## 5. Leakage Prevention
- **Sequence Preprocessing**: `SequencePreprocessor` rigorously calculates normalization statistics (`mean`, `std`) **strictly on the TRAINING split**.
- **Patient Isolation**: The `OncologySequenceLoader` splits the raw data by `patient_id` prior to sequence building. 100% Patient Leakage Free (verified via testing).

## 6. Current Dataset Shapes
- **Images**: `(1, 128, 128)`
- **Sequences**: `(8, 3)` [8 time-steps, 3 clinical features]

## 7. Limitations
- Resizing BreastMNIST to 128x128 is solely for architectural prototyping; it remains fundamentally 28x28 information.
- The sequence dataset is a synthetic stand-in meant strictly for validating recurrent network engineering, NOT for deriving clinical insights.
