# Stage 2 Deep Learning - Exploratory Data Analysis Report

## 1. Dataset Overview
This report encompasses two distinct datasets formulated for Stage 2 DL modeling:
- **BreastMNIST**: Biomedical imaging subset (780 images).
- **Synthetic Longitudinal Sequence Data**: Temporal clinical arrays (1000 patients).

## 2. Image Dataset Characteristics
- **Total Images**: 780
- **Splits**: Train: 546 | Val: 78 | Test: 156
- **Original Source Resolution**: 28x28
- **Channels**: 1
- **Pixel Bounds**: Min 1.0, Max 255.0

## 3. Image Class Distribution
- **Malignant**: 210 (26.92%)
- **Normal/Benign**: 570 (73.08%)
- **Imbalance Ratio**: 2.71:1 (Normal to Malignant)
*Finding: Significant imbalance exists. The official splits correctly inherit this skewed distribution.*

## 4. Image Resolution Limitations
Preprocessing via bicubic interpolation expands spatial dimensions from 28x28 to 128x128. **128x128 is interpolated from the original 28x28 image and does not add medical information.**

## 5. Image Pixel Analysis
- Mean: 85.01
- StdDev: 52.46
Distribution histograms are available in the figures directory.

## 6. Image Quality
- **Corrupted**: 0
- **Duplicates Detected (Exact Arrays)**: 1

## 7. Image Leakage Limitations
Patient-level leakage cannot be independently verified from available BreastMNIST metadata as patient IDs are not provided.

## 8. Temporal Dataset Overview
- **Total Patients**: 1000 (Train: 700, Val: 150, Test: 150)
- **Responder**: 402 (40.2%)
- **Non-Responder**: 598 (59.8%)
- **Duplicate Records**: 7

## 9. Sequence Length Analysis
- **Min Length**: 4
- **Max Length**: 8
- **Mean Length**: 6.01

## 10. Missingness vs Padding
- **Actual Injected Clinical Dropout**:
  - ctDNA: 610 missing points
  - Biomarker 2: 613 missing points
  - Tumor Volume: 290 missing points
- **Sequence Padding**: 1990 artificial cells added to reach the max tensor dimension of 8 time steps.

## 11. Temporal Feature Distributions
{
  "ctdna_level": {
    "mean": 641.2020206069622,
    "std": 6484.390123205565,
    "min": 0.0,
    "max": 403149.74541867536
  },
  "biomarker_2": {
    "mean": 5.327888251749741,
    "std": 2.8535150041358457,
    "min": 0.0,
    "max": 13.287088042007406
  },
  "tumor_volume": {
    "mean": 13.680842140671622,
    "std": 57.88807156764248,
    "min": 0.1,
    "max": 3574.705266692881
  }
}

## 12. Responder vs Non-Responder Trends
Visualizations show high standard-deviation overlap between classes (plotted as shaded bands). **Synthetic data behavior — not clinically validated.**

## 13. Synthetic Data Realism Assessment
The correlation magnitudes are sufficiently low to moderate, confirming no single feature acts as a perfect trivial threshold. Significant standard deviation overlap exists in trajectories.
Feature-to-Target average correlations: {"ctdna_level": -0.14158220843440295, "biomarker_2": -0.0017326820335049036, "tumor_volume": -0.22852446128619366}

## 14. Leakage Assessment
- Patient-level split isolation confirmed via data engineering validators.
- No test-set statistics utilized for global mean/std normalization logic.
- Target cleanly separated from sequential feature arrays.

## 15. Key Findings
- 4 comprehensive EDA figures generated analyzing pixel limits, trajectory variations, and dimensional interpolations.
- Data successfully scales without destructive artifacting, but resolution ceilings apply.
- Time-series variance guarantees the temporal models cannot converge trivially.

## 16. Limitations
- MedMNIST imagery resolution prevents high-level histological diagnostic mapping.
- Sequence dataset is purely synthetic.

## 17. Recommendations for CNN
- Class-weighted Cross-Entropy loss is recommended to combat 2.7:1 imbalance.
- Precision/Recall tracking over Accuracy.

## 18. Recommendations for LSTM/Transformer
- Use active padding masks when loading sequences.
