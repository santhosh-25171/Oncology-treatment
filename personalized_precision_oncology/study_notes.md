# Stage 02 Deep Learning — Data Engineering Study Notes

**Subtitle:** Personalized Precision Medicine for Oncology Treatment Optimization

---

## Cover Page

**Stage 02 Deep Learning**  
**Data Engineering — Detailed Study Notes**  
**Personalized Precision Medicine for Oncology Treatment Optimization**  

* **Stage:** Stage 02 — Deep Learning
* **Role:** Data Engineer
* **Purpose:** Study / Viva Preparation
* **Dataset type:** Synthetic research/educational dataset

---
<div style="page-break-after: always;"></div>

## SECTION 1 — STAGE 02 OVERVIEW

### What is Stage 02 Deep Learning?
Stage 02 focuses on leveraging state-of-the-art Deep Learning (DL) architectures to process highly complex, multimodal medical data. While traditional machine learning (ML) typically relies on manually engineered features from tabular datasets, deep learning is capable of automatically discovering representations and learning complex patterns directly from raw data like high-dimensional images and sequential time-series data.

### How Stage 02 Differs from Stage 01 ML
* **Stage 01 (Machine Learning):** Primarily involved classical models (like Random Forests or Gradient Boosting) operating on structured, tabular data where features (e.g., patient age, single biomarker level) were pre-computed.
* **Stage 02 (Deep Learning):** Replaces feature engineering with representation learning. It processes unstructured data—such as pixel arrays from images and raw chronological sequences—and maps them to predictive outcomes using multi-layered neural networks.

### Why Deep Learning is Being Used
The goal of personalized precision oncology requires understanding subtle morphological patterns in tumor tissues and complex dynamic trajectories in patient biomarkers. Deep Learning excels at this through specific architectures:
1. **Image Data (CNNs):** Convolutional Neural Networks (CNNs) process spatial data, extracting features like cellular atypia, tissue inflammation, and necrosis from histopathology-style image tiles.
2. **Temporal Data (LSTMs / Transformers):** Long Short-Term Memory networks (LSTMs) or Transformers process sequential longitudinal data (e.g., biomarker changes over time), capturing the temporal dependencies and discovering how a patient’s disease evolves.

### How Data Engineering Supports These Models
Deep Learning models are data-hungry and sensitive to poor-quality inputs. Data Engineering ensures that the right data reaches the model in the correct format, completely free of leaks or corruption. 

#### Overall Pipeline Flow:
```text
DATA
  ↓
DATA ENGINEERING
  ↓
EDA (Exploratory Data Analysis)
  ↓
PREPROCESSING
  ↓
MODEL
  ↓
TRAINING
  ↓
EVALUATION
  ↓
INTEGRATION
```

---
<div style="page-break-after: always;"></div>

## SECTION 2 — WHAT IS DATA ENGINEERING?

### Definition
**Data Engineering** is the process of collecting, generating, cleaning, organizing, labeling, validating, splitting, and preparing data so that machine-learning or deep-learning models can use it reliably.

### Responsibilities of the Data Engineer

1. **Data Collection/Generation**
   * **What it means:** Gathering or procedurally generating the raw data.
   * **Why it is needed:** Without data, models cannot learn.
   * **What was done:** Procedurally generated synthetic pathology-style image tiles and longitudinal biomarker sequences for 2,000 patients.
2. **Data Cleaning**
   * **What it means:** Identifying and correcting errors, such as missing values or invalid formats.
   * **Why it is needed:** “Bad data → Bad model.” Dirty data leads to incorrect learning.
   * **What was done:** Filtered out missing images, handled corrupt data, and ensured temporal consistency.
3. **Data Preprocessing**
   * **What it means:** Standardizing formats (e.g., 224×224 resolution, RGB).
   * **Why it is needed:** Neural networks require fixed-size, standardized tensor inputs.
   * **What was done:** Saved all images precisely as 224×224 RGB files.
4. **Data Labeling**
   * **What it means:** Assigning the correct target (e.g., "malignant" or "responder").
   * **Why it is needed:** Supervised learning models need a ground truth to calculate error and adjust weights.
   * **What was done:** Assigned tissue classes to images and 90-day progression targets to sequences.
5. **Data Augmentation/Variation**
   * **What it means:** Creating diversity within the dataset to simulate real-world conditions.
   * **Why it is needed:** Prevents the model from memorizing specific artifacts and helps it generalize.
   * **What was done:** Introduced intra-class variations, stain variations, and cellular pattern diversities.
6. **Dataset Organization**
   * **What it means:** Structuring files cleanly so scripts can load them easily.
   * **Why it is needed:** Disorganized data leads to buggy dataloaders and pipeline crashes.
   * **What was done:** Organized files into `spatial` and `temporal` directories with accompanying metadata CSVs.
7. **Train/Validation/Test Splitting**
   * **What it means:** Dividing the dataset into subsets for learning, tuning, and final evaluation.
   * **Why it is needed:** To evaluate how the model performs on unseen data.
   * **What was done:** Performed a strict patient-level split (1400/300/300).
8. **Data Quality Validation**
   * **What it means:** Writing scripts to automatically verify the dataset’s integrity.
   * **Why it is needed:** Manual checking is impossible for large datasets. 
   * **What was done:** Generated a `data_quality_report.csv` checking image counts, missing values, and file paths.
9. **Leakage Detection**
   * **What it means:** Ensuring no future or test data accidentally influences the training phase.
   * **Why it is needed:** Leakage results in models that look perfect during training but fail completely in real life.
   * **What was done:** Validated that no patient IDs overlap between train/val/test splits.
10. **Documentation**
    * **What it means:** Creating a manual or data dictionary describing what every file and column means.
    * **Why it is needed:** Other engineers and data scientists need to understand the data without reverse-engineering scripts.
    * **What was done:** Created `dataset_dictionary.csv`, `README.md`, and this study document.

---
<div style="page-break-after: always;"></div>

## SECTION 3 — DATASET OVERVIEW

Below is the summary of the finalized dataset generated for this project. These metrics represent the true scale of the data engineered for Deep Learning models.

| Metric | Value |
| :--- | :--- |
| **Patients** | 2,000 |
| **Spatial image records** | 16,000 |
| **Actual image files** | 16,000 |
| **Temporal observations** | 58,979 |
| **Image resolution** | 224 × 224 |
| **Image format** | RGB |
| **Tissue classes** | 6 |
| **Prediction horizon** | 90 days |

### Patient Split Distribution
* **Training:** 1,400 patients
* **Validation:** 300 patients
* **Test:** 300 patients

### What These Numbers Mean
* **2,000 Patients:** The unique individuals tracked in this dataset.
* **16,000 Images:** Multiple pathological tissue patches exist per patient (average of 8 tiles per patient).
* **58,979 Observations:** Longitudinal tracking means each patient has multiple time steps recorded over their treatment history.
* **224×224 RGB:** Standard input size for modern image models like ResNet, VGG, or Vision Transformers.

---

## SECTION 4 — PATIENT MASTER DATA

### The Role of `patient_master.csv`
The `patient_master.csv` file serves as the central hub of truth for the entire dataset. It lists all 2,000 patients and holds their demographic and baseline clinical data. 

### Why Patient IDs are Important
In a multimodal dataset, different data modalities (images, tabular time-series) live in separate files and folders. The **Patient ID** (e.g., `P0001`) acts as the relational key that ties everything together. 

### Relationship Hierarchy
```text
Patient
  ↓
Slides (Whole Slide Images)
  ↓
Tiles (224x224 cropped regions)
  ↓
Temporal observations (Blood tests, biomarkers over time)
  ↓
Treatment (Drug cycles applied)
  ↓
Prediction targets (What happens at 90 days)
```

### Why this is Important for Multimodal DL
In later stages, a Multimodal Deep Learning architecture will attempt to combine spatial features (from CNNs) and temporal features (from LSTMs) to make a unified prediction. Without strict patient-level relationship tracking, the dataloader would mix up Patient A's tumor image with Patient B's blood tests, destroying the model's ability to learn anything coherent.

---
<div style="page-break-after: always;"></div>

## SECTION 5 — SPATIAL / IMAGE DATA ENGINEERING

The spatial component consists of histopathology-style image tiles, which act as proxy data for whole slide images (WSIs). 

### Image Specifics
* **Dimensions:** 224 × 224 pixels.
* **Format:** RGB (Red, Green, Blue color channels).

### Why Metadata Matters
Even though a CNN primarily "sees" only raw pixel arrays (the 224x224 images), the `spatial_metadata.csv` provides critical context:
* **Identifiers:** Patient ID, Slide ID, Tile ID, Image path
* **Split Information:** Whether the image belongs to train, val, or test.
* **Tissue Labels:** Ground truth diagnosis (e.g., "malignant").
* **Pathological Scores:** Cellular atypia, necrosis, inflammation, tumor probability, tumor margin status.
* **Technical Information:** Artifact information, stain information, magnification.

### Usefulness of Metadata in DL
While the CNN trains on images, the metadata is essential for:
1. **Dataloading:** Locating the physical `.jpg` files using `image_path`.
2. **Loss Calculation:** Supplying the target `tissue_class` label.
3. **Subpopulation Analysis:** Evaluating if the model performs worse on images with high "artifact flags" or unusual "stain types."

---

## SECTION 6 — SIX IMAGE CLASSES

The dataset categorizes tissues into 6 distinct classes. 

| Class | Meaning | What the CNN should learn |
| :--- | :--- | :--- |
| **1. normal** | Healthy, unaltered tissue. | Regular, uniform cellular spacing without aggressive growth. |
| **2. benign** | Non-cancerous abnormal growth. | Well-defined boundaries, lacking invasive characteristics. |
| **3. malignant** | Cancerous, invasive tissue. | High cellular atypia, dense clustering, chaotic structural patterns. |
| **4. tumor_margin** | The boundary zone between tumor and healthy tissue. | Transition patterns blending normal and cancerous cellular densities. |
| **5. necrotic** | Dead tissue, often found inside fast-growing tumors. | Lack of cellular structure, fragmented debris, specific color shifts. |
| **6. inflammatory** | Tissue showing immune response. | High density of small immune cells infiltrating the tissue area. |

> **IMPORTANT:** 
> These are **procedurally generated synthetic pathology-style images for prototype/educational use**. They are not clinically validated pathology images. They simulate the *statistical distributions* and *visual complexity* of real WSIs to allow Deep Learning pipeline development.

---
<div style="page-break-after: always;"></div>

## SECTION 7 — IMAGE GENERATION

### Procedural Generation
Because accessing 16,000 real, labeled, patient-matched whole-slide images presents massive privacy and logistical barriers, this project utilizes procedural generation using Python libraries like `NumPy` and `Pillow/PIL`. 

The generation engine mathematical simulates:
* Tissue-like background structures
* Cellular patterns and nuclei placement
* Appropriate stain variations (e.g., standard H&E, dark H&E)
* Artifact variations (e.g., blur, folding)

### Intra-Class vs. Inter-Class Variation
If we simply copy-pasted 1,000 identical images of a "malignant" tumor, the CNN would instantly overfit—memorizing that one specific pixel arrangement rather than learning what "malignancy" actually means. 

* **Intra-class variation:** Differences *within the same category*. (e.g., A malignant tumor image from Patient A looks slightly different from Patient B due to different staining intensity or cell density, but both are still malignant).
* **Inter-class variation:** Differences *between different categories*. (e.g., A normal tissue image looks structurally different from a necrotic image).

The generation script explicitly enforces intra-class variation so the CNN is forced to learn robust, generalizable features.

---

## SECTION 8 — IMAGE AUGMENTATION AND VARIATION

### Dataset Generation Variation vs. Training-Time Augmentation
It is vital to distinguish between two distinct concepts:

1. **Dataset Generation Variation (Completed):** 
   During the Data Engineering phase, we deliberately generated base images with built-in variability (different base stains, simulated blur artifacts). This guarantees the foundational dataset on disk is diverse.

2. **Training-Time Augmentation (Future Work):** 
   During the actual Deep Learning model training (in PyTorch/TensorFlow), the dataloader will apply on-the-fly random transformations to an image right before passing it to the GPU. 

### Why Augmentation is Useful
Augmentation artificially expands the size of the dataset. Even with 16,000 images, a large CNN can overfit. By randomly altering the image slightly every epoch, the network never sees the exact same image twice.

### Common Augmentation Techniques (To be applied at training time)
* **Rotation:** Randomly spinning the image (tumors have no fixed orientation).
* **Flipping:** Horizontal or vertical mirroring.
* **Cropping:** Zooming in on specific patches.
* **Brightness/Contrast variation:** Simulating different microscope lighting conditions.
* **Color/Stain Jittering:** Modifying RGB channels to simulate different laboratory chemical staining variations.

---
<div style="page-break-after: always;"></div>

## SECTION 9 — IMAGE QUALITY VALIDATION

A dataset is useless if the dataloader crashes midway through training because a file is missing. The data engineering pipeline executed strict quality assurance.

### Validation Checks Performed:

| Check Performed | Status | Details |
| :--- | :--- | :--- |
| **Image existence (16,000)** | **PASS** | Expected: 16000, Actual: 16000 |
| **Missing images** | **PASS** | 0 missing files |
| **Corrupt images** | **PASS** | All files can be opened by PIL |
| **Broken image paths** | **PASS** | CSV paths correctly map to OS files |
| **Invalid image dimensions** | **PASS** | All images are exactly 224x224 |
| **Duplicate image paths** | **PASS** | 0 duplicate paths |
| **Duplicate tile IDs** | **PASS** | 0 duplicate tile identifiers |
| **Duplicate slide IDs** | **PASS** | Slides are shared correctly across patients |
| **Patient leakage** | **PASS** | 0 patients exist in multiple splits |

These checks confirm the image dataset is 100% physically intact and structurally valid for tensor conversion.

---

## SECTION 10 — CLASS DISTRIBUTION

A critical aspect of data engineering is profiling the target labels.

### Current Class Counts
* **normal:** 5,268
* **benign:** 4,351
* **inflammatory:** 2,186
* **malignant:** 1,790
* **tumor_margin:** 1,352
* **necrotic:** 1,053

### Class Imbalance
**What it means:** Some categories appear much more frequently than others (e.g., normal vs. necrotic).
**Why it matters:** A CNN is "lazy." If 90% of the data is "normal," the network might achieve 90% accuracy simply by guessing "normal" every time, completely ignoring the minority classes.

### Minority Classes Requiring Attention
Malignant, tumor_margin, and necrotic are the most critical classes clinically, yet they have the fewest samples. 

### Planned Future Solutions (Deep Learning Phase)
* **Data Augmentation:** Heavier augmentation on minority classes.
* **Class Weights:** Modifying the loss function (e.g., Cross-Entropy with weights) to penalize the model more heavily if it misclassifies a minority class.
* **Weighted Sampling:** Forcing the dataloader to oversample minority classes during batch creation.
* **Focal Loss:** An advanced loss function that forces the model to focus on "hard-to-predict" examples.

*(Note: These solutions are planned for the modeling phase and are not yet applied to the physical dataset).*

---
<div style="page-break-after: always;"></div>

## SECTION 11 — TEMPORAL DATA ENGINEERING

While images capture a spatial snapshot, oncology requires observing how the patient changes over time. 

### Temporal Dataset Overview
* **Observations:** 58,979
* **Modality:** Longitudinal, sequence-based tabular data.

### Longitudinal Observations
Unlike static data, longitudinal data tracks the same variables recorded repeatedly over a period (e.g., every 15 days). Variables included in the `biomarker_timeseries.csv` include:
* **ctDNA level:** Circulating tumor DNA in the bloodstream.
* **ctDNA change percentage:** Trend analysis of ctDNA.
* **tumor volume:** Physical size estimation.
* **tumor growth rate:** Velocity of expansion.
* **Blood biomarkers:** CEA, CYFRA21-1, CRP, LDH.

Also integrated is the context under which these observations happen:
* **Treatment status:** Active / Follow-up
* **Treatment cycle:** Which round of therapy the patient is receiving
* **Treatment type:** The specific protocol used
* **Dose intensity:** Dosage level applied
* **Response status:** Clinical assessment at that time point

### Usage in Deep Learning
This data is specifically engineered for sequence models:
* **LSTM (Long Short-Term Memory):** Can ingest these time points chronologically, retaining an internal "memory" of how the patient's ctDNA fluctuated historically.
* **Transformers:** Can look at the entire sequence of visits simultaneously using "attention" mechanisms to discover long-range dependencies (e.g., noticing that a specific biomarker spike on Day 15 strongly correlates with relapse on Day 90).

---

## SECTION 12 — TEMPORAL SEQUENCES

### The Chronological Chain
```text
Patient P001
  ↓ Day 1  (Baseline)
  ↓ Day 15 (Cycle 1 start)
  ↓ Day 30 (Observation)
  ↓ Day 45 (Cycle 2 start)
  ↓ Day 60 (Observation)
  ↓ Future prediction
```

### Why Chronological Ordering Matters
Recurrent Neural Networks (RNN/LSTMs) ingest data sequentially. If rows in the dataset are accidentally scrambled (e.g., Day 60 appears before Day 15), the model learns a chaotic, biologically impossible trajectory.

### Protecting Against Temporal Leakage
When forming input tensors, the model must only "see" data up to the current day to predict the future. If Day 90 information accidentally slips into the Day 60 input sequence, the model is "cheating." Data Engineering strictly enforces temporal integrity.

---
<div style="page-break-after: always;"></div>

## SECTION 13 — TREATMENT TIMELINE

The `treatment_timeline.csv` encapsulates the therapeutic interventions applied to the patient.

### Key Variables:
* **Treatment status:** Is the patient currently on drug?
* **Treatment cycle:** The current therapy iteration.
* **Treatment type:** Targeted therapy vs. broad chemotherapy.
* **Dose intensity:** 100% vs reduced doses.
* **Treatment dates:** Start and stop timeframes.
* **Response status:** Ongoing clinical evaluation.

### Why Treatment Context is Essential
Biomarker trajectories do not exist in a vacuum. A sudden drop in tumor volume is excellent news if the patient is on active treatment, but a rapid rise in ctDNA is catastrophic if it happens *despite* high dose intensity. By providing treatment context to the LSTM, the model learns the *interaction* between biology and pharmacology.

---

## SECTION 14 — 90-DAY TARGETS

The ultimate goal of the Stage 02 model is prediction. 

### Prediction Horizon
**90 Days:** The model observes current and past sequences/images, and attempts to forecast the patient's state exactly 90 days into the future.

### Target Variables
Located in `progression_targets.csv`:
* `progression_90d`: Binary flag (Yes/No) will the disease progress?
* `future_tumor_volume_cm3`: Regression target for physical size.
* `future_tumor_growth_rate`: Regression target for growth speed.
* `response_category_90d`: Multiclass target for clinical outcome.

### Prediction Targets vs. Input Features
These variables are the **answers to the test**. They are carefully isolated in their own file so they are NEVER accidentally fed into the model as input data (which would cause massive target leakage).

---

## SECTION 15 — RESPONSE CATEGORIES

To provide nuanced clinical outputs, patients are grouped into response categories:

* **responder:** Tumor is shrinking, biomarkers falling.
* **stable:** Disease is halted, neither growing nor shrinking.
* **progressive:** Disease is actively advancing despite intervention.
* **mixed:** Some lesions shrink while others grow.
* **delayed_progression:** Initial stability followed by sudden failure.
* **resistant:** Complete lack of response to therapy from Day 1.

### Usefulness for Sequence Models
By classifying patients into these distinct longitudinal archetypes, the Transformer/LSTM model can be trained on a multi-class classification task, outputting a probability distribution. (e.g., "Patient A has a 70% chance of being 'resistant' based on their Day 15 ctDNA trajectory.")

---
<div style="page-break-after: always;"></div>

## SECTION 16 — TRAIN / VALIDATION / TEST SPLIT

Data must be partitioned to train and evaluate the model honestly.

* **Training Set (1,400 patients):** The data the model "studies" to learn internal parameters and weights.
* **Validation Set (300 patients):** Used during training to tune hyperparameters and check for overfitting. The model does not learn from this data directly.
* **Test Set (300 patients):** Completely locked away until the final day. Used for the final, unbiased evaluation of model accuracy.

### Why Splitting by Patient is Critical
A severe mistake in medical ML is a "random" or "image-level" split. 

**Example of Bad Splitting:**
* Patient 001's Tile 1 → Sent to Training
* Patient 001's Tile 2 → Sent to Test

**Why this is disastrous:**
If the CNN sees Tile 1 during training, it learns the unique structural artifacts, background noise, or specific stain signature of Patient 001. During testing, when it sees Tile 2, it isn't recognizing "malignancy"—it's just recognizing Patient 001! This causes massive "patient leakage" and artificial 99% accuracy.

**The Correct Approach (Implemented):**
* Patient 001 → ALL tiles and temporal data go to Training.
* Patient 002 → ALL tiles and temporal data go to Test.
This guarantees the model proves its worth on *entirely unseen human beings*.

---

## SECTION 17 — DATA LEAKAGE

**Data Leakage** occurs when information from outside the training dataset is used to create the model, allowing the model to "cheat" and perform artificially well during development, only to fail completely in production.

### Types of Leakage
* **Patient Leakage:** (Described above) A single patient's records crossing the train/test barrier.
* **Target Leakage:** The model accidentally receives a feature that implies the answer. (e.g., Using "cause of death" to predict 90-day progression).
* **Temporal Leakage:** Using Day 90 biomarker values to predict the Day 90 outcome. Time-series models must be strictly causally masked.
* **Split Leakage:** Allowing information from the Validation or Test set to influence the preprocessing phase (e.g., calculating the global mean of the entire dataset to normalize the Training data, rather than calculating the mean from the Training data alone).

> **Remember this:** Leakage creates artificially high model performance. If your Deep Learning model achieves 99.9% accuracy on a complex medical task on its first run, you almost certainly have a data leak.

---
<div style="page-break-after: always;"></div>

## SECTION 18 — DATA QUALITY

Data Engineering is essentially about building trust in the numbers. **Bad data → Bad model.**

### Quality Factors Handled
* **Missing values:** Handled or imputed to prevent NaN crashes in PyTorch tensors.
* **Duplicate records:** Removed so the model doesn't over-weight repeated events.
* **Invalid values:** E.g., ensuring no negative ages or impossible categorizations.
* **Impossible tumor volumes:** Ensuring physical constraints (a tumor cannot have negative cm³).
* **Biomarker outliers:** Handling biologically impossible readings that would skew model loss gradients.
* **Invalid treatment dates:** Ensuring chronological consistency (Treatment End cannot be before Treatment Start).
* **Missing images:** Asserting every path exists on disk.
* **Corrupt images:** Verifying headers so the dataloader doesn't throw a decoding error.
* **Invalid dimensions:** Ensuring standard 224x224 geometry.
* **Temporal inconsistencies:** Making sure Day 30 happens after Day 15.

---

## SECTION 19 — DATASET DOCUMENTATION

A well-engineered dataset requires a map. The following files provide full context:

* `README.md`: High-level overview, project goals, and folder structure.
* `dataset_dictionary.csv`: Defines exactly what every column name means and its data type.
* `data_quality_report.csv`: The automated PASS/FAIL report verifying data integrity.
* `patient_master.csv`: The core demographic linkage table.
* `train_validation_test_split.csv`: The cryptographic-style map of which patient belongs to which split.
* `spatial_metadata.csv`: Detailed labels and pathological scores for every image tile.
* `labels.csv`: Ground truth tissue classifications.
* `biomarker_timeseries.csv`: The chronological sequence of patient lab values.
* `treatment_timeline.csv`: The historical log of interventions applied.
* `progression_targets.csv`: The 90-day future outcomes for model prediction.

---
<div style="page-break-after: always;"></div>

## SECTION 20 — DATA ENGINEERING PIPELINE

The automated workflow that generated the data from nothing to a DL-ready state:

```text
PATIENT GENERATION 
  (Base demographics)
  ↓
SPATIAL DATA GENERATION 
  (Assigning slides/tiles)
  ↓
IMAGE GENERATION 
  (Drawing the 16,000 .jpg files)
  ↓
IMAGE LABELING 
  (Assigning tissue_class and atypia scores)
  ↓
TEMPORAL DATA GENERATION 
  (Simulating sequential lab values)
  ↓
TREATMENT DATA 
  (Aligning drug cycles with timelines)
  ↓
90-DAY TARGET GENERATION 
  (Calculating final prediction goals)
  ↓
PATIENT-LEVEL SPLIT 
  (1400 / 300 / 300 strict isolation)
  ↓
QUALITY VALIDATION 
  (Automated QA checks against dataset)
  ↓
LEAKAGE CHECK 
  (Verifying split isolation and temporal integrity)
  ↓
DOCUMENTATION 
  (Generating dictionaries and reports)
  ↓
HANDOFF TO DL ENGINEER 
  (Data is now ready for PyTorch/TensorFlow)
```

---

## SECTION 21 — WHAT IS COMPLETED?

### Completed
* [x] Dataset creation
* [x] Patient master
* [x] Image metadata
* [x] Image labels
* [x] 16,000 actual image files
* [x] Temporal biomarker data
* [x] Treatment timeline
* [x] 90-day targets
* [x] Patient-level split
* [x] Data quality validation
* [x] Leakage checks
* [x] Dataset dictionary
* [x] Quality report
* [x] Data Engineering documentation

### Not yet completed (Future Work)
* [ ] DL EDA (Exploratory Data Analysis at Tensor Level)
* [ ] CNN training
* [ ] LSTM/Transformer training
* [ ] Model evaluation
* [ ] Grad-CAM analysis
* [ ] Multimodal fusion
* [ ] API integration

---
<div style="page-break-after: always;"></div>

## SECTION 22 — LIMITATIONS

It is vital to state the limitations of this dataset clearly for any viva or review:

1. The pathology images are **synthetic**.
2. They are **procedurally generated** using mathematical scripting, not captured via microscope.
3. They are **not real clinical WSI data**.
4. They are **not clinically validated** by human pathologists.
5. The current spatial dataset does **not contain actual CT/MRI scans**.
6. The dataset is intended purely for **prototype / educational model development** (proving the architecture works).
7. Clinical deployment would require **real, de-identified, expert-labeled medical data** and rigorous external validation.

---
<div style="page-break-after: always;"></div>

## SECTION 23 — VIVA QUESTIONS

**1. What is Data Engineering?**
The process of generating, cleaning, organizing, and verifying data so models can learn reliably.

**2. Why is Data Engineering important in DL?**
Deep Learning is highly sensitive to dirty data. "Bad data means a bad model." Data engineering prevents crashes and leakage.

**3. Why are images 224 × 224?**
It is the standard input tensor size required by major pre-trained CNN architectures (ResNet, VGG, ViT).

**4. Why RGB?**
Neural networks process image data as matrices. RGB splits the image into three channels (Red, Green, Blue) capturing stain colors efficiently.

**5. What is augmentation?**
Modifying images during training (rotating, flipping) so the model sees new variations and avoids overfitting.

**6. What is class imbalance?**
When some labels (e.g., normal) appear much more frequently than others (e.g., necrotic), causing the model to become biased.

**7. Why is patient-level splitting required?**
To prevent patient leakage. If Tile A is in training and Tile B is in testing for the same patient, the model memorizes the patient, inflating test accuracy.

**8. What is data leakage?**
When information from outside the training set leaks into the training process, allowing the model to "cheat."

**9. What is target leakage?**
When the input data accidentally contains information about the final answer you are trying to predict.

**10. What is temporal leakage?**
When a sequence model accidentally uses future data points to predict past or present events.

**11. Why is chronological ordering important?**
LSTMs and Transformers learn from time sequences. Disordered time steps teach the model impossible biological trajectories.

**12. What is the purpose of the validation set?**
To tune model hyperparameters during training and check for overfitting without touching the final test data.

**13. What is the purpose of the test set?**
A locked vault of data used only once at the end to provide an honest evaluation of model performance.

**14. Why use CNN for image data?**
Convolutional networks are specifically designed to extract spatial features, edges, and textures from 2D grids of pixels.

**15. Why use LSTM/Transformer for temporal data?**
These architectures are designed to "remember" sequential dependencies and analyze changes across time.

**16. What is a tumor margin?**
The boundary transition zone blending healthy tissue and malignant tumor cells.

**17. Why are metadata and labels important?**
Even if the CNN only sees pixels, metadata allows us to locate files, calculate loss against ground truth labels, and perform subgroup analysis.

**18. Why validate image files?**
To guarantee zero missing files, zero corruption, and correct dimensions, preventing the dataloader from crashing mid-training.

**19. Why document the dataset?**
To provide a source of truth so downstream data scientists understand what each column and split represents.

**20. What is the limitation of synthetic medical data?**
It proves the software architecture works, but cannot be deployed clinically without retraining on real, expert-validated patient data.

---
<div style="page-break-after: always;"></div>

## SECTION 24 — QUICK REVISION PAGE

### 🔢 The Numbers
* **Patients:** 2,000
* **Images:** 16,000
* **Temporal Observations:** 58,979
* **Image Format:** 224 × 224 RGB
* **Classes:** 6 (normal, benign, malignant, tumor_margin, necrotic, inflammatory)
* **Split:** 1,400 (Train) / 300 (Validation) / 300 (Test)
* **Target:** 90-day prediction horizon

### ⚙️ Pipeline Flow
```text
DATA 
 → CLEAN 
 → LABEL 
 → AUGMENT/VARIATE 
 → SPLIT 
 → VALIDATE 
 → DOCUMENT 
 → MODEL
```

### 🧠 Most Important Concepts

* **Patient-level split:** The golden rule. A patient exists in only ONE split to avoid memorization.
* **Data leakage:** The enemy. Any "cheating" that inflates accuracy.
* **Temporal integrity:** Time must only flow forward; future data cannot predict the past.
* **Class imbalance:** Dominant classes can blind the model to rare, critical classes like necrosis.
* **Image quality:** Missing or corrupt files will crash training pipelines.
* **Synthetic data limitation:** Proves the tech, but not the clinical efficacy. Real-world deployment requires real-world data.
