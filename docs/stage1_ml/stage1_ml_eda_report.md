# Stage 1 ML — Exploratory Data Analysis

## 1. Objective
EDA is performed to systematically understand the data's underlying structure, distributions, quality issues, and relationships. It guides preprocessing steps like imputation and feature engineering, ensuring models are built on valid assumptions without introducing leakage or destroying clinical nuance.

## 2. Dataset Used
This EDA analyzes the cleaned oncology dataset (`oncology_cleaned.csv`).
- **Rows**: 5000
- **Columns**: 36
- **Features**: 34
- **Targets**: 2 (`toxicity_risk`, `therapy_response`)

## 3. Dataset Characteristics
- **Numerical data**: 25 features including clinical metrics (e.g., age, BMI) and lab results.
- **Categorical data**: 9 features including historical statuses and string categories.
- **Treatment variables**: Encompasses dose and regimens.
- **Biomarker/genomic variables**: Captures molecular profiles like ctDNA levels and mutation burden.

## 4. Missing Value Analysis
- **What they are**: Observations where data was not recorded or was removed during cleaning due to being impossible.
- **Found**: 7 numerical features contain missing values.
- **Why they matter**: ML models typically cannot process `NaN` values directly.
- **Why we aren't imputing yet**: Imputation relies on aggregate statistics (e.g., mean, median). Doing this on the whole dataset before splitting causes *data leakage*, allowing the training set to "peek" at the test set's distribution. Thus, they are preserved as `NaN` for now.

## 5. Target Variable Analysis
**Toxicity Risk**:
- Low: 2025 (40.5%)
- High: 1954 (39.1%)
- Moderate: 1021 (20.4%)

**Therapy Response**:
- Partial Response: 2306 (46.1%)
- Complete Response: 1777 (35.5%)
- Non-Responder: 917 (18.3%)

**Class Imbalance**: Class imbalance occurs when some target classes significantly outnumber others. While our targets are not catastrophically imbalanced, slight imbalance exists. This matters for ML because models might become biased toward predicting the majority class to minimize overall error, ignoring rare but critical outcomes.

## 6. Numerical Analysis
Sample of numerical feature distributions (Mean | Median | Std | Min | Max):
- **age**: 56.80 | 56.80 | 12.65 | 18.00 | 85.00
- **performance_status**: 1.47 | 1.00 | 1.06 | 0.00 | 4.00
- **treatment_dose**: 75.67 | 75.33 | 18.01 | 12.17 | 140.00
- **treatment_duration**: 15.93 | 15.90 | 6.77 | 2.00 | 40.00
- **renal_function**: 69.62 | 69.20 | 18.41 | 20.00 | 125.00

**Outliers**: Outliers represent extreme numerical deviations. In oncology, extreme values (e.g., very high white blood cell count) might be clinically valid markers of severe disease rather than data errors. Thus, we do not automatically delete them.

## 7. Categorical Analysis
Categorical data distributes patients across discrete groups. Dominant categories form the baseline patient profile, while rare categories require attention so they aren't ignored during training. All missing categorical data was standardized to `unknown`.

## 8. Correlation Analysis
- **Correlation**: A statistical measure of how two numerical variables move together.
- **Positive/Negative**: Positive correlation means variables increase together; negative means one decreases as the other increases.
- **Why it matters**: Highly correlated features (multicollinearity) can destabilize linear models and make feature importance difficult to interpret, as they provide redundant information.

**Top Correlations Found**:
- tumor_size & baseline_tumor_volume: 0.92
- baseline_tumor_volume & tumor_size: 0.92
- renal_function & creatinine: -0.40
- creatinine & renal_function: -0.40
- genetic_risk_score & mutation_burden: 0.17
- mutation_burden & genetic_risk_score: 0.17
- ctDNA_level & tumor_size: 0.08
- tumor_size & ctDNA_level: 0.08
- tumor_size & inflammatory_marker: 0.08
- inflammatory_marker & tumor_size: 0.08

## 9. Feature–Target Relationships
We observed patterns between features and our targets. For instance, `age` and `tumor_size` show observable distributional shifts across different `therapy_response` and `toxicity_risk` groups.
*IMPORTANT: These are purely observed dataset relationships and do NOT definitively prove clinical causation.*

## 10. EDA Findings
1. Dataset is structurally intact with zero missing targets.
2. Moderate missingness exists in laboratory numericals.
3. Targets exhibit moderate imbalance.
4. No extreme, impossible outliers (like negative age) exist anymore due to prior cleaning.

## 11. How EDA Helps the Next ML Step
- **Feature Engineering**: Identifies skewed features needing log transformations.
- **Feature Selection**: Identifies highly correlated features that can be dropped.
- **Encoding**: Shows the cardinality of categoricals to choose between One-Hot or Ordinal encoding.
- **Imputation**: Shows distribution shapes to decide between mean or median imputation.
- **Train/test splitting**: Identifies imbalance, dictating the need for stratified splitting.
- **Model selection**: Suggests tree-based models might be preferred if data is highly non-linear or contains many outliers.

## 12. Viva Preparation

1. **What is EDA?** Exploratory Data Analysis is the process of examining datasets to summarize their main characteristics, often using statistical graphics and data visualization methods.
2. **Why is EDA necessary before ML?** It ensures data quality, reveals underlying structures, helps identify appropriate preprocessing strategies, and prevents garbage-in, garbage-out.
3. **What did you find in this dataset?** I found a clean structural base of 5000 rows with some expected missingness in lab values and moderate class distributions in the targets.
4. **How many rows and columns are present?** 5,000 rows and 36 columns.
5. **What are the target variables?** `toxicity_risk` and `therapy_response`.
6. **What is the difference between a feature and a target?** Features are the independent inputs used to make predictions, while targets are the dependent outcomes the model learns to predict.
7. **What types of features are present?** Both numerical (continuous metrics) and categorical (discrete groups) features.
8. **How did you analyze missing values?** I calculated the absolute count and percentage of missing values per column.
9. **Why shouldn't missing values always be deleted?** Deleting them can cause massive data loss and introduce bias if the missingness is not random (e.g., sicker patients missing certain tests).
10. **What is class imbalance?** A condition where the classes in the target variable are not represented equally.
11. **Is our dataset imbalanced?** Yes, slightly. For instance, the 'Low' toxicity risk class outnumbers the 'Moderate' class.
12. **Why is class imbalance a problem?** Models can become biased toward the majority class, achieving high accuracy by simply ignoring the minority class, which might be the most clinically important one.
13. **What is an outlier?** An observation that lies an abnormal distance from other values in a random sample from a population.
14. **Why shouldn't every outlier be removed?** In medicine, outliers often represent true physiological extremes (e.g., severe illness) rather than errors. Removing them prevents the model from learning how to predict severe cases.
15. **What is correlation?** A statistical relationship indicating how much two variables change together.
16. **Does correlation mean causation?** No. Two variables can be correlated due to a third unseen confounding variable, without one directly causing the other.
17. **Why did we use a correlation heatmap?** To quickly visually identify multicollinearity (highly correlated redundant features) across all numerical variables at once.
18. **What relationship did you observe between important biomarkers and targets?** Certain biomarkers like tumor size show observable distributional differences across therapy response classes.
19. **How does EDA help model building?** It dictates the exact preprocessing blueprint (scaling, encoding, imputation) required for the models to learn effectively.
20. **What will be the next step after EDA?** Data splitting, followed by feature engineering and imputation, using the insights gained here.

## 13. Short Viva Summary

### Explain my EDA in 60 seconds
"I performed an Exploratory Data Analysis on our 5,000-patient oncology dataset to understand its characteristics before modeling. I analyzed the distributions of our 34 features and 2 targets, checking for missing values, outliers, and class imbalance. I found that while the data is structurally sound, there is moderate missingness in lab variables and slight class imbalance in our targets. I generated correlation heatmaps and feature distributions, deliberately preserving missing values and clinical outliers so they can be handled appropriately in the ML pipeline without causing data leakage. This analysis gives us the exact blueprint for our upcoming feature engineering stage."

### Explain my EDA in 3 minutes
"In this stage, I conducted a comprehensive Exploratory Data Analysis on the cleaned dataset of 5,000 patients. My objective was to validate data assumptions, understand distributions, and identify challenges for the ML pipeline.

I first analyzed the structural integrity, confirming 34 features and 2 targets (`toxicity_risk` and `therapy_response`). During missing value analysis, I identified several lab features with absent data. Crucially, I did not impute them yet, as doing so before the train-test split causes data leakage; they will be handled dynamically in the ML pipeline.

I analyzed our target distributions and identified moderate class imbalance. This is clinically typical but statistically challenging, indicating we will need stratified splitting and appropriate evaluation metrics like Macro F1 rather than raw accuracy.

For numerical features, I examined the statistical spread and outliers. I actively chose not to blindly remove outliers, as extreme values in oncology often represent severe, valid clinical states rather than data entry errors. 

I also mapped feature correlations using a heatmap to identify redundant variables that could cause multicollinearity, and plotted feature-target relationships to observe baseline associations. 

Overall, this EDA confirmed the dataset is viable for modeling and provided the exact insights needed to design our robust feature engineering and cross-validation strategy, which is the immediate next step."
