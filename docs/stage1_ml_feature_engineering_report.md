# Stage 1 ML — Feature Engineering

## 1. Objective
Feature engineering transforms raw data into formats suitable for Machine Learning models while enriching it with domain-specific derived metrics.

## 2. Feature Transformations

### Original Features
The dataset began with 34 original predictor features.

### Engineered Features
We created 6 medically meaningful features:
- **age_group**
- **bmi_category**
- **tumor_size_category**
- **treatment_intensity**
- **high_clinical_risk**
- **biomarker_interaction**

### Missing Value Handling
- **Numerical**: Imputed using `Median` strategy. Median is robust to extreme clinical outliers.
- **Categorical**: Filled with `'unknown'`.
- **Data Leakage Prevention**: All imputers were fitted **exclusively on the training set**. The validation and test sets were transformed using the learned training distributions.

### Encoding Methods
- Categorical features were encoded using `OneHotEncoder`. This converts string labels into binary vectors.

### Scaling Methods
- Numerical features were standardized using `StandardScaler` (zero mean, unit variance).

### Feature Selection
- **Variance Check**: Removed 0 features with near-zero variance.
- **Correlation Check**: Removed 5 highly correlated redundant features (Pearson correlation > 0.90) to prevent multicollinearity.

## 3. Summary
- **Before**: 34 features
- **After**: 66 processed features (including One-Hot encoded categories)
