# ROLE REPORT 3: MACHINE LEARNING & DEEP LEARNING (ML/DL)
## Personalized Precision Medicine for Oncology Treatment Optimization

```
======================================================================================================
ROLE:               Machine Learning & Deep Learning Engineer
PROJECT:            Personalized Precision Medicine for Oncology Treatment Optimization
PIPELINE STAGES:    Feature Engineering (66 Features) | Classical Ensemble Modeling | 
                    Deep Computer Vision (ResNet-18) | Longitudinal Transformers | 
                    Clinical NLP / Whisper | Small Language Model (LoRA) | Platt Calibration
CHAMPION MODELS:    Tabular Risk: Calibrated XGBoost (78.25% Recall, Brier 0.1978, t = 0.48)
                    Toxicity Risk: CatBoost (53.31% Recall, Brier 0.2089)
                    Therapy Response: Random Forest (51.69% Recall, Brier 0.2056)
                    Histopathology Vision: ResNet-18 CNN (82.05% Accuracy, ROC-AUC 0.8095)
                    Biomarker Sequences: Temporal Transformer (84.67% Accuracy, ROC-AUC 0.9272)
                    Clinical Synthesis: Qwen2.5-0.5B-Instruct + LoRA (r=8, alpha=16)
======================================================================================================
```

---

## 1. Executive Mission & Role Definition
The **Machine Learning & Deep Learning Engineer** is responsible for designing, formulating, optimizing, fine-tuning, and calibrating the predictive architectures that translate clinical data into actionable prognostic intelligence. 

In precision oncology, the ML/DL Engineer must balance discriminative accuracy with probabilistic calibration and computational feasibility, ensuring models can output reliable confidence intervals and risk strata that clinicians can trust at the bedside.

---

## 2. Mathematical Formulations of Algorithmic Architectures

### 2.1 Tabular Ensemble Algorithms

#### 1. Extreme Gradient Boosting (XGBoost)
XGBoost minimizes a regularized objective function at step $t$:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
Where the model complexity penalty is:
$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
Applying a second-order Taylor expansion around the previous prediction $\hat{y}_i^{(t-1)}$:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l\left(y_i, \hat{y}_i^{(t-1)}\right) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
Where the first- and second-order derivatives are:
$$g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$$
For a given tree structure, the optimal weight $w_j^*$ for leaf $j$ containing sample set $I_j = \{i \mid q(x_i) = j\}$ is calculated analytically by taking $\frac{\partial \mathcal{L}}{\partial w_j} = 0$:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$
The corresponding optimal objective loss is:
$$\mathcal{L}^* = -\frac{1}{2} \sum_{j=1}^T \frac{\left(\sum_{i \in I_j} g_i\right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$
The gain achieved by splitting a leaf into left ($I_L$) and right ($I_R$) children is:
$$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

#### 2. CatBoost (Categorical Boosting)
CatBoost utilizes **Ordered Boosting** to eliminate target leakage and prediction shift during gradient updates. Furthermore, it employs **Oblivious (Symmetric) Decision Trees**, where the exact same splitting criterion is applied across all nodes at the same tree depth:
- Oblivious trees provide balanced structure, prevent overfitting on noisy clinical datasets, and execute inference with $O(d)$ bitwise operations rather than dynamic branching.

#### 3. Random Forest (Bagging & Variance Reduction)
Random Forest trains $B$ unpruned decision trees, each trained on a bootstrap sample $\mathcal{D}_b$ drawn with replacement from the training set. At each split, only a random subset of features $m = \sqrt{p}$ is considered.
- The ensemble prediction is:
  $$\hat{f}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)$$
- *Variance Reduction Theorem*: For $B$ identically distributed trees each with variance $\sigma^2$ and positive pairwise correlation $\rho$, the variance of the ensemble is:
  $$\text{Var}(\hat{f}_{\text{RF}}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
  As $B \to \infty$, the second term approaches zero, while feature subsampling minimizes $\rho$, driving total variance far below that of any individual tree.

---

### 2.2 Deep Representation Learning Architectures

#### 1. Spatial Histopathology CNN (ResNet-18)
Biopsy images require learning complex spatial hierarchies without suffering gradient degradation:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
Where the identity shortcut connection passes input $\mathbf{x}$ directly across layers. Backpropagation gradients are governed by:
$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$$
Even if deep weight layers experience vanishing gradients ($\frac{\partial \mathcal{F}}{\partial \mathbf{x}} \to 0$), the identity matrix $\mathbf{I}$ ensures that error signals propagate unimpeded across all 18 layers.

#### 2. Temporal Transformer for Serial Lab Sequences
Serial biomarker sequences ($N \times T \times D$) are mapped through Multi-Head Self-Attention layers:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
Where $Q = X W^Q, K = X W^K, V = X W^V$ project inputs into $h=4$ attention heads with dimension $d_k = 64$. A sinusoidal positional encoding vector is added to incorporate clinic visit chronology:
$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

#### 3. Small Language Model (SLM) Parameter-Efficient Fine-Tuning (LoRA)
For Qwen2.5-0.5B-Instruct, the model parameters $W_0 \in \mathbb{R}^{d \times k}$ are kept frozen. Low-rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ are updated:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Where rank $r = 8$ and scaling hyperparameter $\alpha = 16$. Matrix $A$ is initialized with Gaussian distribution $\mathcal{N}(0, \sigma^2)$ and $B$ is initialized to zero, ensuring $\Delta W = 0$ at the start of training.

---

## 3. Feature Engineering Science (66 Features)
The ML Engineer engineered 66 high-dimensional features from 37 raw EHR variables:
1. **Derived Oncology Biomarkers**:
   - `treatment_intensity`: $\frac{\text{treatment\_dose}}{\text{treatment\_duration}}$
   - `high_clinical_risk`: $\mathbb{I}(\text{comorbidity} > \text{median}) \times \mathbb{I}(\text{ECOG} > \text{median})$
   - `biomarker_interaction`: $\text{biomarker\_1} \times \text{biomarker\_2}$
2. **Clinical Bins**:
   - `age_group`: Categorized into `<50`, `50-65`, `>65`.
   - `bmi_category`: WHO classification (`underweight, normal, overweight, obese`).
   - `tumor_size_category`: Discretized into TNM T-stages (`T1_small, T2_medium, T3_large`).
3. **One-Hot Encoding**: Handled via `OneHotEncoder(handle_unknown='ignore')`, expanding categorical clinical dimensions into 50 sparse columns.

---

## 4. Training Strategies, Loss Functions & Calibration

### 4.1 Loss Function Formulation
To counter the $18.3\%$ minority non-responder class, Focal Loss was formulated:
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Where $p_t$ is the model's estimated probability for the ground truth class. Setting $\gamma = 2.0$ dynamically down-weights easy examples, concentrating the gradient on difficult borderline patients.

### 4.2 Probability Calibration via Platt Scaling
Raw ensemble decision trees produce distorted probabilities because leaf proportions do not reflect true posterior probabilities. Platt Scaling fits a post-hoc logistic regression model:
$$P(y=1 \mid f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
Parameters $A$ and $B$ are estimated by minimizing negative log-likelihood on the held-out validation set ($n=375$):
$$\min_{A, B} -\sum_{i=1}^{n_{\text{val}}} \left[ t_i \log(p_i) + (1 - t_i) \log(1 - p_i) \right]$$
Where $t_i$ are smoothed target probabilities:
$$t_i = \begin{cases} \frac{N_+ + 1}{N_+ + 2} & \text{if } y_i = 1 \\ \frac{1}{N_- + 2} & \text{if } y_i = 0 \end{cases}$$

### 4.3 Optimal Decision Threshold Tuning
The standard decision threshold $t = 0.50$ is clinically suboptimal. The ML Engineer swept thresholds across $[0.20, 0.80]$ on the validation set, finding the optimal clinical operating point at $t^* = 0.48$:
- At $t = 0.50$: High-Risk Recall = $52.1\%$.
- At $t = 0.48$: High-Risk Recall = **$78.25\%$** (reducing fatal false negatives by $50.2\%$).

---

## 5. Viva Voce & Technical Defense (ML/DL Engineer)

#### Q1: Why does XGBoost use second-order Taylor expansion while standard Gradient Boosting uses only first-order gradients?
> **Answer:** "Standard gradient boosting uses only first-order gradients $g_i$ (gradient descent in function space), requiring an empirical line search to determine step sizes for each leaf. XGBoost uses second-order Taylor expansion incorporating Hessian values $h_i$. The presence of $h_i$ allows XGBoost to analytically solve for the exact optimal leaf weight $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$ and optimal split gain in closed form (Newton-Raphson step), drastically accelerating convergence and stabilizing regularization."

#### Q2: Why is Platt Scaling preferred over Isotonic Regression for smaller validation datasets?
> **Answer:** "Isotonic Regression is a non-parametric piecewise-constant approach that fits an isotonic (monotonically non-decreasing) step function. While it is more flexible, it is prone to severe overfitting on small validation datasets ($N < 1,000$). Platt Scaling assumes a parametric sigmoid relationship, requiring only two parameters ($A$ and $B$) to be fitted. This makes it far more robust to sample variance and less susceptible to overfitting on our $n=375$ validation split."

#### Q3: How does the Temporal Transformer handle irregular time intervals between oncology clinic visits?
> **Answer:** "Standard RNNs assume fixed, discrete time intervals $\Delta t$. In real-world oncology, patients may visit after 14 days, 21 days, or miss an appointment. In our Temporal Transformer, we concatenate an explicit continuous elapsed-time delta embedding $\Delta t_k = t_k - t_{k-1}$ into the input feature vector before feeding it to the self-attention layer. This allows the Query-Key attention mechanism to weight biomarker shifts proportionally to the actual physical time that has elapsed."
