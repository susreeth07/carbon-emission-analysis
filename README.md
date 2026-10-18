# Carbon Emission Analysis Using Machine Learning

## 1. Project Objective
The objective of this project is to model and quantify historical industrial greenhouse gas emissions (`total_emissions_MtCO2e`) as a function of reported production volumes, commodity types, ownership entity classifications, and operational years from the Carbon Majors Database. 

> **Important Boundary & Scope Note:**  
> This project models historical stoichiometric relationships and accounting patterns across reported fossil fuel extractions. It does **not** claim future emissions forecasting, causal discovery, or emissions reduction/decarbonization mechanisms.

---

## 2. Dataset Description
- **Dataset:** `emissions_medium_granularity.csv` (Carbon Majors Database, preserved in `data/`).
- **Records & Dimensions:** 12,551 historical observations across 122 parent organizations spanning years 1854 to 2022.
- **Integrity Check:** 0 missing values; 0 duplicates; 0 invalid/negative emission records removed.
- **Target Variable:** `total_emissions_MtCO2e` (Continuous regression target in million metric tonnes of $\text{CO}_2$ equivalent).
- **Predictor Variables:**
  - `production_value` (Continuous numeric: extraction volume).
  - `commodity` (Categorical, 9 fuel types: Oil & NGL, Natural Gas, Bituminous Coal, Metallurgical Coal, Lignite Coal, Sub-Bituminous Coal, Thermal Coal, Anthracite Coal, Cement).
  - `parent_type` (Categorical, 3 ownership structures: Investor-owned Company, State-owned Entity, Nation State).
  - `year` (Discrete numerical: historical reporting year).

---

## 3. Data Preprocessing & Leakage Prevention
To prevent data leakage, all transformation estimators are contained in a unified scikit-learn `Pipeline` utilizing `ColumnTransformer` and are **fitted exclusively on the training partition** of any evaluation split:
1. **`FunctionTransformer(np.log1p)`:** Applied to `production_value` to compress extreme positive skewness spanning several orders of magnitude ($0.0044$ to $27,192.0$).
2. **`OneHotEncoder(handle_unknown='ignore')`:** Applied to nominal categorical variables (`commodity`, `parent_type`) to create binary dummy indicators.
3. **Numeric Passthrough:** `year` is passed through as a standardized numerical feature.
4. **Target Handling:** The target `total_emissions_MtCO2e` remains in its native $\text{MtCO}_2\text{e}$ units during regression fitting and testing. No emission-breakdown or operational sub-component variables are used.

---

## 4. Algorithms Evaluated
1. **Ridge Regression ($\alpha = 1.0$):** Linear baseline with L2 weight regularization.
2. **RandomForestRegressor ($n=200$, `random_state=42`):** Non-linear tree ensemble capable of capturing multi-way interactions without manual interaction feature construction.

---

## 5. Experimental Results Across Evaluation Strategies

### Strategy A: Standard Shuffled 70:30 Random Split (`random_state=42`)
- **Train Set:** 8,785 records (70.0%) across 122 entities.
- **Test Set:** 3,766 records (30.0%) across 122 entities.
- **Overlap:** Entities present in both sets over different historical years.

| Model | Test MAE ($\text{MtCO}_2\text{e}$) | Test RMSE ($\text{MtCO}_2\text{e}$) | Test $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | 116.5436 | 275.7941 | 0.2671 |
| **Random Forest Regressor** | **1.6841** | **14.7755** | **0.9979** |

---

### Strategy B: Group-Aware (Unseen-Entity) Split (`GroupShuffleSplit`, `random_state=42`)
- **Train Set:** 8,250 records (65.7%) across **85 entities**.
- **Test Set:** 4,301 records (34.3%) across **37 entities**.
- **Overlap:** **Strictly zero overlap (0 entities)**. Evaluates true out-of-organization generalization.

| Model | Test MAE ($\text{MtCO}_2\text{e}$) | Test RMSE ($\text{MtCO}_2\text{e}$) | Test $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | 110.0540 | 157.5395 | 0.3495 |
| **Random Forest Regressor** | **4.4026** | **25.9860** | **0.9823** |

---

## 6. Physical Stoichiometry & Feature Importance Analysis

### Emission Factor Stoichiometry
Empirical analysis of the emission-to-production ratio ($\text{Emissions} / \text{Production}$) reveals that the Carbon Majors methodology relies on fixed stoichiometric combustion emission factors for all 8 fossil fuels:
- Anthracite Coal: $\approx 2.9177\text{ MtCO}_2\text{e}/\text{Mt}$ ($\sigma \approx 10^{-15}$)
- Bituminous Coal: $\approx 2.7141\text{ MtCO}_2\text{e}/\text{Mt}$ ($\sigma \approx 10^{-15}$)
- Metallurgical Coal: $\approx 2.9666\text{ MtCO}_2\text{e}/\text{Mt}$ ($\sigma \approx 10^{-15}$)
- Natural Gas: $\approx 0.0729\text{ MtCO}_2\text{e}/\text{Bcf}$ ($\sigma \approx 10^{-16}$)
- Oil & NGL: $\approx 0.3988\text{ MtCO}_2\text{e}/\text{Mbbl}$ ($\sigma \approx 10^{-16}$)
- Cement: $\approx 0.5842\text{ MtCO}_2\text{e}/\text{Mt}$ ($\sigma \approx 0.077$, varies by clinker ratio)

Because emissions are fundamentally $\text{Production} \times \text{Factor}_{\text{Commodity}}$, Random Forest easily splits by commodity and scales linearly along production, achieving an $R^2 > 0.98$ on both evaluation regimes.

### Feature Importance & Impurity Bias
Mean Decrease in Impurity (MDI) rankings:
1. `Commodity: Bituminous Coal` (43.00%)
2. `log1p(Production Value)` (32.85%)
3. `Parent Type: Nation State` (11.04%)
4. `Commodity: Natural Gas` (8.02%)
5. `Year` (2.72%)

*Limitation Note:* MDI can be inherently biased toward continuous variables with many unique split points (e.g., `production_value`) over one-hot binary flags. Despite this bias, bituminous coal remains the dominant split due to historically massive nation-state coal volumes.

---

## 7. Project Limitations
1. **Accounting vs. Direct Physical Measurement:** The dataset is derived from reported extraction volumes multiplied by standardized emission accounting coefficients, rather than direct continuous atmospheric sensor measurements.
2. **Tree Extrapolation Constraint:** Random Forest cannot extrapolate predictions for production quantities outside the minimum/maximum bounds observed in the training split.
3. **Temporal Invariance of Coefficients:** Historical records assume relatively constant combustion emission factors per fuel unit across centuries.

---

## 8. Project Structure
```text
CARBON/
│
├── app.py                                 # Interactive Streamlit Faculty Demo Dashboard
├── data/
│   └── emissions_medium_granularity.csv   # Carbon Majors dataset
│
├── src/
│   ├── train.py                           # Model training, comparison & baseline pipeline
│   └── validate_generalization.py         # Out-of-entity group validation & factor analysis
│
├── outputs/
│   ├── models/
│   │   ├── ridge_pipeline.joblib          # Trained Ridge pipeline
│   │   └── random_forest_pipeline.joblib  # Trained Random Forest pipeline
│   ├── metrics/
│   │   ├── model_comparison.csv           # Phase 3 model comparison table
│   │   ├── evaluation_metrics.json        # Test metrics JSON for both models
│   │   └── generalization_comparison.csv  # Random vs. Group-Aware split metrics
│   └── plots/
│       ├── predicted_vs_actual.png        # Predicted vs. Actual scatter plots
│       ├── residual_plot.png              # Residual distribution plots
│       ├── rf_feature_importance.png      # Random Forest feature importance chart
│       └── generalization_comparison.png  # Random vs. Group-Aware comparison figure
│
├── requirements.txt                       # Project dependencies
└── README.md                              # Complete project documentation
```

---

## 9. How to Install and Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Faculty Demo Dashboard (Interactive Web UI):**
   ```bash
   streamlit run app.py
   ```
   *Opens in your default browser at `http://localhost:8501`.*

3. **Run baseline training & model comparison (CLI):**
   ```bash
   python src/train.py
   ```

4. **Run generalization & reliability validation (CLI):**
   ```bash
   python src/validate_generalization.py
   ```
