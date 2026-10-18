# Faculty Demonstration Guide: Carbon Emission Analysis

**Project Title:** Carbon Emission Analysis Using Machine Learning  
**Target Duration:** 5–7 Minutes  
**Demonstration Platform:** Interactive Streamlit Web Dashboard (`http://localhost:8501`)

---

## 1. Quick Launch Commands

Open PowerShell or Command Prompt in the project folder:
```bash
# Step 1: Navigate to project root
cd C:\Users\susre\OneDrive\Documents\CARBON

# Step 2: (Optional) Activate environment if using venv
.\venv\Scripts\activate

# Step 3: Ensure dependencies are installed
pip install -r requirements.txt

# Step 4: Launch the Dashboard
streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 2. Recommended 5–7 Minute Presentation Flow

```text
[0:00 - 1:00]  Project Overview & Problem Formulation
[1:00 - 2:00]  Dataset Explorer & Data Quality
[2:00 - 3:30]  Model Comparison (Ridge vs. Random Forest) & Stoichiometric Insight
[3:30 - 4:45]  Diagnostics & Generalization Validation (Unseen-Entity Test)
[4:45 - 6:00]  Live Interactive Prediction Demo
[6:00 - 7:00]  Summary, Limitations & Faculty Q&A
```

---

## 3. Section-by-Section Walkthrough Script

### Minute 0:00 – 1:00 | Section 1: Project Overview
- **What to Show:** Navigate to **"1. Project Overview"**. Point out the 4 summary metric cards (12,551 rows, 169 years from 1854–2022, 122 parent entities, 9 commodities).
- **What to Say:**
  > *"Good morning professors. My project focuses on 'Carbon Emission Analysis Using Machine Learning'. We model historical industrial greenhouse gas emissions using 12,551 records from the Carbon Majors Database. Our objective is to evaluate whether machine learning models can accurately capture emissions from reported extraction volume, fuel commodity, governance type, and year."*
- **Highlight:** Show the Mermaid architecture diagram: 70/30 split $\rightarrow$ `ColumnTransformer` (fitted on train only) $\rightarrow$ Ridge and Random Forest pipelines.

---

### Minute 1:00 – 2:00 | Section 2: Dataset Explorer
- **What to Show:** Navigate to **"2. Dataset Explorer"**. Adjust the Year slider (e.g., 1970–2022) and deselect a commodity to show live table and chart updates.
- **What to Say:**
  > *"Here we inspect the underlying data. We see that Oil & NGL and Natural Gas represent the largest share of observations, followed by Bituminous Coal. In our data quality audit, we confirmed zero missing values, zero duplicates, and zero negative emission anomalies."*
- **Highlight:** Point out the log-log scatter plot of production volume versus emissions, highlighting how emissions scale across fuel types.

---

### Minute 2:00 – 3:30 | Section 3: Model Comparison & Scientific Rationale
- **What to Show:** Navigate to **"3. Model Comparison"**. Focus on the comparison table and metric cards.
- **Reference Numbers to State:**
  - **Ridge Baseline:** $\text{MAE} = 116.54\text{ MtCO}_2\text{e}$, $\text{RMSE} = 275.79\text{ MtCO}_2\text{e}$, $R^2 = 0.2671$.
  - **Random Forest:** $\text{MAE} = \mathbf{1.68\text{ MtCO}_2\text{e}}$, $\text{RMSE} = \mathbf{14.78\text{ MtCO}_2\text{e}}$, $R^2 = \mathbf{0.9979}$.
- **What to Say (Key Technical Explanation):**
  > *"Notice that linear Ridge achieves an $R^2$ of only 0.267, while Random Forest achieves 0.998. Why such a stark difference? In physical reality, emissions are multiplicative: $\text{Emissions} \approx \text{Production} \times \text{Factor}_{\text{Commodity}}$. A linear model without cross-product interaction terms tries to fit a single additive slope across all fuels. Random Forest, however, splits by commodity at early tree nodes and scales linearly along production, perfectly reconstructing the physical combustion laws."*

---

### Minute 3:30 – 4:45 | Section 4 & 5: Prediction Diagnostics & Generalization Validation
- **What to Show:**
  1. Briefly open **"4. Prediction Analysis"** $\rightarrow$ Show the Predicted vs. Actual plot (tight $y=x$ line for RF vs. wide dispersion in Ridge) and Feature Importance (Bituminous Coal and Production Value leading).
  2. Switch to **"5. Generalization Validation"**.
- **What to Say (Crucial Defense of High $R^2$):**
  > *"Faculty often ask: Is an $R^2$ of 0.998 due to data leakage or entity memorization? To prove it is not, we conducted a Group-Aware Split on `parent_entity`. We held out 37 entire parent organizations (34.3% of records) with strictly zero entity overlap between training and testing. On organizations the model had never seen before, Random Forest still achieves an $R^2$ of 0.9823 and MAE of 4.40 MtCO2e. This confirms the model learned universal physical combustion rules, not company identities."*

---

### Minute 4:45 – 6:00 | Section 6: Interactive Prediction Demo
- **What to Show:** Navigate to **"6. Interactive Prediction Demo"**.
- **Live Test to Run:**
  - Commodity: `Oil & NGL`
  - Production Volume: `100.0` Million bbl/yr
  - Parent Type: `Investor-owned Company`
  - Year: `2022`
  - Click **"Compute Estimated Emissions"**.
- **Result to Highlight:**
  - Estimated Emissions: $\approx \mathbf{39.90}\text{ MtCO}_2\text{e}$
  - Implied Factor: $\approx \mathbf{0.3990}\text{ MtCO}_2\text{e} / \text{Mbbl}$ (matches the physical combustion factor $\approx 0.3988$).
  - Inference speed: $< 10\text{ ms}$ using the cached pre-trained pipeline (`random_forest_pipeline.joblib`).

---

### Minute 6:00 – 7:00 | Summary & Conclusion
- **Concluding Remarks:**
  > *"In conclusion, this project illustrates the transition from an additive linear baseline to a non-linear tree ensemble that respects stoichiometric domain physics. We verified reliability through out-of-entity group testing, maintaining full leak-free preprocessing in scikit-learn pipelines."*

---

## 4. Five Likely Faculty Questions & Answers

### Q1: "Why is your Random Forest $R^2$ so high ($0.998$)? Did you overfit or leak the target?"
**Answer:**  
> *"No, there is zero target leakage. All preprocessing and encoders are fitted strictly on training folds inside a scikit-learn Pipeline. The high $R^2$ is explained by domain stoichiometry: fossil fuel emissions in the Carbon Majors database are calculated using IPCC combustion factors multiplied by production volume ($\text{Production} \times \text{Factor}_{\text{fuel}}$). The decision trees naturally split on fuel type and multiply by volume. Furthermore, when evaluated on 37 completely unseen parent entities (zero entity overlap), the model retains an $R^2$ of 0.9823, proving it has learned true combustion relationships rather than memorized data."*

### Q2: "What is the mathematical difference between MAE, RMSE, and $R^2$ here?"
**Answer:**  
> *"MAE (1.68 MtCO2e) is the average absolute error, which treats all error sizes linearly. RMSE (14.78 MtCO2e) squares errors before averaging, penalizing rare large misses on mega-producers more heavily. $R^2$ (0.9979) measures the proportion of total variance explained by the model relative to a horizontal mean baseline; it is a goodness-of-fit statistic, not an accuracy percentage."*

### Q3: "Why did you use `np.log1p` on production value if tree models are invariant to monotonic transformations?"
**Answer:**  
> *"Because we designed a unified `ColumnTransformer` shared across both models. While tree splits are invariant to monotonic transformations, linear Ridge regression is extremely sensitive to multi-order magnitude skewness (production ranges from 0.004 to 27,192). Applying `np.log1p` stabilized the variance and prevented Ridge from collapsing entirely."*

### Q4: "Can this model forecast future emissions for 2030 or 2050?"
**Answer:**  
> *"No, and we explicitly do not claim future forecasting. This model estimates emissions given reported extraction activity within historical operating regimes. Forecasting future emissions would require predicting future energy demand, policy shifts, and extraction quotas, which are outside this project's scope."*

### Q5: "What are the limitations of this model?"
**Answer:**  
> *"First, the data represents historical engineering and accounting calculations based on reported production, rather than continuous atmospheric sensor measurements. Second, tree-based models cannot extrapolate beyond the maximum production volumes seen in training. Third, historical combustion coefficients are assumed relatively stationary across reporting centuries."*

---

## 5. Technical Honesty Principles

1. **Error Metrics:** MAE and RMSE measure prediction error (lower is better, in original $\text{MtCO}_2\text{e}$ units).
2. **$R^2$ Interpretation:** $R^2$ is coefficient of determination (variance explained), **not** an "accuracy score".
3. **Stoichiometric Reality:** High performance reflects the near-deterministic relationship between fuel extraction volumes and chemical carbon release.
4. **Generalization Scope:** Group-aware validation on unseen entities confirms out-of-organization validity, but does not claim universal transfer to unrepresented fuel types or novel industrial processes.
5. **Data Source:** Historical emissions estimates from Carbon Majors, not direct satellite or smokestack sensor readings.

---

## 6. Troubleshooting If Dashboard Fails to Launch

| Issue | Cause | Immediate Resolution |
| :--- | :--- | :--- |
| `Port 8501 is already in use` | Another Streamlit instance is running | Run `streamlit run app.py --server.port 8502` |
| `ModuleNotFoundError: No module named 'streamlit'` | Active environment lacks packages | Run `pip install -r requirements.txt` |
| `Dataset not found at: data/...` | Working directory is incorrect | Ensure your terminal prompt is at `C:\Users\susre\OneDrive\Documents\CARBON` |
| Browser shows blank screen | Browser cache issue | Hard refresh using `Ctrl + F5` or open an Incognito window |
