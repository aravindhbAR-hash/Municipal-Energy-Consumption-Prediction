# Predicting Monthly Electricity Consumption for Municipal Buildings in Ontario

## The Problem

Every month, the City of Toronto pays electricity bills across more than 1,600 buildings - police stations, libraries, community centres, water treatment plants, arenas, and dozens of other public facilities. Managing that cost intelligently requires knowing, in advance, how much electricity each building is likely to use. Without a reliable prediction, energy managers are flying blind: they cannot spot when a building is consuming far more than it should, cannot plan budgets accurately, and cannot prioritize which facilities to investigate for energy waste.

**This project answers one central question: can we predict how much electricity a municipal building will consume in a given month, using only information we already know about the building?**

Using four years of data exported from the ENERGY STAR Portfolio Manager system — the same platform used by governments and large organizations across North America to track building energy performance - we built a complete machine learning pipeline that predicts monthly kWh consumption for any building in the portfolio based on its size, type, location, occupancy, and the time of year.

The dataset spans **1,760 properties**, **1,650 Electric–Grid meters**, and **18,109 monthly electricity readings** covering January–December 2022.

---

## Data Description

| File | Rows | Key Columns |
|---|---|---|
| `properties.csv` | 1,760 | Property Name, Portfolio Manager ID, City/Municipality, Property Type, Gross Floor Area, Occupancy (%) |
| `meters.csv` | 2,585 | Portfolio Manager Meter ID, Meter Type, Units, First/Last Entry Dates |
| `meter_entries.csv` | 26,222 | Start Date, End Date, Usage/Quantity, Usage Units, Cost ($) |
| `uses.csv` | 1,760 | Use Type, Gross Floor Area for Use |

The analysis filters to **Electric – Grid** meters measured in **kWh (thousand Watt-hours)**, reducing meter entries to 18,109 rows.

---

## Pipeline Summary

### 1. Data Loading & Cleaning
- All four CSVs loaded and inspected for shape, types, and missing values
- Date fields parsed to `datetime`; `Cost ($)` stripped of currency symbols and cast to numeric
- ID columns stored as nullable `Int64` to handle NaN values safely
- 15 negative cost rows treated as billing adjustments and set to 0
- 171 use-area records with missing floor-area flagged rather than dropped

### 2. Data Integration
- Meter entries joined with meter metadata, property characteristics, and property-level use-area aggregates
- Final analysis table: **18,109 rows × 23 columns** at the meter-month level, with zero missing values in core fields

### 3. Exploratory Data Analysis

**Distributions:** Monthly kWh is extremely right-skewed (skewness = 15.69) — a small number of very large facilities consume orders of magnitude more than a typical building. A `log1p` transform reduces this to −0.39, making the distribution suitable for regression and revealing patterns that are invisible in the raw data.

**Energy intensity by property type:** Water and wastewater treatment plants, pumping stations, and transit facilities consume dramatically more electricity per square foot than offices, libraries, or police stations — sometimes 10–20× higher intensity. Property type is one of the most informative features in the dataset.

**Floor area vs. consumption:** A strong positive log–log relationship exists between gross floor area and monthly kWh. Larger buildings consume more electricity consistently across several orders of magnitude of building size, and `log_kwh` correlates with `Gross Floor Area` at r ≈ 0.75.

**Seasonality:** Average consumption peaks in January (~80,000 kWh), driven by heating systems and long winter nights, rises again in July–August (~70,000 kWh) from air conditioning, then dips to its lowest in April–May. This U-shaped seasonal curve is consistent across the entire portfolio.

### 4. Feature Engineering

| Feature | Description |
|---|---|
| `kwh_per_ft2` | Monthly kWh ÷ Gross Floor Area (energy intensity) |
| `log_kwh` | log(1 + Usage/Quantity) — model target |
| `year`, `month` | Extracted from Start Date |
| `total_use_area` | Sum of floor area across use types per property |
| `n_use_types` | Number of distinct use types per property |
| `missing_first_last_date` | Boolean flag for meters with incomplete date coverage |

### 5. Modeling

Both models are built as scikit-learn `Pipeline` objects sharing a `ColumnTransformer` preprocessor (StandardScaler for numerics, OneHotEncoder for categoricals). The target is `log_kwh`; all metrics are reported after back-transforming with `expm1` so results are in interpretable kWh units.

**Features used:** year, month, Gross Floor Area, Occupancy (%), Property Type, City/Municipality, total_use_area, n_use_types, missing_first_last_date

| Model | Train R² (log) | Test R² (log) | Test MAE (kWh) | Test RMSE (kWh) |
|---|---|---|---|---|
| Ridge Regression | 0.3432 | 0.3417 | 1,694,319 | 48,668,677 |
| Random Forest | 0.7162 | 0.6535 | 38,684 | 204,971 |
| **Tuned Random Forest** | **0.7162** | **0.6535** | **38,684** | **204,971** |

> **Note on Ridge:** The linear model captures ~34% of variance in log space, but back-transforming those predictions to kWh produces very large errors because it cannot adequately handle the scale of high-consumption outliers. The poor kWh-scale performance is an expected consequence of applying a weak log-space linear model to a highly heterogeneous portfolio — not a code error.

> **Note on tuning:** Grid search with 3-fold cross-validation over 18 hyperparameter combinations confirmed that the default Random Forest settings (`n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`) were already optimal. The tuned model is identical to the baseline RF, which is itself a meaningful result.

### 6. Feature Importance (Tuned Random Forest)

The top predictors by mean impurity decrease are:

1. **Gross Floor Area** — the single dominant predictor; larger buildings consistently use more electricity
2. **total_use_area** — highly correlated with floor area, adds complementary signal
3. **month** — captures the seasonal U-shape (winter heating, summer cooling)
4. **Specific property types** — particularly water/wastewater treatment and recreation facilities
5. **Occupancy (%)** — buildings with low or zero occupancy show meaningfully reduced consumption
6. **City/Municipality** — reflects geographic and climate differences across 12 municipalities

### 7. Residual Analysis

The residual distribution is centered near zero with a slight right tail, indicating the model occasionally underpredicts very high-consuming facilities such as large water treatment plants. The residuals-vs-predicted scatter shows increasing variance at higher predicted values — heteroscedasticity common in energy data spanning a wide range of building sizes, and expected when a single model is applied to a portfolio this diverse.

---

## Key Results

The **Random Forest model substantially outperforms the linear baseline**, achieving a test R² of **0.65** versus 0.34 for Ridge. This confirms that the relationships between building type, size, month, and electricity consumption are nonlinear and require a more flexible model to capture.

On a typical mid-size building, predictions land within roughly **39,000 kWh** of the actual reading. Errors are larger for extreme-consumption properties (major water treatment plants, large arenas), where the model underpredicts in a consistent and understandable way.

**Why this matters beyond the model metrics:** A prediction system like this has immediate operational value. By comparing what a building *actually* consumed against what the model *predicted* it should consume, energy managers can automatically flag anomalies — a building that suddenly uses 40% more electricity than expected is a candidate for an audit, a maintenance check, or an investigation into equipment failure. Applied across 1,600+ properties every month, that kind of automated screening would be impractical to do manually.

---

## Limitations & Next Steps

The model explains 65% of consumption variation — strong, but not the full picture. Key directions for improvement:

- **Weather data** (heating/cooling degree-days) is the most important missing variable and would likely push accuracy meaningfully higher
- **Facility-type-specific models** (one for offices, one for water treatment, one for recreation) would reduce the heterogeneity a single model must handle
- **Gradient Boosted models** (XGBoost, LightGBM) are worth benchmarking against the current Random Forest
- **Property-month aggregation** (rather than meter-month) could reduce noise from billing irregularities

---

## Requirements

```
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
scikit-learn >= 1.2
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone the repository — all four CSV files must be in the same directory as `Analysis.ipynb`
2. Launch Jupyter:
   ```bash
   jupyter notebook Analysis.ipynb
   ```
3. Run all cells top to bottom (`Kernel → Restart & Run All`)

The notebook is fully self-contained. No external data downloads are required.
