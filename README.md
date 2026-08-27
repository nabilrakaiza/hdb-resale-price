# HDB Resale Price Regression Analysis

Predicting Singapore HDB resale flat prices with multiple linear regression. Coursework for **ST3131 Regression Analysis** (NUS, Semester 2 AY2025/26).

The goal was not maximum predictive accuracy. It was to arrive at the best *linear* model: one that predicts well, keeps its coefficients interpretable, avoids multicollinearity, and satisfies the assumptions that make OLS inference valid in the first place.

**Final model:** `log(resale_price) ~ town + floor_area_sqm + remaining_lease + storey_avg + dist_to_mrt`
**Adjusted R²:** 0.895 on 2,456 transactions

---

## Dataset

`hdb-resale-July-2020.csv`, a single month (July 2020) of resale transactions extracted from [data.gov.sg](https://data.gov.sg/dataset/resale-flat-prices).

| Column | Description |
|---|---|
| `month` | Month of sale (constant: July 2020) |
| `town` | Town the flat belongs to |
| `flat_type` | Type of flat (3-room, 4-room, etc.) |
| `block` | Block number |
| `street_name` | Street name of the address |
| `storey_range` | Binned storey interval (e.g. "07 TO 09") |
| `floor_area_sqm` | Floor area in square metres |
| `flat_model` | Model name of the flat |
| `lease_commence_date` | Year the lease started (HDB leases run 99 years) |
| `remaining_lease` | Remaining lease at time of sale, as a string |
| `resale_price` | Resale price in SGD (target) |

---

## Repo structure

```
.
├── data/
│   ├── hdb-resale-July-2020.csv     # raw data
│   └── hdb-geocoded.csv             # cached OneMap lookups
├── notebooks/
│   └── analysis.ipynb               # full pipeline
├── report/
│   └── ST3131_report.pdf            # 6-page statistical report
└── README.md
```

---

## How the analysis was done

### 1. Preprocessing

The raw data is not shaped for linear regression, so four transformations came first.

**Dropped constant columns.** `month` holds one unique value across every row, so it carries zero information and was removed.

**Parsed the lease string.** `remaining_lease` arrives as text like `"61 years 04 months"`. Regex extracts the year and month components and converts them into a single continuous variable measured in months, which is what the model can actually use.

**Geocoded addresses into proximity features.** `block` and `street_name` are high-cardinality categoricals. One-hot encoding them would blow up the design matrix and produce coefficients nobody can interpret. Instead, each `block + street_name` pair was sent to the **OneMap API** to retrieve latitude, longitude, and postal code. Those coordinates then became two continuous predictors:

- `dist_to_cbd`: haversine distance to the central business district
- `dist_to_mrt`: haversine distance to the nearest MRT station

This turns hundreds of address categories into two numbers that encode what the address was really proxying for all along, which is location value.

**Converted storey ranges to a numeric scale.** `storey_range` is ordinal but stored as bins. Treating it as categorical would throw away the ordering, so the midpoint of each bin was extracted into `storey_avg`.

### 2. Exploratory data analysis

Four passes over the data, each answering a different question.

**Histograms (univariate).** `resale_price` is positively skewed, which is the first hint that a log transform may be needed later. `floor_area_sqm` and `lat` look roughly normal. `dist_to_mrt` and `storey_avg` are right-skewed. `lon` is bimodal, reflecting Singapore's east and west residential clusters.

**Scatterplots (numeric predictors vs target).** `floor_area_sqm` and `storey_avg` trend positively with price. `dist_to_mrt` and `dist_to_cbd` trend negatively, which matches intuition about location premiums. More importantly, the spread of `resale_price` visibly widens across predictor levels, which is heteroscedasticity showing up before a single model has been fitted.

**Boxplots (categorical predictors vs target).** Price varies substantially across `town`, confirming location as a primary driver. `flat_type` and `flat_model` also separate clearly, with specialised models commanding premiums over standard apartments.

**Correlation check.** `lease_commence_date` and `remaining_lease` plot as an almost perfect straight line, which makes sense because remaining lease is just 99 years minus flat age. Keeping both would guarantee severe multicollinearity, so one had to go before modelling started.

### 3. Baseline model (M0)

An OLS fit with ten regressors: `town`, `flat_type`, `flat_model`, `storey_avg`, `floor_area_sqm`, `remaining_lease`, `dist_to_cbd`, `dist_to_mrt`, `lat`, `lon`.

**Fit:** R² = 0.897, adjusted R² = 0.895. Overall regression significant at p < 0.001.

Three problems surfaced immediately:

**Insignificant coordinates.** `lat` and `lon` were not statistically significant. Once `dist_to_cbd` and `dist_to_mrt` are in the model, the raw coordinates are redundant, so they were dropped.

**High VIF.** `dist_to_cbd`, `town`, `flat_type`, `flat_model`, and `floor_area_sqm` all showed inflated variance inflation factors. The causes are structural rather than accidental:
- `dist_to_cbd` is largely determined by `town`, since a town's location fixes its distance from the centre
- `flat_type` and `floor_area_sqm` measure nearly the same thing, because a 5-room flat is almost always larger than a 3-room
- certain `flat_model` designations are systematically tied to larger floor areas

**Failed residual diagnostics.** The Q-Q plot showed a heavy right tail with residuals bending upward away from the theoretical line, violating normality. The residuals-vs-fitted plot showed a combined U-shape and funnel, indicating both non-constant variance and possible non-linearity in the mean function.

### 4. Fixing the violations

**Multicollinearity, approach A: respecification.** The collinear predictors fall into two natural groups:
- Structural: `flat_type`, `flat_model`, `floor_area_sqm`
- Locational: `dist_to_cbd`, `town`

Iterative elimination kept one representative from each. `floor_area_sqm` and `town` won on the balance between R² and parsimony, and VIFs dropped to acceptable levels with negligible loss in fit.

**Multicollinearity, approach B: Ridge regression.** Fitted on the unpruned feature set for comparison, reaching R² = 0.9247. Ridge shrinks correlated coefficients rather than dropping predictors, which is more robust out of sample. It was still rejected, because the L2 penalty biases the coefficients and the whole point of this model is reading marginal effects off them. Ridge wins on prediction, OLS wins on the actual objective.

**Heteroscedasticity and normality: log transform.** Taking `log(resale_price)` as the target fixed the funnel. Residuals-vs-fitted became a random cloud, so homoscedasticity is largely satisfied. The Q-Q plot improved but retained a slight upward bend in the right tail.

**On the remaining outliers.** Trimming the upper tail would produce a prettier Q-Q plot and better normality statistics. Those points were kept anyway. They are genuine sales of ultra-premium flats, not data errors, and removing them would give the model a false sense of precision while blinding it to exactly the high-end segment that drives the top of the market.

### 5. Final model (MN)

```
log(resale_price) ~ town + remaining_lease + dist_to_mrt + floor_area_sqm + storey_avg
```

R² = 0.896, adjusted R² = 0.895, 2,456 observations, 29 model degrees of freedom.

Because the target is logged, coefficients read as approximate percentage changes:

- **`remaining_lease` (quantitative):** coefficient 0.0011. Holding everything else constant, one additional month of remaining lease is associated with roughly a **0.11% increase** in resale price.
- **`town` (categorical):** coefficient −0.1157 for Clementi, relative to the Bukit Timah baseline. Controlling for size, storey, lease, and MRT distance, a Clementi flat sells for approximately **15.05% less** than a comparable Bukit Timah unit.

Note that the final adjusted R² is identical to the baseline's despite dropping five predictors. The model got simpler, its assumptions got satisfied, and it gave up nothing in explanatory power.

---

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

The geocoding step calls the OneMap API once per unique address and is the slow part of the pipeline. Results are cached to `data/hdb-geocoded.csv`, so it only runs on a cold start.

---

## Limitations and next steps

- **Single month of data.** July 2020 only, so nothing here captures seasonality or market movement over time, and the model should not be extrapolated to other periods.
- **Missing amenity features.** Proximity to primary schools and shopping malls are well-known price drivers in Singapore and are absent here.
- **No macroeconomic context.** Interest rates and cooling measures shift the whole price surface, and a single-month cross-section cannot see them.
- **Linear-only by design.** The assignment scope was linear models. Tree ensembles would almost certainly predict better, at the cost of the interpretability this model was built for.

---

## Tech

Python, pandas, NumPy, statsmodels, scikit-learn, matplotlib, seaborn, OneMap API.

---

## Note

Coursework submitted for ST3131 at NUS. Published after the module concluded, for reference only. AI assistance was used for debugging, code generation, and editing, as declared in the submitted report.
