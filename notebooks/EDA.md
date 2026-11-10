# Exploratory Data Analysis Summary

Empirical summary of `data/train.csv` and `data/test.csv` informing the cold-start validation split, feature engineering, and hierarchical interval calibration in `main.py`.

## 1. Panel Structure

- **Train**: `29,100 rows x 19 columns` (including targets `price_p05` and `price_p95`).
- **Test**: `28,050 rows x 18 columns` (including `row_id`, without targets).
- **Date key**: `dt` is present in both splits.

## 2. Data Integrity

- **Missing values**: `0%` across all columns in both train and test.
- **Duplicate `(product_id, dt)` pairs**: `0`.
- **Interval validity**: `0` rows where `price_p05 > price_p95`.
- **Minimum positive interval width**: `~0.000109` (no zero-width intervals).

## 3. Temporal Coverage & Cold-Start SKU Regime

- **Date ranges**:
  - Train: `2024-03-28 -> 2024-05-26` (`60` dates), forming a complete balanced panel of `485 SKUs x 60 dates`.
  - Test: `2024-03-28 -> 2024-06-25` (`90` dates), composed of two distinct regimes:
    - **`150` cold-start SKUs** observed across all `90` dates (`2024-03-28 -> 2024-06-25`).
    - **`485` warm SKUs** observed across the future `30` dates (`2024-05-27 -> 2024-06-25`).
  - Daily SKU count in test is either `150` (overlapping historical dates) or `635` (future forecast horizon).

## 4. Feature Diagnostics & Distribution Shift

### 4.1 Dynamic Category Hierarchy

- Unique cardinality (identical across train and test):
  - `management_group_id`: `7`
  - `first_category_id`: `29`
  - `second_category_id`: `76`
  - `third_category_id`: `197`
- `product_id`: `485` in train and `635` in test (`150` unseen SKUs).
- Category assignments are **time-varying within the same `product_id`**:
  - Share of SKUs with >1 distinct value over 60 days: `management_group_id` `85.15%`, `first_category_id` `96.91%`, `second_category_id` `100%`, `third_category_id` `100%`.
  - Mean transitions per SKU across 60 days: `management_group_id` `3.60`, `first_category_id` `5.74`, `second_category_id` `53.35`, `third_category_id` `29.46`.
  - Unique 4-level category paths: `13,491` in train vs. `13,284` in test; **`58.7%`** of test paths (`77.3%` on cold-start SKUs) never appear in train.

### 4.2 Calendar & Promo Flags

- `month`: `3-5` in train, `3-6` in test; `week_of_year`: `13-21` in train, `13-26` in test; `dow`: `0-6`; `day_of_month`: `1-31` (`0%` mismatch against `dt`).
- `holiday_flag` mean: `0.3667` (train) vs. `0.3214` (test).
- `activity_flag` mean: `0.5489` (train) vs. `0.5558` (test).

### 4.3 Standardized Context Covariates

Context features are pre-standardized at the `(product_id, dt)` level (unique values per date match the active SKU count):

- **Mean / Std (`train -> test`)**:
  - `n_stores`: `-0.0023 / 0.9899 -> 0.1701 / 1.0128` (`PSI = 1.374`; nearly constant within each SKU, median intra-SKU std `0.0359`)
  - `precpt`: `-0.1866 / 0.9618 -> 0.3610 / 1.0141` (`PSI = 0.702`; IQR outlier share jumps `0.1131 -> 0.3478` in test)
  - `avg_temperature`: `-0.4748 / 0.8472 -> 0.6606 / 0.9293` (`PSI = 0.753` due to seasonal March-May -> June progression)
  - `avg_humidity`: `-0.0502 / 1.0185 -> 0.2202 / 0.9849` (`PSI = 0.514`)
  - `avg_wind_level`: `0.0604 / 1.1036 -> 0.1048 / 0.8766` (`PSI = 0.132`)

## 5. Target Distribution (`price_p05`, `price_p95`)

- `price_p05`: mean `1.0215`, std `0.2469`, min `0.0000` (`104` rows across `4` SKUs), max `3.5122`.
- `price_p95`: mean `1.1169`, std `0.2381`, min `0.000109`, max `3.5466`.
- Interval width (`price_p95 - price_p05`): mean `0.0954`, median `0.0521`, p75 `0.1557`, max `1.4180`.
- Tight-interval concentration: `34.2%` have width `< 0.001`, `36.8%` `< 0.01`, `49.4%` `< 0.05`, `62.6%` `< 0.1`.

## 6. Key Drivers & Temporal Structure

- **Promotional activity (`activity_flag`)** lowers price levels and widens intervals:
  - `price_p05`: `1.1005 -> 0.9566` (`delta = -0.1439`, `r = -0.290`)
  - `price_p95`: `1.1485 -> 1.0910` (`delta = -0.0575`, `r = -0.120`)
  - Interval width: `0.0480 -> 0.1344` (`delta = +0.0864`)
- **Scale stabilization**: Raw width correlates negatively with `price_p05` (`r = -0.313`) and positively with `price_p95` (`r = +0.172`), whereas log-scale midpoint/width decouples level from spread (`r = 0.025`).
- **Lag-1 autocorrelation**: Pooled lag-1 correlation reaches `0.669` on `price_p05`, `0.599` on `price_p95`, and `0.808` on interval width (median intra-SKU correlations `0.487 / 0.478 / 0.484`), motivating EWM temporal post-processing.
- **Behavioral clustering & anomalies**: SKU-level `KMeans` separates narrow-spread high-price items (`Cluster 0`: `322` SKUs, mean width `0.051`) from wide-spread promo-sensitive items (`Cluster 1`: `163` SKUs, mean width `0.183`), while `IsolationForest` isolates extreme-spread SKUs (`product_id` `452, 60, 291, 69, 591, 392, 624, 444, 89, 613`).
