HDB Resale Price Regression Analysis

ST3131 (Regression Analysis) assignment at NUS. Predicts HDB resale prices using multiple linear regression on July 2020 transaction data.

Pipeline covers feature engineering (OneMap API geocoding for distance to CBD and nearest MRT, storey midpoint extraction, lease parsing), EDA, and diagnostics. The baseline OLS model showed high VIF, heteroscedasticity, and non-normal residuals. Fixed via model respecification and a log transform on the target. Ridge was tested as an alternative but dropped for interpretability.

Final model: log(resale_price) ~ town + floor_area_sqm + remaining_lease + storey_avg + dist_to_mrt, adjusted R² = 0.895.
