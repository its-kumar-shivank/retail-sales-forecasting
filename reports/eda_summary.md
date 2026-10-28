# EDA Summary

## Revenue and growth

- 2022: ₹127.9 crore
- 2023: ₹149.0 crore (YoY +16.5%)
- 2024: ₹173.6 crore (YoY +16.5%)
- 2025: ₹201.6 crore (YoY +16.1%)

## Category mix

- Grocery & Staples: ₹200.2 crore, margin 12.5%
- Beverages & Snacks: ₹119.1 crore, margin 25.2%
- Apparel: ₹95.4 crore, margin 41.0%
- Electronics: ₹89.9 crore, margin 18.5%
- Personal Care: ₹74.5 crore, margin 35.4%
- Home & Kitchen: ₹72.9 crore, margin 32.9%

## Weekly pattern

- Weekend uplift is strongest in Apparel (Sat index 1.32) and weakest in Personal Care (Sat index 1.09).

## Festival effect (Diwali)

- Peak demand index in the 3 days before Diwali: Home & Kitchen 3.5x, Electronics 3.1x, Apparel 3.0x, Beverages & Snacks 2.2x, Grocery & Staples 1.6x, Personal Care 1.5x
- Diwali moves between Oct 20 and Nov 12, so a same-day-last-year lag misses it; a `days_to_diwali` feature is needed.

## Promotions

- 19.5% of store-product-days are on promotion.
- Median lift at 20% off: Electronics 2.01x, Apparel 1.87x, Home & Kitchen 1.80x, Personal Care 1.65x, Beverages & Snacks 1.59x, Grocery & Staples 1.40x

## Stores

- Top store by revenue: BharatMart Andheri (₹105.2 crore).
- Best space productivity: BharatMart MI Road (Express), ₹14,462 per sq ft per year. Small Express stores punch above their size.

## Autocorrelation

- ACF at lag 7 = 0.54, lag 28 = 0.41, lag 364 = 0.44. Strong weekly and yearly memory -> lags at multiples of 7 and lag 364 are the core forecasting features.

## Target distribution

- Units are non-negative counts, right-skewed, 0.8% zeros. Fast movers (grocery ~100/day) and slow movers (electronics ~7/day) differ by >10x, so the model uses a Tweedie loss (built for skewed non-negative targets) instead of plain MSE.
