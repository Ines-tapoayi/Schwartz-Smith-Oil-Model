# Schwartz-Smith-Oil-Model
## Overview
Implementation of the Schwartz & Smith (2000) two-factor commodity pricing model for WTI crude oil futures.

The project combines:
- Kalman Filtering
- Maximum Likelihood Estimation (MLE)
- Monte Carlo simulations
- Stress testing
- Interactive Plotly dashboards

to analyze short-term and long-term dynamics in oil prices.

## Model
The model decomposes oil prices into:
- χ (chi): short-term mean-reverting component
- ξ (xi): long-term equilibrium component

The state-space representation is estimated using a Kalman Filter.

## Dataset
WTI crude oil futures data:
- Period: 2000–2026
- 6,551 observations
- Multiple maturities interpolated on fixed pillars

## Methodology
### Data Processing
- Futures decoding
- Maturity computation
- Interpolation on fixed maturities

### Quantitative Modeling
- Kalman Filter
- Maximum Likelihood Estimation
- State extraction
- Monte Carlo simulation
- Stress testing

### Risk Analytics
- VaR / CVaR
- Scenario analysis
- Geopolitical stress tests

## Interactive Dashboards
The project includes 6 Plotly dashboards:
1. Candlestick & Volume
2. Schwartz-Smith decomposition
3. 12-month forecast
4. Risk profile & VaR
5. Geopolitical stress test
6. Backtesting

## Technologies
- Python
- NumPy
- Pandas
- SciPy
- Plotly
- Quantitative Finance
- State-space models

## Key Features
- Full Kalman-MLE calibration
- Monte Carlo engine (500 simulations)
- Commodity forward curve modeling
- Interactive HTML dashboards
- Power BI export

## Author
Inès Tapoayi
