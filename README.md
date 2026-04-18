# econ-lab-21-forecasting
# Time Series Forecasting — ARIMA, GARCH & Block Bootstrap

**Course:** ECON 5200 · Lab 21  
**Topic:** Diagnostic debugging of a broken ARIMA pipeline, GARCH(1,1) volatility modeling, and distribution-free forecast inference via moving block bootstrap.

---

## Objective

Diagnose three specification errors in a broken ARIMA forecasting pipeline, rebuild it as a SARIMA model with clean residuals, extend the analysis with a GARCH(1,1) volatility model on S&P 500 returns, and quantify forecast uncertainty using block bootstrap intervals.

---

## Methodology

### Part 1 — Diagnosis
Identified three planted errors in the original pipeline:
- **Error 1 — Integration order misspecification.** The original fit `ARIMA(cpi, order=(2,0,1))` on raw CPI levels despite an ADF statistic of 0.93 (p = 0.99) on log(CPI), failing to reject the unit root. The original ADF call also used `regression='ct'`, testing against trend-stationarity despite CPI having a stochastic rather than deterministic trend.
- **Error 2 — Ignored seasonal structure.** The pipeline applied non-seasonal ARIMA to `CPIAUCNS` (not seasonally adjusted), leaving significant residual autocorrelation at lags 12, 24, and 36 driven by energy cycles, holiday spending, and retail seasonality.
- **Error 3 — Forecasting without diagnostics, with a scale-mismatch plot.** The original pipeline skipped the Ljung-Box Q-test required before trusting forecast confidence intervals, and plotted a log-scale forecast against level-scale actuals, producing a visually meaningless chart. No train/test split was used, precluding any out-of-sample validation.

### Part 2 — Corrected Pipeline
- Applied log transformation to stabilize the multiplicative variance of the price index.
- Verified stationarity: ADF on Δlog(CPI) yielded a statistic of −3.45 (p = 0.009), rejecting the unit root at the 1% level.
- Used `pmdarima.auto_arima` with BIC selection over the SARIMA(p, 1, q)(P, 1, Q)[12] class, with a COVID regime dummy as an exogenous regressor to absorb the 2020 demand shock and 2021–2022 inflation break.
- Enforced a Ljung-Box "gate" on residuals at lags 12, 24, and 36 before permitting forecast generation, using proper degrees-of-freedom adjustment for the fitted ARMA parameters.
- Produced 24-month out-of-sample forecasts with Jensen-corrected back-transformation: $\hat{y} = \exp(\hat{\mu} + \hat{\sigma}^2/2)$ for point forecasts and quantile-preserving exponentiation for CI bounds.

### Part 3 — GARCH(1,1) on S&P 500 Returns
- Estimated a constant-mean GARCH(1,1) with Gaussian innovations via quasi-maximum likelihood (QMLE) on daily log returns, using Bollerslev–Wooldridge robust standard errors.
- Verified the covariance-stationarity condition α₁ + β₁ < 1 and computed implied long-run volatility and shock half-life.
- Visualized conditional volatility alongside realized returns to identify historical volatility regimes.

### Part 4 — Reusable Evaluation Module (`src/forecast_evaluation.py`)
Implemented two functions designed for general time series forecasting workflows:
- `compute_mase(actual, forecast, insample, m)` — Mean Absolute Scaled Error (Hyndman & Koehler, 2006), scale-free and benchmarked against a seasonal-naive model of period m.
- `backtest_expanding_window(series, model_fn, ...)` — Expanding-window pseudo-out-of-sample backtest following West (1996), accepting any callable forecaster via dependency injection for model-agnostic evaluation.
Both functions include NumPy-style docstrings, PEP 484 type hints, informative error messages, and a runnable self-test block.

### Challenge — Block Bootstrap Forecast Intervals
Implemented residual-based moving block bootstrap (Künsch, 1989; Pascual, Romo & Ruiz, 2004) for distribution-free forecast inference. Unlike analytical ARIMA intervals that assume iid Gaussian residuals, block bootstrap resamples the empirical residual distribution in contiguous blocks of length ⌈n^(1/3)⌉, preserving serial dependence while adapting to fat tails and mild heteroskedasticity.

---

## Key Findings

- **Corrected pipeline achieves MAPE = 0.74% over a 24-month out-of-sample horizon**, with 100% empirical coverage of the held-out actuals by the 95% forecast interval on the 2024-03 to 2026-03 period.
- **S&P 500 volatility persistence α₁ + β₁ ≈ 0.983**, implying a half-life of approximately **39.5 trading days** (roughly two calendar months) for a volatility shock to decay by half. This near-IGARCH behavior is consistent with the stylized facts in Engle & Bollerslev (1986) and the volatility-clustering literature.
- **Implied unconditional annualized volatility of 18.4%**, within the historical range for broad U.S. equity indices and serving as a sanity check on model specification.
- The ARCH coefficient α₁ = 0.120 quantifies the moderate sensitivity of conditional variance to new return shocks; the GARCH coefficient β₁ = 0.863 quantifies the strong persistence of past conditional variance — together capturing the empirical volatility-clustering phenomenon that Gaussian iid models cannot reproduce.
- Block bootstrap intervals adapt to the empirical residual distribution and do not rely on the Gaussian assumption underlying statsmodels' analytical confidence intervals, providing more honest uncertainty quantification in the presence of heavy tails or residual dependence.

---

## Caveats & Limitations

- The Gaussian innovation assumption in GARCH(1,1) is likely violated given the well-documented leptokurtosis of equity returns (Mandelbrot, 1963; Fama, 1965); a Student-t specification would likely produce tighter fit and more accurate tail-risk inference.
- Expanding-window backtests assume a reasonably stable data-generating process. The 2020–2022 period contains two major regime shifts (pandemic shock, inflation break), and rolling-window backtests could complement the expanding-window results for robustness.
- The block bootstrap implementation holds parameter estimates fixed across replications; a fully Pascual–Romo–Ruiz bootstrap would re-estimate SARIMA parameters in each replication to additionally capture parameter uncertainty.

---

## Repository Structure
