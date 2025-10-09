# Valuation Engine Technical Specification

## 1. Project Brief & System Architecture

### 1.1 Project Mandate & Strategic Vision
- Build an institutional-grade Streamlit dashboard for public company valuation.
- Integrate Damodaran-style intrinsic valuation with Mauboussin expectations investing to surface actionable "Expectations Gap" insights.
- Deploy as a scalable cloud service (Render) with robust architecture, clear UX, and defensible valuation outputs.

### 1.2 Core Methodologies Overview
- **Damodaran Intrinsic Valuation**
  - Narrative-driven FCFF DCF yielding intrinsic value per share.
  - Multi-stage growth, WACC discounting, enterprise-to-equity bridge.
- **Mauboussin Expectations Investing**
  - Reverse DCF solving for market-implied value drivers (e.g., sales growth).
  - Focus on deducing consensus expectations from current market price.
- **Synthesis View**
  - Programmatically juxtapose analyst assumptions vs market-implied expectations.
  - Quantify the Expectations Gap to highlight variant perception opportunities.

### 1.3 Technical Stack & Environment
- Python 3.10+ with Streamlit front-end.
- Core libraries: `pandas`, `numpy`, `scipy`, `requests`, `alpaca-py`, `tiingo`.
- `requirements.txt` at repo root pins dependencies.
- Secrets (ALPACA_API_KEY, TIINGO_API_KEY, POLYGON_API_KEY) supplied via environment variables.

### 1.4 Data Ingestion Architecture
- `modules/data_provider.py` centralizes API access with provider-agnostic helpers.
- Each helper implements primary (Polygon via Alpaca) and secondary (Tiingo) fallbacks per Table 1 mappings.
- Handles normalization, authentication, retry/error handling, and caching hooks for Streamlit.
- Ensures resilience against provider deprecations and supplies consistent data structures to models.

## 2. Damodaran Intrinsic Valuation Implementation

### 2.1 Conceptual Summary
- FCFF-based multi-stage DCF discounted by WACC.
- Enterprise value adjusted to equity value and per-share intrinsic value.

### 2.2 Module 1 – Historical Financials
- `get_historical_financials(ticker: str, num_years: int) -> pd.DataFrame`.
- Fetch annual statements via `data_provider`, harmonize fields into canonical DataFrame (line items x fiscal years).
- Compute derived metrics: YoY revenue growth, EBIT margin, effective tax rate, sales-to-capital ratio.

### 2.3 Module 2 – WACC Calculation
- `calculate_wacc(ticker: str, historical_df: pd.DataFrame, user_assumptions: dict) -> float`.
- Cost of equity via CAPM (Rf, beta, ERP); beta fetched with user override.
- Cost of debt from latest statements; tax rate from history or overrides.
- Market value weights: market cap (real-time price × shares) and book debt.
- Returns blended WACC.

### 2.4 Module 3 – FCFF Forecasting
- `forecast_fcff(historical_df: pd.DataFrame, user_assumptions: dict) -> pd.DataFrame`.
- Uses user narrative inputs (forecast horizon, growth decay, target margins, sales-to-capital).
- Projects revenue, EBIT, reinvestment via sales-to-capital linkage.
- Computes FCFF = EBIT*(1-tax) − reinvestment.

### 2.5 Module 4 – Terminal Value & Equity Bridge
- `calculate_intrinsic_value(ticker: str, fcff_forecast_df: pd.DataFrame, wacc: float, user_assumptions: dict) -> dict`.
- Terminal value via Gordon Growth (stable growth ≤ risk-free rate).
- Discount FCFFs + terminal value to present for enterprise value.
- Adjust for debt/cash, divide by diluted shares to produce intrinsic value per share.
- Returns dict with enterprise, equity, and per-share values.

### 2.6 Scenario & Sensitivity Analysis
- Higher-order runner executes Base/Bull/Bear assumption sets.
- Sensitivity engine iterates over parameter ranges (e.g., WACC, margin) to produce tornado/line charts.

## 3. Mauboussin Expectations Investing Implementation

### 3.1 Conceptual Summary
- Reverse DCF determines market-implied growth (or horizon) consistent with observed enterprise value.

### 3.2 Module 5 – Market-Implied Enterprise Value
- `get_market_enterprise_value(ticker: str) -> float`.
- Uses `data_provider` for latest price, shares outstanding, debt, cash.
- Computes market cap and enterprise value baseline for solver.

### 3.3 Module 6 – Reverse DCF Solver
- `solve_for_implied_growth(ticker: str, target_ev: float, user_assumptions: dict) -> float`.
- Objective function runs forward DCF pipeline with candidate growth rate.
- Root finding via `scipy.optimize.brentq` across predefined bracket (e.g., −20% to 100%).
- Returns market-implied sales growth; alternate mode solves for forecast horizon given growth.

### 3.4 Module 7 – Expectations Analysis
- `analyze_expectations(implied_growth_rate: float, historical_df: pd.DataFrame, user_assumptions: dict) -> dict`.
- Compares implied growth vs historical 3/5/10-year CAGR.
- Generates pro-forma forecast consistent with implied growth.
- Supports alternative mode for market-implied forecast period.
- Supplies structured qualitative prompts for strategic plausibility assessment.

## 4. Streamlit Frontend & UX

### 4.1 App Structure
- Multi-page Streamlit app:
  - `Home.py` overview.
  - `pages/1_Damodaran_Valuation.py`.
  - `pages/2_Mauboussin_Expectations.py`.
  - `pages/3_Synthesis_View.py`.
- Global sidebar: ticker input, Run Analysis button, company snapshot.
- Session state persists data/models across pages.

### 4.2 Damodaran Page
- Tabs: Assumptions, Valuation Output, Scenario & Sensitivity.
- Sliders with historical context, narrative text area.
- Metrics for intrinsic value, price, upside/downside; tables/charts for projections.
- Scenario comparison table; sensitivity tornado chart.

### 4.3 Mauboussin Page
- Tabs: Implied Expectations, Strategic Analysis.
- Metric card for market-implied growth; comparison table vs history.
- Pro-forma forecast table; qualitative prompts for growth drivers, competitive advantage, risks/catalysts.

### 4.4 Synthesis View
- Dual columns summarizing user narrative vs market expectations.
- Central Expectations Gap card quantifying delta between required vs implied growth.
- Supports exporting insights (e.g., download button) for analyst reporting.

## 5. Deployment & Operations

### 5.1 Version Control
- Git/GitHub with Gitflow-style branching.
- Repo layout includes `modules/`, `pages/`, `requirements.txt`, `README.md`, `.gitignore`.

### 5.2 Render Deployment
- Web Service connected to GitHub.
- Build: `pip install -r requirements.txt`.
- Start: `streamlit run Home.py --server.port $PORT --server.address 0.0.0.0`.
- Environment variables store API keys; enable auto-deploy on main branch updates.

### 5.3 Reliability & Performance
- Defensive error handling for API calls with user-friendly Streamlit errors.
- Logging via Python `logging` to stdout/stderr for Render monitoring.
- Streamlit caching (`@st.cache_data`, `@st.cache_resource`) to minimize redundant API calls.

---

### Appendix: Key Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `modules/data_provider.py` | Unified data access with provider fallbacks, normalization, caching hooks. |
| `modules/damodaran_model.py` | Implements historical prep, WACC, FCFF forecast, valuation, scenario/sensitivity. |
| `modules/mauboussin_model.py` | Implements market EV, reverse DCF solver, expectations analysis workflows. |
| `Home.py` | Landing page and global session initialization. |
| `pages/1_Damodaran_Valuation.py` | Intrinsic valuation UI and scenario tooling. |
| `pages/2_Mauboussin_Expectations.py` | Market expectations UI with qualitative prompts. |
| `pages/3_Synthesis_View.py` | Comparative Expectations Gap dashboard. |

