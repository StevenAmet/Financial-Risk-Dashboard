# -------------------------------
# run in cmd using: streamlit run app.py
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Financial Risk Dashboard", layout="wide")

# -------------------------------
# UI
# -------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 13px;
}
[data-testid="stMetric"] {
    background-color: #F5F7FA !important;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #D0D7DE;
}
[data-testid="stMetric"] * {
    color: #000000 !important;
}
@media (prefers-color-scheme: dark) {
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
    }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("🏦 Financial Risk Dashboard")
st.caption("Market | Credit | Liquidity | Capital | Quant Models")
st.markdown("**Created by Steven Amet**")

# -------------------------------
# CSV UPLOAD
# -------------------------------
st.markdown("### 📁 Upload Portfolio")

st.info("""
Upload a CSV with the following columns:

- asset_type → loan, bond, equity, cash  
- value → numeric (€)  
- PD → probability of default (loans only)  
- LGD → loss given default (loans only)  
- duration → bond sensitivity to rates  
""")

uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Scenario Controls")

scenario = st.sidebar.selectbox(
    "Preset Scenario",
    ["Custom", "2008 Crisis", "COVID Shock"]
)

if scenario == "2008 Crisis":
    st.sidebar.markdown("""
    **2008 Crisis Scenario**
    - Equity markets crash ~50%
    - Credit defaults spike
    - Liquidity dries up
    """)
    equity_shock, rate_shock, credit_multiplier = -0.5, 0.03, 3
    deposit_run, liquidity_stress = 0.3, 0.2

elif scenario == "COVID Shock":
    st.sidebar.markdown("""
    **COVID Shock Scenario**
    - Rapid equity selloff
    - Moderate credit stress
    - Short-term liquidity pressure
    """)
    equity_shock, rate_shock, credit_multiplier = -0.3, 0.01, 2
    deposit_run, liquidity_stress = 0.2, 0.1

else:
    equity_shock = st.sidebar.slider("Equity Shock (%)", -60, 0, -40) / 100
    rate_shock = st.sidebar.slider("Rate Shock (%)", 0, 5, 3) / 100
    credit_multiplier = st.sidebar.slider("Credit Multiplier", 1.0, 5.0, 3.0)
    deposit_run = st.sidebar.slider("Deposit Run (%)", 0, 50, 20) / 100
    liquidity_stress = st.sidebar.slider("Liquidity Stress (%)", 0, 50, 10) / 100

# -------------------------------
# PORTFOLIO
# -------------------------------
if uploaded_file:
    portfolio = pd.read_csv(uploaded_file)
else:
    portfolio = pd.DataFrame({
        "asset_type": ["loan", "loan", "bond", "bond", "equity", "equity", "cash"],
        "value": [1_000_000, 800_000, 1_200_000, 600_000, 500_000, 400_000, 700_000],
        "PD": [0.02, 0.03, 0, 0, 0, 0, 0],
        "LGD": [0.4, 0.5, 0, 0, 0, 0, 0],
        "duration": [0, 0, 5, 7, 0, 0, 0]
    })

# -------------------------------
# OVERVIEW
# -------------------------------
st.markdown("### 📊 Portfolio Overview")
summary = portfolio.groupby("asset_type")["value"].sum()
st.dataframe(summary)

# -------------------------------
# FUNCTIONS
# -------------------------------
def run_stress(df, n=500):
    results = []
    for _ in range(n):
        pnl = 0
        systemic = np.random.normal(-2.5, 1)
        for _, row in df.iterrows():
            if row["asset_type"] == "loan":
                pd_s = min(row["PD"] * credit_multiplier * (1 + abs(systemic)), 1)
                if np.random.rand() < pd_s:
                    pnl -= row["LGD"] * row["value"]
            elif row["asset_type"] == "equity":
                pnl += row["value"] * equity_shock * (1 + abs(systemic))
            elif row["asset_type"] == "bond":
                pnl += row["value"] * (-row["duration"] * rate_shock)
        results.append(pnl)
    return np.array(results)

def liquidity_model(df):
    cash = df[df["asset_type"] == "cash"]["value"].sum()
    bonds = df[df["asset_type"] == "bond"]["value"].sum()
    hqla = cash + bonds * (0.9 - liquidity_stress)

    loans = df[df["asset_type"] == "loan"]["value"].sum()
    equities = df[df["asset_type"] == "equity"]["value"].sum()

    outflows = (
        loans * (0.1 + deposit_run) +
        equities * (0.15 + deposit_run) +
        bonds * (0.05 + liquidity_stress)
    )

    return hqla / outflows

def basel_capital(df):
    weights = {"cash":0,"bond":0.2,"loan":1,"equity":1}
    df = df.copy()
    df["rw"] = df["asset_type"].map(weights)
    rwa = (df["rw"] * df["value"]).sum()
    return rwa, 1_000_000 / rwa

def generate_returns(n=300):
    return pd.DataFrame({
        "loan": np.random.normal(0.06, 0.08, n),
        "bond": np.random.normal(0.03, 0.05, n),
        "equity": np.random.normal(0.10, 0.18, n),
        "cash": np.random.normal(0.02, 0.01, n)
    })

# -------------------------------
# RUN
# -------------------------------
stress = run_stress(portfolio)
returns = generate_returns()

mean_returns = returns.mean()
cov = returns.cov()
scores = mean_returns / np.diag(cov)
weights = scores / scores.sum()

X = returns
y = returns.sum(axis=1)
model = LinearRegression().fit(X, y)
pred = model.predict(X)

VaR_95 = np.percentile(stress, 5)
VaR_99 = np.percentile(stress, 1)
ES = stress[stress <= VaR_95].mean()

lcr = liquidity_model(portfolio)
rwa, cet1 = basel_capital(portfolio)

# -------------------------------
# METRICS + EXPLANATION
# -------------------------------
st.markdown("### 📌 Key Risk Metrics")

st.info("""
**How to interpret:**
- **VaR 99%** → worst loss expected in extreme conditions (1% scenarios)
- **Expected Shortfall** → average loss beyond VaR (tail risk)
- **LCR** → liquidity strength (must be >1 to survive stress)
- **CET1** → capital buffer vs risk-weighted assets
- **Sharpe Ratio** → return per unit of risk (higher = better)
""")

ret = np.dot(weights, mean_returns)
risk = np.sqrt(weights.T @ cov @ weights)
sharpe = ret / risk

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("VaR 99%", f"€{VaR_99/1e6:.2f}M")
c2.metric("Expected Shortfall", f"€{ES/1e6:.2f}M")
c3.metric("LCR", f"{lcr:.2f}x")
c4.metric("CET1", f"{cet1:.2%}")
c5.metric("Sharpe", f"{sharpe:.2f}")

# -------------------------------
# ALERTS
# -------------------------------
if VaR_99 < -2_000_000:
    st.error("⚠️ Extreme losses exceed €2M — consider reducing risk")

if returns.corr().max().max() > 0.8:
    st.warning("⚠️ High correlation detected — poor diversification")

# -------------------------------
# CORRELATION (UNCHANGED)
# -------------------------------

st.markdown("### 🔗 Correlation Matrix")

st.info("""
This matrix shows how different asset classes move relative to each other.

How to interpret:

- **+1 correlation** → assets move together  
    → increases systemic risk (everything falls at once)

- **0 correlation** → no relationship  
    → assets behave independently

- **Negative correlation** → assets move in opposite directions  
    → provides diversification benefits

Why this matters:

- High correlation across assets = **concentrated risk**
- Low or negative correlation = **diversified portfolio**

👉 Key takeaway:
Even if assets look different, if they move together, your portfolio may still be risky.
""")

fig_corr, ax = plt.subplots(figsize=(4.5, 2.5))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
plt.tight_layout()
st.pyplot(fig_corr)

# -------------------------------
# PCA + LOADINGS
# -------------------------------
# -------------------------------
# PCA
# -------------------------------
st.markdown("### 📊 PCA Risk Drivers")

st.info("""
Principal Component Analysis (PCA) identifies the main underlying drivers of risk in the portfolio.

How to read the chart:

- Each bar represents a "risk factor" (principal component)
- The height shows how much of total portfolio risk it explains

Key insights:

- **First bar (PC1)**:
    → the most important risk driver  
    → often represents broad market/systemic risk

- If PC1 is very large:
    → portfolio is heavily exposed to ONE dominant factor  
    → high systemic vulnerability  

- If multiple bars are similar:
    → risk is spread across multiple sources  
    → better diversification  

👉 PCA Loadings Table:
- Shows which assets contribute most to each risk factor  
- Example:
    - If equities + loans dominate PC1 → market/credit driven risk  

👉 Simple interpretation:
- Concentrated PCA = fragile portfolio  
- Balanced PCA = resilient portfolio  

This answers:
*"What is REALLY driving my risk under the surface?"*
""")

pca = PCA().fit(returns)

fig_pca, ax = plt.subplots(figsize=(4.5, 2.5))
ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color="#55A868")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.tight_layout()
st.pyplot(fig_pca)

loadings = pd.DataFrame(
    pca.components_,
    columns=returns.columns,
    index=[f"PC{i+1}" for i in range(len(returns.columns))]
)
st.dataframe(loadings)

# -------------------------------
# LOSS DISTRIBUTION
# -------------------------------

st.markdown("### 📉 Loss Distribution")

st.info("""
This chart shows all simulated profit and loss outcomes from the Monte Carlo stress test.

How to read this:

- Each bar represents how often a specific profit/loss outcome occurs  
- The left side (negative values) represents losses  
- The further left you go → the more severe the loss  

Key insights:

- **Left tail** = extreme worst-case losses (rare but critical scenarios)  
- **VaR lines** = thresholds for expected worst losses  
    - VaR 95% → losses exceeded only 5% of the time  
    - VaR 99% → losses exceeded only 1% of the time  

- **Expected Shortfall (ES)** goes beyond VaR:
    → it shows the *average loss when things are really bad*

- **Width of distribution**:
    - Wide = high volatility (uncertain portfolio)
    - Narrow = more stable outcomes

👉 In simple terms:
This answers: *"How bad can things get — and how often?"*
""")

fig, ax = plt.subplots(figsize=(5, 2.5))
sns.histplot(stress, bins=40, kde=True, ax=ax, color="#4C72B0")
ax.axvline(VaR_95, linestyle="--", color="orange")
ax.axvline(VaR_99, linestyle="--", color="red")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'€{x/1e6:.1f}M'))
plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# OPTIMIZATION
# -------------------------------
st.markdown("### ⚙️ Portfolio Optimization")

st.info("""
This section shows the optimal allocation of assets to balance return and risk.

How it works:

- The model evaluates expected returns vs volatility (risk)
- It assigns weights to maximize efficiency

Key metrics:

- **Return** → expected portfolio performance  
- **Risk** → volatility (uncertainty of returns)  
- **Sharpe Ratio** → return per unit of risk  

How to interpret:

- Higher return = more aggressive strategy  
- Lower risk = more stable portfolio  
- Higher Sharpe Ratio = better risk-adjusted performance  

👉 In practice:
This helps answer:
*"Am I taking the right amount of risk for the return I'm getting?"*
""")

st.write(f"Return: {ret:.2%}")
st.write(f"Risk: {risk:.2%}")

# -------------------------------
# ML
# -------------------------------
st.markdown("### 🤖 ML Risk Prediction")

st.info("""
This model uses machine learning (linear regression) to predict total portfolio returns.

What the chart shows:

- Each point = one observation  
- X-axis = actual returns  
- Y-axis = predicted returns  

How to interpret:

- Points close to the diagonal line → accurate predictions  
- Points far from the line → prediction errors  

Key insights:

- Tight clustering around the line = strong predictive model  
- Wide scatter = weaker model / more randomness in returns  

👉 Important note:
Financial markets are inherently noisy, so perfect prediction is not expected.

👉 This answers:
*"How well can we estimate future performance using historical patterns?"*
""")

fig_ml, ax = plt.subplots(figsize=(4.5, 2.5))
ax.scatter(y, pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--", color="red")
plt.tight_layout()
st.pyplot(fig_ml)

# -------------------------------
# EXECUTIVE SUMMARY
# -------------------------------
st.markdown("### 🧠 Executive Summary")

st.write(f"""
This portfolio shows a Value-at-Risk of €{abs(VaR_99)/1e6:.2f}M, indicating potential downside exposure under extreme conditions.

Liquidity coverage stands at {lcr:.2f}, suggesting the portfolio is {'strong' if lcr > 1.5 else 'adequate' if lcr > 1 else 'at risk'}.

Capital adequacy (CET1) is {cet1:.2%}, reflecting {'strong capitalization' if cet1 > 0.15 else 'moderate capital levels'}.

👉 Overall, risk is driven by market shocks and asset correlations.
""")

# -------------------------------
# PDF EXPORT
# -------------------------------
def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Risk Report - Steven Amet", styles["Title"]))
    content.append(Paragraph(f"VaR: €{VaR_99/1e6:.2f}M", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer

st.download_button("📄 Download Report", generate_pdf(), "risk_report.pdf")