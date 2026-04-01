# 🏦 Institutional Financial Risk Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red)
![License](https://img.shields.io/badge/License-MIT-green)

A full-stack **financial risk analytics platform** built in Python using Streamlit.

This dashboard simulates how a bank or investment portfolio behaves under stress, helping users understand:
- How much money could be lost in extreme scenarios
- Whether the portfolio has enough liquidity (cash)
- Whether the institution has enough capital to survive shocks

---

## 🌐 Live Demo

👉 **Streamlit App:**  
https://financial-risk-dashboard-sa.streamlit.app/

> Interactive dashboard where you can upload a portfolio, adjust stress scenarios, and instantly see risk metrics and visualizations.

---

## 🚀 What This Project Does (In Simple Terms)

This tool answers three key questions:

### 1. 📉 “How much could I lose?”
Using **Monte Carlo simulation**, the model generates thousands of possible future outcomes and estimates:
- **VaR (Value at Risk)** → worst expected loss in normal bad conditions  
- **Expected Shortfall** → average loss in extreme scenarios  

👉 Think of this as: *“What happens if things go really wrong?”*

---

### 2. 💧 “Do I have enough cash?”
The model calculates:

- **Liquidity Coverage Ratio (LCR)**  
  = liquid assets / stressed cash outflows  

👉 If LCR > 1 → you're safe  
👉 If LCR < 1 → potential liquidity crisis  

---

### 3. 🏦 “Can I survive financially?”
Using Basel-style rules:

- **CET1 Ratio (Capital Strength)**  
  = capital / risk-weighted assets  

👉 Higher = safer bank  
👉 Lower = higher risk of failure  

---

## ⚙️ Scenario Analysis

You can simulate real-world crises by adjusting:

- 📉 Equity crash (stock market drop)  
- 📈 Interest rate shock (bond losses)  
- ⚠️ Credit stress (loan defaults increase)  
- 💸 Deposit run (customers withdraw money)  
- 🧊 Liquidity stress (assets become harder to sell)  

### Preset Scenarios:
- **2008 Financial Crisis** → extreme systemic stress  
- **COVID Shock** → sharp but shorter disruption  

---

## 📁 Portfolio Input (Very Important)

You can upload your own portfolio using a CSV file.

### Required Format:

```csv
asset_type,value,PD,LGD,duration
loan,1000000,0.02,0.4,0
bond,1200000,0,0,5
equity,500000,0,0,0
cash,300000,0,0,0

📌 Column Explanation
asset_type → loan, bond, equity, cash
value → monetary value (€)
PD → probability of default (loans only)
LGD → % loss if loan defaults
duration → bond sensitivity to rates

👉 If no file is uploaded, a sample portfolio is used.

📈 Quantitative Models
🔗 Correlation Matrix
High correlation → poor diversification
Low correlation → good diversification
📊 PCA
Identifies main risk drivers
One dominant factor = systemic risk
⚙️ Optimization
Balances return vs risk
🤖 Machine Learning
Predicts outcomes
Measures model accuracy
📉 Visualizations
Loss Distribution → shows possible losses
Correlation Heatmap → asset relationships
PCA Chart → risk factor contribution
ML Plot → prediction accuracy
📊 Use Cases
Banking dashboards
Portfolio stress testing
Finance learning
Quant demos
📂 Project Structure
financial-risk-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
⚙️ Run Locally
git clone https://github.com/StevenAmet/financial-risk-dashboard.git
cd financial-risk-dashboard
pip install -r requirements.txt
streamlit run app.py
⚠️ Disclaimer

Educational use only. Not financial advice.

👤 Author

Steven Amet
GitHub: https://github.com/StevenAmet

LinkedIn: https://www.linkedin.com/in/steven-amet/