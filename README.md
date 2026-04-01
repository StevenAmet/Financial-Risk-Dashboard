# 🏦 Institutional Financial Risk Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red)
![License](https://img.shields.io/badge/License-MIT-green)

A full-stack financial risk analytics platform built with Python and Streamlit.

This dashboard simulates how a bank or investment portfolio behaves under stress, helping users understand:

- 📉 Potential losses under extreme scenarios  
- 💧 Liquidity sufficiency  
- 🏦 Capital adequacy under financial shocks  

---

## 🌐 Live Demo

👉 https://financial-risk-dashboard-sa.streamlit.app/

An interactive dashboard where you can:
- Upload your own portfolio  
- Adjust stress scenarios  
- Instantly visualize risk metrics  

---

## 🚀 What This Project Does

This tool answers three critical financial risk questions:

---

### 1. 📉 How much could I lose?

Using Monte Carlo simulation, the model generates thousands of possible outcomes:

- Value at Risk (VaR): worst expected loss in normal adverse conditions  
- Expected Shortfall (ES): average loss in extreme scenarios  

👉 What happens if things go really wrong?

---

### 2. 💧 Do I have enough cash?

Liquidity Coverage Ratio (LCR):

LCR = liquid assets / stressed cash outflows

- LCR > 1 → Safe  
- LCR < 1 → Potential liquidity risk  

---

### 3. 🏦 Can I survive financially?

CET1 Ratio:

CET1 Ratio = capital / risk-weighted assets

- Higher → Stronger institution  
- Lower → Higher risk of failure  

---

## ⚙️ Scenario Analysis

Simulate real-world financial shocks:

- 📉 Equity crash  
- 📈 Interest rate shock  
- ⚠️ Credit stress  
- 💸 Deposit run  
- 🧊 Liquidity freeze  

### Preset Scenarios

- 2008 Financial Crisis – Severe systemic stress  
- COVID Shock – Sharp, short-term disruption  

---

## 📁 Portfolio Input (Important)

Upload your own portfolio as a CSV file.

### Required Format

asset_type,value,PD,LGD,duration  
loan,1000000,0.02,0.4,0  
bond,1200000,0,0,5  
equity,500000,0,0,0  
cash,300000,0,0,0  

### Column Definitions

- asset_type → loan, bond, equity, cash  
- value → monetary value (€)  
- PD → probability of default (loans only)  
- LGD → loss given default (%)  
- duration → bond sensitivity to interest rates  

If no file is uploaded, a sample portfolio is used.

---

## 📈 Quantitative Models Included

### Correlation Matrix
- Measures diversification  
- High correlation → concentrated risk  
- Low correlation → diversified portfolio  

### PCA (Principal Component Analysis)
- Identifies main risk drivers  
- One dominant factor → systemic risk  

### Optimization
- Balances return vs risk  

### Machine Learning
- Predicts outcomes  
- Evaluates model accuracy  

---

## 📉 Visualizations

- Loss distribution  
- Correlation heatmap  
- PCA factor chart  
- ML prediction accuracy  

---

## 📊 Use Cases

- Banking dashboards  
- Portfolio stress testing  
- Finance education  
- Quantitative demonstrations  

---

## 📂 Project Structure

financial-risk-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE

---

## ⚙️ Run Locally

git clone https://github.com/StevenAmet/financial-risk-dashboard.git  
cd financial-risk-dashboard  
pip install -r requirements.txt  
streamlit run app.py  

---

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute financial advice.

---

## 👤 Author

Steven Amet  

GitHub: https://github.com/StevenAmet  
LinkedIn: https://www.linkedin.com/in/steven-amet/