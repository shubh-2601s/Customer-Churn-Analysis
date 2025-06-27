# 📉 Telecom Customer Churn Analysis

**Project Objective:**  
Predict customer churn in a telecom company and derive actionable insights to reduce churn and retain valuable customers.

---

## 📌 Overview

In the highly competitive telecom industry, customer retention is as critical as customer acquisition. This project focuses on building a churn prediction model using machine learning and deriving business strategies from the data.

---

## 🧰 Tools & Technologies Used

| Tool            | Purpose                             |
|-----------------|-------------------------------------|
| Python          | Data processing & ML modeling       |
| Pandas, NumPy   | Data manipulation                   |
| Scikit-learn    | Model building & evaluation         |
| ELI5, SHAP      | Model interpretability              |
| SQL (MySQL)     | Data aggregation from database      |
| Matplotlib, Seaborn | Data visualization              |
| PowerPoint      | Business presentation of findings   |

---

## 🗃️ Dataset Summary

- **Source:** SQL table `telecom_customers`  
- **Total Records:** 200 customers  
- **Features:**  
  - Age, Gender, Region, Plan Type, Monthly Charges  
  - Total Recharges, Call Duration, Complaints, Internet Usage  
  - Last Recharge Date (used to engineer `recency_days`)  
- **Target Variable:** `churn_flag` (1 = Churned, 0 = Retained)

---

## 🔧 Data Processing

- Handled missing values:
  - Numerical: Filled with median
  - Categorical: Filled with mode
- Categorical encoding using **Label Encoding**
- Feature engineering:
  - **`recency_days`** = Days since last recharge
- Final feature set: 12 variables

---

## 🤖 Model Building

- **Algorithm:** Random Forest Classifier (100 trees)
- **Train/Test Split:** 80:20 with stratification on churn
- **Performance Metrics:**
  - Accuracy: **80%**
  - ROC AUC Score: **0.8958**
  - High recall for churners (**81%**)

---

## 🧠 Model Explainability

Used **SHAP** and **ELI5** to interpret model predictions:

### 🔍 Top Predictive Features:
1. Monthly Charges
2. Recency Days
3. Internet Usage
4. Call Duration Total
5. Number of Complaints

Visualizations saved in the `plots/` folder:
- `shap_summary_dot.png`
- `shap_summary_bar.png`
- `eli5_feature_weights.html`

---

## 👥 Customer Segmentation

Customers were grouped based on predicted churn probability:

| Segment    | Churn Probability     | Description                |
|------------|------------------------|----------------------------|
| **At Risk**  | > 0.7                 | Needs immediate attention  |
| **Dormant**  | 0.3 – 0.7             | Watchful segment           |
| **Loyal**    | < 0.3                 | Retain with incentives     |

Segment plot: `customer_segments.png`

---

## 🌍 Regional Insights

- Analyzed churn rates by region (decoded after label encoding)
- Identified regions with higher churn than others
- Used for geo-targeted retention strategy

Plot saved: `churn_by_region.png`

---

## ✅ Final Recommendations

1. **Proactively engage At-Risk customers**
2. **Launch loyalty benefits for low-churn customers**
3. **Improve support experience to reduce complaints**
4. **Segment marketing campaigns based on churn risk**
5. **Target high-churn regions with personalized offers**

---

## 📄 Deliverables

- `telecom_churn_analysis.py`: Main Python notebook
- `plots/`: Folder with all generated plots
- `Telecom_Churn_Report.pptx`: Final presentation/report
- `README.md`: Project documentation

---

## 📁 Folder Structure

```

├── telecom\_churn\_analysis.py
├── telecom\_customers.csv
├── Telecom\_Churn\_Report.pptx
├── plots/
│   ├── shap\_summary\_dot.png
│   ├── shap\_summary\_bar.png
│   ├── eli5\_feature\_weights.html
│   ├── customer\_segments.png
│   └── churn\_by\_region.png
└── README.md

```

---

## 👨‍💻 Author

**Shubham S**  
_Data Analyst Intern, Elevate Labs_

[LinkedIn](#) · [GitHub](#)

---

## 📬 License

This project is for educational & research purposes. Feel free to fork and adapt with credit.

```

---


