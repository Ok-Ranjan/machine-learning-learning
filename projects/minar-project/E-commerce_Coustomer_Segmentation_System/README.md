# 🛒 SmartCart — E-Commerce Customer Segmentation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-yellowgreen?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Unsupervised-Clustering-lightblue" />
  <img src="https://img.shields.io/badge/Project-Minor%20Project-purple" />
</p>

> An intelligent customer segmentation system built using **unsupervised machine learning** to help SmartCart, a growing e-commerce platform, identify and understand distinct customer behaviour patterns and enable personalized marketing strategies.

---

## 📌 Problem Statement

SmartCart currently uses **generic marketing strategies** for all 2,240 customers — without understanding the differences among customer types. This leads to:

- ❌ Missed opportunities to retain **high-value customers**
- ❌ Delayed identification of **churn-prone users**
- ❌ **Inefficient, one-size-fits-all** marketing spend

**Solution:** Apply clustering algorithms to group customers into meaningful segments based on their purchasing behaviour, spending patterns, engagement levels, and demographic attributes.

---

## 🎯 Objectives

- Discover hidden patterns in customer behaviour from historical transaction data
- Segment customers into **actionable, meaningful groups**
- Provide **data-driven recommendations** for personalised marketing and customer retention

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Total Records | 2,240 customers |
| Total Features | 22 attributes |
| Source | SmartCart internal CRM data |
| File | `smartcart_customers.csv` |

### Feature Categories

| Category | Features |
|----------|----------|
| **Demographics** | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| **Spend (Amount)** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| **Purchase Frequency** | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases` |
| **Engagement** | `NumWebVisitsMonth`, `Recency`, `Response`, `Complain` |
| **Enrollment** | `Dt_Customer` |

---

## 🔄 Project Workflow

```
Raw Data
   │
   ├── EDA (shape, types, null check, statistics)
   │
   ├── Data Preprocessing
   │     ├── Missing Value Imputation (Income → mean)
   │     ├── Feature Engineering
   │     │     ├── Age = 2026 - Year_Birth
   │     │     ├── Customer_Tenure_Days
   │     │     ├── Total_Spending = sum of Mnt* columns
   │     │     ├── Total_Children = Kidhome + Teenhome
   │     │     ├── Living_With (Alone / Partner)
   │     │     └── Education (Undergraduate / Graduate / Postgraduate)
   │     ├── Outlier Removal (Income > 600K, Age > 90)
   │     └── Drop redundant raw columns
   │
   ├── Encoding → One-Hot Encoding (Education, Living_With)
   │
   ├── Scaling → StandardScaler
   │
   ├── Dimensionality Reduction → PCA (3 components, ~44% variance)
   │
   ├── Optimal K Selection
   │     ├── Elbow Method (WCSS)
   │     └── Silhouette Score → K = 4
   │
   ├── Clustering
   │     ├── KMeans (K=4)
   │     └── Agglomerative Clustering — Ward Linkage (K=4) ✅ Selected
   │
   └── Cluster Analysis & Business Insights
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Core language |
| **Pandas & NumPy** | Data manipulation |
| **Matplotlib & Seaborn** | Data visualization |
| **scikit-learn** | ML algorithms (KMeans, Agglomerative, PCA, StandardScaler, OHE) |
| **kneed** | Automated elbow/knee detection |
| **Jupyter Notebook** | Interactive development & presentation |

---

## 📈 Key Results

### Optimal Clusters: **K = 4**
Confirmed by both Elbow Method and Silhouette Score.

### Selected Algorithm: **Agglomerative Clustering (Ward Linkage)**
Produced better visual cluster separation and handles non-spherical data better than KMeans.

### Customer Segments Discovered

| Cluster | Persona | Income | Spending | Key Trait | Marketing Strategy |
|---------|---------|--------|----------|-----------|-------------------|
| **C0** | 🔴 Family Shoppers | Moderate | Moderate | Many children, high web visits | Discount coupons, family bundles |
| **C1** | 🔵 Premium Buyers | High | High | Few children, partnered, catalog shoppers | Loyalty programs, VIP access |
| **C2** | 🟡 Passive Browsers | Low | Very Low | Many children, high web visits, solo | Heavy discounts, flash sales |
| **C3** | 🟢 High-Value Singles | High | High | Few children, alone, **best response rate** | Premium services, personalisation |

> **C3 has the highest ROI potential** — best campaign response AND highest spending.

---

## 📂 Repository Structure

```
📦 E-commerce_Customer_Segmentation_System/
 ┣ 📓 SmartCart_Clustering_System.ipynb   ← Main notebook (full pipeline)
 ┣ 📄 smartcart_customers.csv            ← Dataset
 ┗ 📄 README.md                          ← Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/SmartCart-Customer-Segmentation.git
cd SmartCart-Customer-Segmentation
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed jupyter
```

### 4. Run the Notebook
```bash
jupyter notebook SmartCart_Clustering_System.ipynb
```

---

## 🔍 Notebook Structure

| Section | Description |
|---------|-------------|
| **Step 1** | Library Imports (centralized) |
| **Step 2** | Exploratory Data Analysis |
| **Step 3** | Data Preprocessing + Feature Engineering |
| **Step 4** | Categorical Encoding (OHE) |
| **Step 5** | Feature Scaling (StandardScaler) |
| **Step 6** | PCA — Dimensionality Reduction |
| **Step 7** | Optimal K — Elbow + Silhouette |
| **Step 8** | Clustering (KMeans vs. Agglomerative) |
| **Step 9** | Cluster Characterization + Business Insights |
| **Final** | Recommendations Table + Project Summary |

---

## 💡 Key Insights

1. **Income and Total Spending** are the two strongest indicators of customer segment membership
2. **High-income customers** prefer catalog and store purchases; they browse the web less
3. **Low-income customers** browse the web most but convert least — they need strong discounts
4. **Agglomerative Clustering (Ward)** outperforms KMeans on this dataset due to non-spherical cluster shapes
5. **4 segments** provide actionable granularity without over-segmentation

---

## 🚀 Future Enhancements

- [ ] Add **RFM (Recency, Frequency, Monetary) features** for richer behavioral scoring
- [ ] Try **DBSCAN** for density-based, outlier-robust clustering
- [ ] Build a **Streamlit dashboard** for real-time customer segment monitoring
- [ ] Deploy as a **REST API** for real-time customer tagging on purchase events
- [ ] Use **t-SNE** for better high-dimensional visualization

---

## 📚 Learning Outcomes

Through this project, the following concepts were applied end-to-end:

- Real-world data cleaning and feature engineering
- Unsupervised learning pipeline design
- Dimensionality reduction with PCA
- Model selection using Elbow and Silhouette methods
- Business-oriented cluster interpretation
- Data storytelling through visualizations

---

## 👨‍💻 Author

**Ranjan**  
B.Tech — Computer Science & Engineering  
Minor Project | 2025–26

---

## 📄 License

This project is for educational purposes. Dataset belongs to SmartCart (simulated CRM data for academic use).
