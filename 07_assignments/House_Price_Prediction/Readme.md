# House Price Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-Linear%20Regression-red)
![Library](https://img.shields.io/badge/Scikit--Learn-green)
![Library](https://img.shields.io/badge/Pandas-green)
![Library](https://img.shields.io/badge/Matplotlib-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

In this Assignent we built a **Linear Regression** model to predict house prices using historical property data.

---

## Problem Statement

A real state company **HomeVista Properties** operates across multiple cities and handles thousands of residential property sales every year. The company wants to automate its house pricing process.

They wants to use **Machine Learning** to build an intelligent system that can **predict the market price of house automatically** based on its physical features, location, and condition.

**Dataset**: [Link](https://drive.google.com/file/d/110QuDECRgtGPU3XSevjb-qgnwI3ZiqUS/view?usp=drive_link) 

---

## Solution

I build a regression model that can accurately predict the house price using historical properties data.

Workflow of this assignment:

1. Import Libraries
2. Load Dataset
3. Data Exploration
4. Data Preprocessing - (Data Cleaning, Encoding)
5. Train-Test Split
6. Model Traing - (Linear Regression)
7. Model Evaluation
8. Feature Sacling Experiment
9. Conclusion

> This solution doesn't provide high accuracy, beacuse here model trained by simple **linear regression**.

---

## Features in the Dataset

Each row represents one residential house and its physical, location, and construction details.

| Feature      | Description           |
| ------------ | --------------------- |
| Id           | Unique house ID       |
| MSSubClass   | Type of dwelling      |
| MSZoning     | Zoning classification |
| LotArea      | Land size             |
| LotConfig    | Lot configuration     |
| BldgType     | Building Type         |
| OverallCond  | House Condtion        |
| YearBuilt    | Construction year     |
| YearRemodAdd | Remodeling year       |
| Exterior1st  | Exterior material     |
| BsmtFinSF2   | Basement finished area|
| TotalBsmtSF  | Total basement area   |
| SalePrice    | Target variable       |