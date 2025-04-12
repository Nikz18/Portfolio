# Healthcare Predictive Modelling: Diabetes and Medical Insurance Analysis

## Introduction
This project focuses on the analysis of two datasets: the **Diabetes Health Indicator Data** and the **Medical Cost Personal Dataset**. The goal is to uncover insights related to diabetes risk factors and understand the key determinants of medical insurance costs. By using machine learning techniques, such as **Principal Component Analysis (PCA)**, clustering, and regression models, the project aims to offer data-driven solutions for improving public health outcomes and providing cost-effective healthcare management.

The **Diabetes dataset**, derived from the CDC’s Behavioral Risk Factor Surveillance System (BRFSS), comprises over 253,000 records detailing health behaviors and chronic conditions in U.S. adults. The **Medical Cost dataset**, sourced from *Machine Learning with R* by Brett Lantz, contains 1,338 records that capture personal demographics and medical insurance charges.

Given the escalating healthcare costs and the rising incidence of diabetes, this project contributes to understanding predictive health factors and cost management, offering actionable insights for both individuals and policymakers.

## Key Findings

### 1. Diabetes Risk Factors:
- Health indicators such as **BMI**, high blood pressure, difficulty walking, and high cholesterol are the most strongly correlated with the likelihood of developing diabetes.
- However, no single factor alone is sufficient for accurately predicting diabetes. A combination of health behaviors, lifestyle factors, and demographic characteristics is crucial for a more robust prediction.

### 2. Unsupervised Learning:
- **Principal Component Analysis (PCA)** effectively reduced the dimensionality of the data, highlighting key subgroups related to diabetes risk factors: demographics, lifestyle factors, health indicators, and access to healthcare.
- **Clustering** revealed distinct groups of individuals based on diabetes risk, showing the potential for segmentation in targeted health interventions.

### 3. Predictive Models:
- Classification models (**logistic regression**, **random forest**, **decision trees**) performed well with accuracy rates around 84% but demonstrated limitations in precision, as reflected in low F1 scores for predicting diabetes.
- The results highlight that diabetes prediction requires a multi-factor approach, combining health factors, lifestyle, and BMI for improved model accuracy.

### 4. Medical Insurance Charges:
- **Regression analysis** identified that age, BMI, smoking status, and region are significant determinants of medical insurance charges.
- **Random forest regression** emerged as the most effective model, with a higher R² value (0.85) and lower error metrics (RMSE, MAE) compared to linear models, suggesting its suitability for predicting healthcare costs.

## Achievements
- **Data Preparation & Exploration**: The project involved extensive data wrangling, exploratory data analysis (EDA), and data visualization using Python libraries like Pandas, Matplotlib, and Seaborn. This facilitated the identification of key patterns and correlations within both datasets.
- **Dimensionality Reduction**: Successfully applied **Principal Component Analysis (PCA)** to reduce the complexity of the diabetes dataset, revealing key latent factors that affect diabetes risk.
- **Machine Learning Models**: Implemented and evaluated various classification models (**logistic regression**, **random forest**, **decision trees**) and regression models (**linear**, **lasso**, **ridge**, **random forest**) to predict diabetes occurrence and medical charges. Achieved strong results with **random forest regression**.
- **Insight Generation**: Identified key factors influencing both diabetes risk and healthcare costs, which can help inform preventive healthcare strategies and improve financial planning for individuals.
- **Practical Applications**: This project provides a data-driven framework for improving healthcare outcomes by predicting diabetes risk and managing medical costs effectively. The insights can be used to inform both public health policies and personal health management.

## Conclusion
This project emphasizes the importance of data-driven approaches in healthcare management, particularly in predicting chronic diseases like diabetes and understanding the drivers of medical costs. By leveraging machine learning, the study uncovers valuable insights that can improve preventive healthcare measures and cost management strategies.

The key findings show that health behaviors, lifestyle factors, and demographics are critical in predicting both diabetes risk and medical charges. While no single factor can provide an accurate prediction, a multi-dimensional approach that combines various health indicators and personal factors is essential for improving prediction accuracy.

## Impact and Future Work:
This research has significant potential in shaping preventive healthcare policies and cost-effective health management programs. Future work could include incorporating additional socioeconomic factors, exploring more advanced machine learning algorithms, and expanding the scope of analysis to different populations and regions for greater generalizability.

By continuing to refine these models, it’s possible to enhance disease prevention strategies and help individuals make more informed decisions regarding their health and healthcare expenses.
