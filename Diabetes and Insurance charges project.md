## Predictive Analysis of Diabetes Risk Factors and Medical Costs
**Objective:** This project analyzed two datasets, the Diabetes Health Indicator Data from the CDC and the Medical Cost Personal Dataset from Brett Lantz's Machine Learning with R, to identify risk factors for diabetes and determinants of medical costs. The goal was to support preventive healthcare and inform cost-effective resource allocation.

**Methodology & Tools:** 
- Data Analysis: Employed Principal Components Analysis (PCA) and hierarchical clustering to explore correlations among 22 variables, covering over 253,000 entries in the Diabetes Health dataset.
- Machine Learning: Built classification models (Logistic Regression, Random Forest, Decision Tree) to predict diabetes occurrence. These models demonstrated variable precision, revealing that a combination of health indicators, such as BMI and smoking status, contributed to predictions.
- Cost Prediction: Applied Random Forest Regression to analyze the Medical Cost dataset, identifying significant predictors of insurance charges, including age, BMI, smoking status, and region.

**Key Results:**
- Discovered that no single factor could accurately predict diabetes, but combined indicators offered moderate predictive power.
- Identified lifestyle factors, such as smoking and obesity, as critical contributors to both diabetes risk and higher medical insurance costs.
- Delivered insights for cost management by forecasting medical expenses with a high degree of accuracy, emphasizing the role of demographics and health behaviors.
  
**Impact:** The findings underscore the importance of holistic health management and informed financial planning in combating chronic diseases like diabetes. This analysis provides actionable insights for optimizing healthcare policies aimed at both disease prevention and cost reduction.

**Tools Used:** Python, ScikitLearn(PCA, Random Forest, Decision Trees, Logistic Regression), Matplotlib for data visualization.
