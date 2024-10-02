## Project Title: Flight Delay Analysis and Prediction

**Abstract:**

This project analyzes flight delay data from the American Statistical Association (ASA) (Harvard Dataverse, 2008). It includes exploratory data analysis, delay prediction modeling, and cascading delay detection to provide insights into the frequency and distribution of flight delays across various airports, airlines, and time frames. The analysis reveals optimal travel conditions and identifies significant factors influencing delays, with the Gradient Boosting model demonstrating the highest accuracy in delay predictions.

**Methodology:**

Utilizing *Jupyter Notebook*, I processed three years of U.S. flight data (2004-2006) by *combining datasets*, *handling missing values*, and *filtering out outliers*. Key variables, such as flight numbers, airplane tail numbers, and IATA codes, were identified and analyzed. I employed various machine learning algorithms, including *Logistic Regression*, *Random Forest*, and *Gradient Boosting*, for delay prediction, and implemented a *data wrangling* process for cascading delay detection.

**Key Findings:**

- **Optimal Travel Times:** Analysis indicated that mornings (05:00 - 10:59) and the month of April are the best times to minimize delays.
- **Impact of Plane Age:** A negative correlation between airplane age and delay stability was observed; older planes exhibited more consistent delay patterns.
- **Cascading Delays:** Results suggested that delays at one airport can cause subsequent delays at others, highlighting the need for improved coordination among airports.
- **Prediction Model Performance:** The Gradient Boosting model outperformed others with the highest accuracy in predicting flight delays.
  
**Conclusion:** 
This comprehensive analysis provides valuable insights for passengers and airlines to enhance scheduling and operational efficiency, thereby improving on-time performance (OTP).
