# Project Title: Airline Flight Delay Analysis and Prediction

## Introduction
This project explores flight delay patterns and develops predictive models to assess the likelihood of flight delays using data from U.S. commercial airlines over three years (2004–2006). The main objectives were to analyze trends in delays across months, weeks, times of day, and different aircraft models, and to use machine learning models to predict flight delays. We combined and pre-processed several datasets, including airport, carrier, and flight data, and applied various data visualization techniques to derive insights. Additionally, multiple classification models were built to predict delays based on various features, including departure and arrival times, flight duration, and aircraft details.

## Key Findings

### 1. Departure Delays by Month and Time:
- A clear pattern was observed in the monthly departure delays, where certain months consistently showed higher delays than others.
- Similarly, delays varied significantly by time of day, with the highest delays occurring during the afternoon and evening hours.

### 2. Impact of Plane Age on Delays:
- Older planes tended to experience more significant delays than newer ones. The analysis showed a strong correlation between plane age and departure delays, with newer models generally having fewer delays.

### 3. Flight Frequency Patterns:
- Flight frequencies between cities were analyzed, revealing the most frequently traveled routes. This was done by aggregating the number of flights departing from and arriving at different airports over the years.

### 4. Cascading Delays:
- An important insight was the identification of cascading delays—when a delay at the origin airport leads to delays at the destination. The analysis revealed significant patterns where certain delays would "snowball," affecting a large number of subsequent flights.

### 5. Predictive Model Performance:
Several machine learning models were implemented to predict flight delays, including:
- **Logistic Regression**: Performed well, providing a solid baseline model for delay prediction.
- **Gradient Boosting**: Outperformed logistic regression in terms of accuracy, particularly in handling complex interactions between variables.
- **Penalized Logistic Regression**: Showed promise in preventing overfitting, especially when tuned with cross-validation.
- **Random Forests and SVM**: Delivered competitive results and helped refine the overall prediction accuracy.

## Achievements
- **Data Preprocessing and Cleaning**: Successfully cleaned and merged datasets, handling missing values, converting times into a consistent format, and categorizing times of day into meaningful intervals.
- **Exploratory Data Analysis (EDA)**: Performed a comprehensive EDA, including identifying the distribution of delays across months, weeks, and times of day. Produced several insightful visualizations (bar charts, scatter plots, and joy plots) to illustrate findings.
- **Predictive Modeling**: Built a series of classification models (Logistic Regression, Gradient Boosting, Penalized Regression, Random Forests, SVM) to predict flight delays. The best-performing model was evaluated using cross-validation and benchmarked against multiple metrics.
- **Visualization of Results**: Created several visualizations that illustrated the distribution of delays by time, plane age, and flight frequency. These visualizations helped in identifying trends and anomalies in the data.

## Conclusion
This project successfully addressed the problem of predicting flight delays by uncovering critical insights from the flight data and employing machine learning to build a robust predictive model. The analysis revealed that time of day, month, and plane age significantly affect delays, and the cascading nature of delays has important implications for airline operations. The machine learning models built showed varying levels of accuracy, with **Gradient Boosting** and **Random Forests** being the most effective in predicting delays. This project provides a comprehensive framework for analyzing and predicting flight delays, which can be used for operational improvements in the airline industry.

## Future Work
- **Enhancing Model Accuracy**: Future work could include incorporating more granular data (e.g., weather conditions, airport congestion) to improve predictive accuracy.
- **Real-Time Prediction System**: Developing a real-time system to predict flight delays based on live data could improve airline operations and customer experience.
- **Cost Analysis**: An analysis of the operational and financial impacts of delays, to provide actionable recommendations for minimizing disruptions.
