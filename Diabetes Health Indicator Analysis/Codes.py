import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import chisquare
import scipy.cluster.hierarchy as shc


# Change the working directory
new_directory = "C:/documents/UOL/ST 3189 Machine Learning/Coursework"
os.chdir(new_directory)

DHI = pd.read_csv("diabetes health indicator.csv")

# Summary and column names
print(DHI.describe())
print(DHI.columns)

# Remove NAs
DHI = DHI.dropna()
print(DHI.isna().sum())


DHI_1 = DHI.copy()
diabetes_02_index = DHI_1[DHI_1['Diabetes_012'] == 2].index
DHI_1.loc[diabetes_02_index, 'Diabetes_012'] = 1
print(DHI_1['Diabetes_012'].value_counts())

# Fix target variable imbalance
#combine prediabetes with diabetes as 1
DHI_1['Diabetes_012'] = np.where(DHI_1['Diabetes_012'] == 2, 1, DHI_1['Diabetes_012'])

mapping = {5: 1, 4: 2,3: 3, 2: 4, 1: 5}

# Replace values in GenHlth column using mapping dictionary
DHI_1['GenHlth'] = DHI_1['GenHlth'].map(mapping)

# Correlation plot
cor_matrix = DHI_1.corr()

plt.figure(figsize=(16, 16))
sns.heatmap(cor_matrix, annot=True, linewidths=2, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})
plt.show()

# Feature importance using Random Forest
predictors = DHI.drop(columns=['Diabetes_012'])
target = DHI['Diabetes_012']
data = StandardScaler().fit_transform(predictors)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=3187)

pca_data = data
pca_result = PCA().fit(pca_data)

# Scree Plot
plt.figure(figsize=(6, 6))
plt.plot(pca_result.explained_variance_ratio_, marker='o')
plt.title("Scree Plot", fontsize=14)
plt.show()
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# Add variable names from the original dataset
pca_df.columns = predictors.columns  

print(pca_df.head())
# Find the optimal number of clusters using the Elbow Method
inertia = []
max_clusters = 10 

for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=3187)
    kmeans.fit(scaled_data)
    
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, max_clusters + 1), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.show()
# Choose the optimal number of clusters (elbow point)
optimal_clusters = 2

# Apply K-Means clustering with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=3187)
clusters_optimal = kmeans_optimal.fit_predict(scaled_data)

# Reduce dimensions for visualization 
pca = PCA(n_components=2)
X_pca_optimal = pca.fit_transform(scaled_data)

# Visualize clusters using Seaborn cluster plot
cluster_df = pd.DataFrame(data={'Cluster': clusters_optimal, 'PC1': X_pca_optimal[:, 0], 'PC2': X_pca_optimal[:, 1]})
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=cluster_df, palette='viridis', marker='o', edgecolor='k')
plt.title('Cluster Plot with Optimal Number of Clusters')
plt.show()
sub1 = DHI.drop(columns=['Diabetes_012']).copy()

# Group the variables
sub1['LifestyleFactors'] = sub1[['PhysActivity', 'Fruits', 'Veggies', 'Smoker', 'HvyAlcoholConsump']].mean(axis=1)
sub1['HealthFactors'] = sub1[['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke']].mean(axis=1)
sub1['OverallHealth'] = sub1[['GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk']].mean(axis=1)
sub1['Demographic'] = sub1 [['Sex', 'Age', 'Education','Income']].mean(axis=1)
# Drop the original variables
sub1.drop(['PhysActivity', 'Fruits', 'Veggies', 'Smoker','Stroke',
           'HighBP', 'HighChol', 'HeartDiseaseorAttack','DiffWalk','PhysHlth',
           'GenHlth', 'MentHlth','Sex', 'Age', 'Education','Income','CholCheck','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost'], axis=1, inplace=True)

sub2 = DHI.copy()
# Extract the subsets of predictors based on the specified variables
physical_activity_df = sub2[['PhysActivity', 'Fruits', 'Veggies', 'Smoker', 'HvyAlcoholConsump']]
health_factors_df = sub2[['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke']]
overall_health_df = sub2[['GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk']]
healthcare_access_df = sub2[['NoDocbcCost', 'AnyHealthcare', 'CholCheck']]
demographics_df = sub2[['Sex', 'Age', 'Education', 'Income']]
target = sub2[['Diabetes_012']]
# Rename the columns using MultiIndex
physical_activity_df.columns = pd.MultiIndex.from_product([['LifestyleFactors'], physical_activity_df.columns])
health_factors_df.columns = pd.MultiIndex.from_product([['HealthFactors'], health_factors_df.columns])
overall_health_df.columns = pd.MultiIndex.from_product([['OverallHealth'], overall_health_df.columns])
healthcare_access_df.columns = pd.MultiIndex.from_product([['HealthcareAccess'], healthcare_access_df.columns])
demographics_df.columns = pd.MultiIndex.from_product([['Demographics'], demographics_df.columns])
target.columns = pd.MultiIndex.from_product([['Target'], target.columns])
# Combine into one DataFrame
combined_df = pd.concat([physical_activity_df, health_factors_df, overall_health_df, healthcare_access_df, demographics_df,target], axis=1)

combined_df


Health = sub2[['GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk','HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke']]
Lifestyle_Demo = sub2[['PhysActivity', 'Fruits', 'Veggies', 'Smoker', 'HvyAlcoholConsump','Sex', 'Age', 'Education', 'Income']]
Body_Index = sub2[['BMI']]

Health_with_diabetes = pd.concat([Health, DHI_1['Diabetes_012']], axis=1)
Lifestyle_with_diabetes = pd.concat([Lifestyle_Demo, DHI_1['Diabetes_012']], axis=1)
BMI_with_diabetes = pd.concat([Body_Index, DHI_1['Diabetes_012']], axis=1)

# 1. Standardize the data
scaler = StandardScaler()
sub1_scaled = scaler.fit_transform(sub1)

# 2. Compute the linkage matrix using hierarchical clustering
linkage_matrix = linkage(sub1_scaled.T, method='ward')

# 3. Plot the dendrogram
plt.figure(figsize=(5, 5))
dendrogram(linkage_matrix, labels=sub1.columns, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram', fontsize=16) 
plt.xlabel('Sample Index', fontsize = 10)  
plt.ylabel('Distance', fontsize=14)  
plt.show()

# Function to plot confusion matrix using ConfusionMatrixDisplay
def plot_confusion_matrix_display(conf_matrix, model_name, font_size=12):
    plt.rcParams.update({'font.size': font_size})
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['0', '1'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
plt.show()

# Split data into features (X) and target (y)
hX = Health_with_diabetes.drop(columns=['Diabetes_012'])
hy = Health_with_diabetes['Diabetes_012']

# Split data into training and testing sets
hX_train, hX_test, hy_train, hy_test = train_test_split(hX, hy, test_size=0.3, random_state=3189)

def train_fcn_h(model, hX_train, hX_test, hy_train, hy_test):
    # Train the model
    model.fit(hX_train, hy_train)
    
    # Predict on the test set
    hy_pred = model.predict(hX_test)
    
    # Evaluate the model
    accuracy = accuracy_score(hy_test, hy_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    precision = precision_score(hy_test, hy_pred)
    print(f"Precision: {precision:.2f}")
    
    f1 = f1_score(hy_test, hy_pred)
    print(f"F1 Score: {f1:.2f}")
    
    # Compute and print confusion matrix
    cm = confusion_matrix(hy_test, hy_pred)
    plot_confusion_matrix_display(cm, model.__class__.__name__, font_size = 14)

lX = Lifestyle_with_diabetes.drop(columns=['Diabetes_012'])
ly = Lifestyle_with_diabetes['Diabetes_012']

# Split data into training and testing sets
lX_train, lX_test, ly_train, ly_test = train_test_split(lX, ly, test_size=0.3, random_state=3189)
                                                    
def train_fcn_l(model, lX_train, lX_test, ly_train, ly_test):
    # Train the model
    model.fit(lX_train, ly_train)
    
    # Predict on the test set
    ly_pred = model.predict(lX_test)
    
    # Evaluate the model
    accuracy = accuracy_score(ly_test, ly_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    precision = precision_score(ly_test, ly_pred)
    print(f"Precision: {precision:.2f}")
    
    f1 = f1_score(ly_test, ly_pred)
    print(f"F1 Score: {f1:.2f}")
    
    # Compute and print confusion matrix
    cm = confusion_matrix(ly_test, ly_pred)
    plot_confusion_matrix_display(cm, model.__class__.__name__, font_size = 14)
BX = BMI_with_diabetes.drop(columns=['Diabetes_012'])
By = BMI_with_diabetes['Diabetes_012']

# Split data into training and testing sets
BX_train, BX_test, By_train, By_test = train_test_split(BX, By, test_size=0.3, random_state=3189)
                                                    
def train_fcn_B(model, BX_train, BX_test, By_train, By_test):
    # Train the model
    model.fit(BX_train, By_train)
    
    # Predict on the test set
    By_pred = model.predict(BX_test)
    
    # Evaluate the model
    accuracy = accuracy_score(By_test, By_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    precision = precision_score(By_test, By_pred)
    print(f"Precision: {precision:.2f}")
    
    f1 = f1_score(By_test, By_pred)
    print(f"F1 Score: {f1:.2f}")
    
    # Compute and print confusion matrix
    cm = confusion_matrix(By_test, By_pred) 
    plot_confusion_matrix_display(cm, model.__class__.__name__, font_size = 14)




# Train and evaluate models for Health_with_diabetes
print("Health_with_diabetes:")
for model in [LogisticRegression(random_state=3189), RandomForestClassifier(random_state=3189),DecisionTreeClassifier(random_state=3189)]:
    print(f"Model: {model.__class__.__name__}")
    
    train_fcn_h(model, hX_train, hX_test, hy_train, hy_test)
# Train and evaluate models for Lifestyle_with_diabetes
print("\nLifestyle_with_diabetes:")
for model in [LogisticRegression(random_state=3189), RandomForestClassifier(random_state=3189),DecisionTreeClassifier(random_state=3189)]:
    print(f"Model: {model.__class__.__name__}")
    train_fcn_l(model, lX_train,lX_test, ly_train, ly_test)
# Train and evaluate models for BMI_with_diabetes
print("\nBMI_with_diabetes:")
for model in [LogisticRegression(random_state=3189), RandomForestClassifier(random_state=3189),DecisionTreeClassifier(random_state=3189)]:
    print(f"Model: {model.__class__.__name__}")
    train_fcn_B(model, BX_train,BX_test, By_train, By_test)

# Function to train Random Forest and plot feature importance
def train_random_forest(df):
    # Split data into features (X) and target (y)
    X = df.drop(columns=['Diabetes_012'])
    y = df['Diabetes_012']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Plot feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    sorted_idx = feature_importance.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance - Random Forest')
    plt.show()

# Train Random Forest and plot feature importance for Health_with_diabetes
print("Feature Importance for Health_with_diabetes:")
train_random_forest(Health_with_diabetes)

# Train Random Forest and plot feature importance for Lifestyle_with_diabetes
print("\nFeature Importance for Lifestyle_with_diabetes:")
train_random_forest(Lifestyle_with_diabetes)




INS = pd.read_csv("insurance.csv")
# Summary and column names
print(INS.describe())
print(INS.columns)

# Remove NAs
INS = INS.dropna()
INS[['sex', 'smoker', 'region']] = INS[['sex', 'smoker', 'region']].astype('category')
INS.dtypes
label = LabelEncoder()
label.fit(INS.sex.drop_duplicates())
INS.sex = label.transform(INS.sex)
label.fit(INS.smoker.drop_duplicates())
INS.smoker = label.transform(INS.smoker)
label.fit(INS.region.drop_duplicates())
INS.region = label.transform(INS.region)
INS.dtypes

# Correlation plot
cor_matrix_INS = INS.corr()
plt.figure(figsize=(5, 5))
sns.heatmap(cor_matrix_INS , annot=True, linewidths=2,cmap='coolwarm', fmt=".2f",annot_kws={"size": 11})
plt.show()
X = INS.drop(columns=['charges'])  # Predictor variables excluding BMI and Outcome
y = INS['charges']  # Target variable
# Standardize predictor variables"
X_scaled = StandardScaler().fit_transform(X)

# Variance Inflation Factor (VIF)
vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

plt.figure(figsize=(7,4))
sns.barplot(x="Feature", y="VIF", data=vif)
plt.title("Variance Inflation Factor (VIF)")
plt.xticks(rotation=45, fontsize =12)
plt.show()
X = INS.drop(columns=['charges']) # Independent variable
y = INS['charges']     # Dependent variable

# Add constant to the independent variable (for intercept)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())
# Setup the regression training environment
target_rgs = INS['charges']
data_rgs= StandardScaler().fit_transform(INS.drop(columns=['charges','sex']))

X_train, X_test, y_train, y_test = train_test_split(data_rgs, target_rgs, test_size=0.3, random_state=3187)


# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_pred = linear_reg.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

print("Linear Regression Results:")
print(f"MSE: {linear_mse}")
print(f"RMSE: {linear_rmse}")
print(f"MAE: {linear_mae}")
print(f"R-squared: {linear_r2}")
print()
# Lasso Regression
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
lasso_pred = lasso_reg.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("Lasso Regression Results:")
print(f"MSE: {lasso_mse}")
print(f"RMSE: {lasso_rmse}")
print(f"MAE: {lasso_mae}")
print(f"R-squared: {lasso_r2}")
# Ridge Regression
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)
ridge_pred = ridge_reg.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("Ridge Regression Results:")
print(f"MSE: {ridge_mse}")
print(f"RMSE: {ridge_rmse}")
print(f"MAE: {ridge_mae}")
print(f"R-squared: {ridge_r2}")
print()
# Random Forest Regression
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Regression Results:")
print(f"MSE: {rf_mse}")
print(f"RMSE: {rf_rmse}")
print(f"MAE: {rf_mae}")
print(f"R-squared: {rf_r2}")
print()
