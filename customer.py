import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
customer = pd.read_csv('Telco-Customer-Churn.csv')
customer.head()
customer.info() 

#data cleaning
#forcing converts to numeric
customer['TotalCharges'] = pd.to_numeric(customer['TotalCharges'], errors='coerce')
#checking for null values
print(f"Nulls in TotalCharges: {customer['TotalCharges'].isnull().sum()}")
#filling nulls with 0 (since they are new customers) or completely drop them
customer['TotalCharges'].fillna(0, inplace=True)
#dropping 'customerID' as it has no predictive power
customer.drop('customerID', axis=1, inplace=True)
customer

#eda
#visualising churn count
sns.countplot(x='Churn', data=customer)
plt.title("Churn Class Distribution")
plt.show()
#visualise churn by contract type
sns.countplot(x='Contract', hue='Churn', data=customer)
plt.title("Churn by Contract Type")
plt.show()

#preprocessing
#binary encoding for the target
customer['Churn'] = customer['Churn'].map({'Yes': 1, 'No': 0})
#one hot encoding for other categoricals
customer_encoded = pd.get_dummies(customer, drop_first=True) 

#model building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#splitting data
x = customer_encoded.drop('Churn', axis=1)
y = customer_encoded['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#training the baseline model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred)) 

from imblearn.over_sampling import SMOTE
from collections import Counter

#standard splitting
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

#checking counts before SMOTE
print(f"Before SMOTE: {Counter(y_train)}")

#initialising SMOTE
smote = SMOTE(random_state=42)

#fitting SMOTE on the training data
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#checking counts after SMOTE
print(f"After SMOTE: {Counter(y_train_resampled)}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#initializing model
rf_model = RandomForestClassifier(random_state=42)
#training on the balanced SMOTE data
rf_model.fit(x_train_resampled, y_train_resampled)
#prediction on the original (unlanaced) data
y_pred = rf_model.predict(x_test)
#results
print(classification_report(y_test, y_pred)) 

#getting the importance scores, and feature names
importances = rf_model.feature_importances_
feature_names = x.columns

#creating a df to ogarnize them
feature_importance_customer = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

#sorting by importance (highest on top)
feature_importance_customer = feature_importance_customer.sort_values(by='Importance', ascending=False)

#plotting the top 10 features
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature',
             data=feature_importance_customer.head(10),
             palette='viridis')
plt.title('Top 10 Drivers of Customer Churn')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show() 

from sklearn.inspection import permutation_importance

#calculaing permutaion importance
#n_repeats=10: running the shufling 10 times to ensure the result isn't just luck
#n_jobs=-1: use all CPU cores to make it faster
result = permutation_importance(
    rf_model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
) 

#organising the data for plotting
perm_importance_customer = pd.DataFrame({
    'Feature': x_test.columns,
    'Importance_Mean': result.importances_mean,
    'Importance_Std': result.importances_std
}) 

#sorting features by impact
perm_importance_customer = perm_importance_customer.sort_values(by='Importance_Mean', ascending=False)

#visuals
plt.figure(figsize=(10,8))
sns.barplot(x='Importance_Mean', y='Feature',
            data=perm_importance_customer.head(15),
            palette='magma'
) 

#adding errors to show stability
plt.errorbar(
    x=perm_importance_customer.head(15)['Importance_Mean'],
    y=perm_importance_customer.head(15)['Feature'],
    xerr=perm_importance_customer.head(15)['Importance_Std'],
    fmt='none', c='black', capsize=3
)
plt.title('Permutation Impotance: What really drives Churn')
plt.xlabel('Decrease in Model Accuracy')
plt.ylabel('Feature')
plt.show() 

from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
#cross-validation
#defining operations 
#'smote': unsampling training fold
#'model': training the classifier
pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

#defining spliting strategy
#stratifiedKFold ensures each fold has the same % of Churners as the original data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#running cross-validation
#using F1-Score because accuracy is misleading for imbalanced data
cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring='f1')
print(f"F1 Scores for each fold: {cv_scores}")
print(f"Average F1 Score: {cv_scores.mean():.4f}")

from sklearn.model_selection import GridSearchCV
#defining parameter grid
#test running number of numbers of tress, and different depths
param_grid= {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
} 

#setting up grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
print("Searching Grid Search.. (this might be slow)")
grid_search.fit(x_train, y_train) 

print("Best F1 Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)

#prediction using the best model found
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(x_test)

#comparing with first result gotten
from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred_optimized)) 

#visualizing results 
#confusion matrix heatmap
#generating matrix
cm = confusion_matrix(y_test, y_pred_optimized)
cm_percent = cm.astype('float') / cm.sum()
plt.figure(figsize=(8,6))
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', cbar=False,
            xticklabels=['Predicted: Stay', 'Predicted: Churn'],
            yticklabels=['Actual: Stay', 'Actual: Churn'])
plt.title('Confusion Matrix: Optimized Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show() 