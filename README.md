# Elevate-HR-Attirtion-Project

## USED PYTHON 
### importing the required pyhton libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Dataset
df = pd.read_csv("Hr Project.csv")

# Intial Exploration
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['Attrition'].value_counts())


# Exploratory Data Analysis

# Attrition Count
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Count")
plt.show()

# Attrition by Department
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title("Attrition by Department")
plt.show()

# Attrition by Overtime
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("Attrition by Overtime")
plt.show()

#Data Preprocessing

# Encode Attrition (Yes/No → 1/0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop unnecessary columns
df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], inplace=True, errors='ignore')

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Split the Data

from sklearn.model_selection import train_test_split

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Test the Classification Model

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Evaluate Model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# SHAP for Model Explainability

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SUMMARY PLOT
shap.summary_plot(shap_values, X_test)


# Explored Cleaned Data for Power BI

df.to_csv("HR_Cleaned.csv", index=False)


# TOOLS USED:
.Python: For analyzing and modeling the data

.Power BI: For creating interactive visual dashboards

# PROCEDURES USED FOR COMPLETING THE PROJECT

1.Loaded the HR dataset in Python using Pandas

2.Explored the data – looked at attrition by department, salary, overtime, etc.

3.Cleaned the data – removed unnecessary columns, handled categories, and converted labels

4.Built a machine learning model (Decision Tree) to predict attrition

5.Evaluated the model using accuracy and confusion matrix

6.Explained the predictions with SHAP to see which factors mattered most

7.Exported the cleaned data to a CSV file

8.Used Power BI to visualize attrition trends and factors






