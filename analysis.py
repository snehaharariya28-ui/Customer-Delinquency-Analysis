# Project Objective:
# This project aims to analyze a customer dataset to identify patterns and key factors that contribute to account delinquency. By examining variables such as income, credit score, loan balance, payment history, and credit utilization, the goal is to understand which types of customers are more likely to default on their payments. The analysis involves data cleaning, exploratory data analysis, and visualization to uncover relationships between customer behavior and delinquency. The expected outcome is to generate actionable insights that can help financial institutions identify high-risk customers early and make better data-driven decisions for risk management and customer segmentation.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("delinquency_prediction_dataset.xlsx")

# understanding the dataset : 

# 1. First 5 rows
print("First 5 rows:\n", df.head())

# 2. Column names
print("\nColumns:\n", df.columns)

# 3. Data types
print("\nData Info:\n")
print(df.info())

# 4. Summary stats
print("\nSummary:\n", df.describe())

# 5. Missing values
print("\nMissing values:\n", df.isnull().sum())

# ------------------------------------------------------------------------------------------
# Data Cleaning
# In this step, we are fixing problems in the dataset so that our analysis becomes accurate. Some columns like Income, Credit Score, and Loan Balance have missing values, so instead of deleting those rows (which would reduce our data), we fill the missing values with the average (mean) of that column. This is a common and simple method. We also remove the Customer_ID column because it is just an identifier and has no impact on whether a customer is delinquent or not. So overall, this step makes the data usable and reliable for analysis.

df['Income'].fillna(df['Income'].mean(), inplace=True)
df['Credit_Score'].fillna(df['Credit_Score'].mean(), inplace=True)
df['Loan_Balance'].fillna(df['Loan_Balance'].mean(), inplace=True)

df.drop('Customer_ID', axis=1, inplace=True)

# ------------------------------------------------------------------------------------------
# Delinquency distribution
# Here, we are trying to understand the target variable — Delinquent_Account. The first line counts how many customers are delinquent (1) and non-delinquent (0). The second line converts this into percentages. This helps us understand the dataset balance, like whether most customers are safe or risky. It gives a quick overview of how serious the delinquency problem is in the dataset before we start deeper analysis.

print("\nDelinquency Count:\n", df['Delinquent_Account'].value_counts())
print("\nDelinquency Percentage:\n", df['Delinquent_Account'].value_counts(normalize=True))

# ------------------------------------------------------------------------------------------
# Income vs Delinquency
# In this step, we are visually comparing important features like Income and Credit Score with delinquency status. The boxplots help us see differences between delinquent (1) and non-delinquent (0) customers. For example, if the box for delinquent customers is lower in income, it means low-income customers are more likely to default. Similarly, we check credit scores. This step is important because graphs make patterns very clear and help us generate insights easily, which is the main goal of a data analyst.

sns.boxplot(x='Delinquent_Account', y='Income', data=df)
plt.title("Income vs Delinquency")
plt.show()                                                    # Income does not show a significant difference between delinquent and non-delinquent customers, indicating that income alone is not a strong predictor of delinquency.


# Credit Score vs Delinquency
sns.boxplot(x='Delinquent_Account', y='Credit_Score', data=df)
plt.title("Credit Score vs Delinquency")
plt.show()                                                                 # Difference is not very strong Credit score alone is not clearly separating the groups

# ------------------------------------------------------------------------------------------
# Count number of "Missed" payments per customer

months = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
df['Missed_Count'] = df[months].apply(lambda row: (row == 'Missed').sum(), axis=1)

# Average missed payments for each group
print (" ------- ")
print(df.groupby('Delinquent_Account')['Missed_Count'].mean())      # On average, how many times do delinquent customers miss payments compared with non-delinquent customers?

# Visualize
sns.boxplot(x='Delinquent_Account', y='Missed_Count', data=df)
plt.title("Missed Payments vs Delinquency")
plt.show()

# ------------------------------------------------------------------------------------------
# lets analyze: Credit Utilization + Debt Ratio

# Credit Utilization vs Delinquency
sns.boxplot(x='Delinquent_Account', y='Credit_Utilization', data=df)
plt.title("Credit Utilization vs Delinquency")
plt.show()

# Debt to Income Ratio vs Delinquency
sns.boxplot(x='Delinquent_Account', y='Debt_to_Income_Ratio', data=df)
plt.title("Debt to Income Ratio vs Delinquency")
plt.show()

# ------------------------------------------------------------------------------------------
# Create Risk Category : 

# Create risk category based on credit score

# EXPLANATION : This code converts credit scores into categories to make analysis easier. The pd.cut() function divides credit scores into ranges: 300–600 = High Risk, 600–750 = Medium Risk, and 750–900 = Low Risk. Each customer gets a label based on the range their score falls into, and this is stored in the new column Risk_Category. This helps us analyze delinquency by groups instead of raw numbers, which gives clearer insights.

df['Risk_Category'] = pd.cut(df['Credit_Score'],
                            bins=[300, 600, 750, 900],
                            labels=['High Risk', 'Medium Risk', 'Low Risk'])                        


# Analyze 
print(df.groupby('Risk_Category')['Delinquent_Account'].mean())

# Plot
sns.barplot(x='Risk_Category', y='Delinquent_Account', data=df)
plt.title("Risk Category vs Delinquency")
plt.show()

# ------------------------------------------------------------------------------------------