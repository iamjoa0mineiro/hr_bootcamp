# ===========================================
# SECTION 1: Import needed libraries
# ===========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =================================================
# SECTION 2: Data Collection and Initial Processing
# =================================================

# 2.1. Dataset Overview
hr = pd.read_csv('HR_Attrition_Dataset.csv')
hr.head()

# 2.2. Data Description
data_types = hr.info()
data_types

data_description = hr.describe().transpose() 
data_description

# 2.3. Data Preprocessing

# 2.3.1. This code let us have an idea of the unique values of each column. It is also possible to verify that there are no blanks.

columns = list(hr.columns)

for col in columns:
    print(f"{col} unique values: ", f"{hr[col].unique()} \n")

# 2.3.2. Since all the employees are over 18 years old, this variable proves to be useless for our analysis
hr.drop("Over18", axis=1, inplace=True)

# 2.3.3. Changing our target variable from categorical to numeric
hr['Attrition'] = hr['Attrition'].apply(lambda x: 1 if x=="Yes" else 0)

# 2.3.4. Create a new variable to group employees by Age Groups
def age_group(age):
    if age < 31:
        return "Young Adults"
    elif age < 51:
        return "Mid Age"
    else:
        return "Elderly"

hr['AgeGroup'] = hr['Age'].apply(age_group)

# ==========================================
# SECTION 3: Exploratory Data Analysis (EDA)
# ==========================================

# 3.1. Visualization
# 3.1.1. Company's Age Analysis
age_group_percentage = hr['AgeGroup'].value_counts(normalize=True) * 100

age_group_df = age_group_percentage.reset_index()
age_group_df.columns = ['AgeGroup', 'Percentage']

plt.figure(figsize=(8, 6))
ax = sns.barplot(x='AgeGroup', y='Percentage', data=age_group_df, order=['Young Adults', 'Mid Age', 'Elderly'])
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%',  
                (p.get_x() + p.get_width() / 2., p.get_height()),  
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.title('Percentage of Each Age Group in the Dataset')
plt.ylabel('Percentage')
plt.xlabel('Age Group')

plt.show()

# INSIGHT: The majority of company's workers (64.01 %) is in the Mid Age group (i.e. [31-50] years old)

# 3.1.2. Employees by Marital Status
marital_status_percentage = hr['MaritalStatus'].value_counts(normalize=True).round(2) * 100

marital_status_df = marital_status_percentage.reset_index()
marital_status_df.columns = ['MaritalStatus', 'Percentage']

plt.figure(figsize=(8, 6))
sns.barplot(y='MaritalStatus', x='Percentage', data=marital_status_df, orient='h')

for index, value in enumerate(marital_status_df['Percentage']):
    plt.text(value, index, f'{value:.2f}%', va='center') 

plt.title('Percentage Distribution of Marital Status')
plt.ylabel('Marital Status')
plt.xlabel('Percentage')
plt.show()

# 3.1.3. Nº Employees by Department
department_count = hr['Department'].value_counts()

plt.figure(figsize=(10, 7)) 

squarify.plot(sizes=department_count, 
              label=department_count.index, 
              value=department_count.values, 
              color=sns.color_palette('pastel')[0:len(department_count)], 
              edgecolor='black', 
              pad=True)

plt.title('Percentage Distribution of Departments')
plt.show()

# INSIGHT: The majority of the company's employees belong to the R&D department (961 people), followed by Sales (446 people) and HR (63 people)

# 3.1.4. Attrition Evaluation
attrition_count = hr['Attrition'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=attrition_count.index, y=attrition_count.values, palette='pastel')

for i, value in enumerate(attrition_count.values):
    plt.text(i, value + 5, str(value), ha='center')

plt.title('Attrition Distribution in the Company')
plt.xlabel('Attrition')
plt.ylabel('Number of Employees')
plt.show()

# INSIGHT: This dataset includes a sample of 237 employees who left the company out of a total of 1,470 employees
# Our main goal is to use machine learning algorithms to predict employee attrition and understand the reasons behind it in order to improve the company's KPIs

# 3.2. Univariate Data Analysis 








# 3.3. Bivariate Data Analysis

# First, we need to convert our categorical data into numeric data

vars_to_convert = [
    "BusinessTravel", 
    "Department", 
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "AgeGroup"
]

hr_temp = hr[vars_to_convert]

hr_dummies = pd.get_dummies(hr_temp, drop_first=True, dtype=int)

hr.drop(vars_to_convert, axis=1, inplace=True)

hr_new = hr.merge(how='left', left_index=True, right_index=True, validate='one_to_one', right=hr_dummies)

# Correlation Matrix
corr_matrix = hr_new.corr()
corr_matrix = corr_matrix[(corr_matrix > 0.7) | (corr_matrix < -0.7)]
sns.heatmap(corr_matrix)
plt.show()

# INSIGHTS: 







# hr_new is now the most updated DataFrame with numeric data only - which allows to run several Data Science & Machine Learning algorithms

# Before we proceed to the feature engeneering stage, there are still 3 crucial steps we need to take:
# 1st - Split the Target variable from the Predictor variables
# 2nd - Split the data into Train and Test sets
# 3rd - Scale our Predictor variables

# 1st: Split the Target variable from the Predictor variables
# Target Variable
y = hr_new['Attrition']
# Predictor Variables
X = hr_new.drop('Attrition',axis=1)

# 2nd: Split the data into Train and Test sets (30% for testing, 70% for trainning)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# 3rd: Scale our Predictor variables
scaler = MinMaxScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Our data is now ready for machine learning to be applied!

# 3.4. Feature Engeneering

