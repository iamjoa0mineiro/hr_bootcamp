# ===========================================
# SECTION 1: Import needed libraries
# ===========================================
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import RFE # wrapper method
from sklearn.linear_model import LogisticRegression #(This is one possible model to apply inside RFE)
from sklearn.linear_model import LassoCV # embedded method
from sklearn.tree import DecisionTreeClassifier # embedded method
from sklearn.model_selection import StratifiedKFold

# =================================================
# SECTION 2: Data Collection and Initial Processing
# =================================================

# ------------------------------------------
# 2.1. Dataset Overview
# ------------------------------------------
hr = pd.read_csv('HR_Attrition_Dataset.csv')
hr.head()

# ------------------------------------------
# 2.2. Data Description
# ------------------------------------------
data_types = hr.info()
data_types

data_description = hr.describe().transpose() 
data_description

# ------------------------------------------
# 2.3. Data Preprocessing
# ------------------------------------------

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

# ------------------------------------------
# 3.1. Visualization
# ------------------------------------------

# 3.1.1. Company's Age Analysis

# 3.1.1.1 Checking the summary of Ages
age_summary = hr['Age'].describe()
print(age_summary)

plt.figure(figsize=(8, 6))
sns.histplot(hr['Age'], kde=True, bins=15)
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 3.1.1.2 Percentage of Each Age Group in the Dataset
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

# ------------------------------------------
# 3.2. Univariate Data Analysis 
# ------------------------------------------

# Describing the data: Identifying measures like the mean, median, mode, variance, standard deviation, and range.
hr.describe()

# DROP EmployeeCount! Checking unique values of EmployeeCount. It's a constant column, it can be dropped.
print(hr['EmployeeCount'].unique())

# SET ID as the EmployeeNumber! Checking unique values of EmployeeNumber. It has 1470 distinct values, has no meaning for our analysis. Hence, it should be dropped.
print(hr['EmployeeNumber'].nunique())
hr.set_index('EmployeeNumber', inplace=True)

# DROP StandardHours! Checking unique values. It's a constant column.
print(hr['StandardHours'].unique())

# DROPPING COLUMNS
hr.drop(['EmployeeCount','StandardHours'], axis=1, inplace=True)

# HISTOGRAMS - No outliers were identified
# Select numerical columns
numerical_columns = hr.select_dtypes(include=np.number).columns
# Calculate the number of rows and columns needed for the subplots
num_cols = 7
num_rows = int(np.ceil(len(numerical_columns) / num_cols))
# Initialize the subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, num_rows * 5))
# Flatten the axis array for easy iteration
axes = axes.flatten()
# Loop through each column and plot a histogram
for i, column in enumerate(numerical_columns):    
    # Add the histogram    
    hr[column].hist(ax=axes[i], # Axis definition                  
                            edgecolor='white', # Color of the border                    
                            color='#AD60B8' # Color of the bins                   
                           )    
    # Add title and axis label    
    axes[i].set_title(f'{column} Histogram')    
    axes[i].set_xlabel(column)    
    axes[i].set_ylabel('Frequency')
# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
# Adjust layout
plt.tight_layout()
# Show the plot
plt.show()

#BOXPLOT - Appears to have outliers on MonthlyIncome, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
# Select numerical columns
numerical_columns = hr.select_dtypes(include=np.number).columns
# Calculate the number of rows and columns needed for the subplots
num_cols = 7
num_rows = int(np.ceil(len(numerical_columns) / num_cols))
# Initialize the subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, num_rows * 5))
# Flatten the axis array for easy iteration
axes = axes.flatten()
# Loop through each column and plot a box plot
for i, column in enumerate(numerical_columns):    
    # Add the box plot    
    hr.boxplot(column=column, ax=axes[i], 
               patch_artist=True,  # Use patch_artist to fill the box with color
               boxprops=dict(facecolor='#AD60B8', color='black'))  # Box fill and edge color
    
    # Add title and axis label
    axes[i].set_title(f'{column} Box Plot')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Values')
# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
# Adjust layout
plt.tight_layout()
plt.show()

#Checking outliers with z-score in: MonthlyIncome, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

#Create dataframe with the z-scores of the columns above
variables = ['MonthlyIncome', 'TotalWorkingYears', 'TrainingTimesLastYear',
             'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
             'YearsWithCurrManager']
hr_zscores = pd.DataFrame()
for column in variables:
    hr_zscores[column + '_zscore'] = stats.zscore(hr[column])

#Verify z-scores between -3 and 3
hr_zscores.describe()

#From the describe function columns: TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager have outliers.
outliers = hr_zscores[(hr_zscores > 3) | (hr_zscores < -3)].dropna(how='all')
#We have 83 outliers according to this, we'll remove them.
outlier_employee_numbers = hr_zscores[(hr_zscores > 3) | (hr_zscores < -3)].dropna(how='all').index
hr_nout = hr.drop(index=outlier_employee_numbers)

# ==========================================
# 3.3. Bivariate Data Analysis
# ==========================================

# 3.3.1 Attrition vs. Ordinal Variables

#Create function for the plots
def perc_barplots(column):
    attrition = hr.groupby([column, 'Attrition']).size().reset_index(name='Count')
    # First, calculate the sum of 'Count' for each value of column
    total_count_by_status = attrition.groupby(column)['Count'].transform('sum')
    # Then, calculate the percentage for each row
    attrition['Percentage'] = attrition['Count'] / total_count_by_status * 100
    #Only need the percentage for the attrition and can drop columns not need. Also sorted by percentage for the plot
    attrition = attrition[attrition['Attrition'] == 1].drop(columns=['Attrition','Count']).sort_values(by='Percentage')
    #Compilate plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=attrition, x=column, y='Percentage', palette='viridis')
    plt.title(f'% of Attrition per {column}')
    plt.xlabel(column)
    plt.xticks(rotation=45)
    plt.ylabel('% of Attrition')
    plt.show()

# Attrition vs. Marital Status
perc_barplots('MaritalStatus')
# Insight: We can check that single employees have a higher likelihood of attrition compared to those who are married or divorced

# Attrition vs. Business Travel
perc_barplots('BusinessTravel')
# Insight: Employees who travel frequently show a significant amount of attrition compared to those who rarely or do not travel at all

# Attrition vs. Overtime
perc_barplots('OverTime')
# Insight: Employees who work overtime show a substantially higher level of attrition compared to those who do not

# Attrition vs. JobRole
perc_barplots('JobRole')
# Insight: Sales Representative, Laboratory Technician, Research Scientist and Sales Executive show a high count of employees leaving, while Managers and Research Directors show lower attrition rates
# Insight: Based on above analysis, MaritalStatus, BusinessTravel, Overtime and JobRole might be important features for the model

# 3.3.2 Chi-Square Test for All Ordinal Variables

ordinal_cols = [
    "BusinessTravel", "Department", "Education", "EducationField",
    "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel",
    "JobRole", "JobSatisfaction", "MaritalStatus", "OverTime",
    "PerformanceRating", "RelationshipSatisfaction", "WorkLifeBalance"
]

# Iterate through all ordinal variables and calculate chi-square
for col in ordinal_cols:
    # Create a contingency table
    contingency_table = pd.crosstab(hr[col], hr['Attrition'])
    # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Print the results in scientific notation for the p-value
    print(f"Chi-square Statistic for {col}: {chi2:.2f}, p-value: {p:.2e}")
    if p < 0.05:
        print(f"Significant relationship between {col} and Attrition\n")
    else:
        print(f"No significant relationship between {col} and Attrition\n")

# Insight: OverTime, JobRole, JobLevel and MaritalStatus are the most significant predictors, with OverTime standing out as particularly important
# Education, Gender, PerformanceRating and RelationshipSatisfaction doesn't have a significant impact

# 3.3.3 Attrition vs. Numerical Variables (using Boxplots)

# Attrition vs. Monthly Income
plt.figure(figsize=(12, 6))
sns.boxplot(data=hr, x='Attrition', y='MonthlyIncome', palette='coolwarm')
plt.title('Monthly Income by Attrition Status')
plt.xlabel('Attrition')
plt.ylabel('Monthly Income')
plt.show()
# Insight: Employees who left the company tend to have lower monthly income
# There are much more outliers for "No Attrition" indicating that higher monthly incomes might encourage employees to stay

# Attrition vs. Years at Company
plt.figure(figsize=(12, 6))
sns.boxplot(data=hr, x='Attrition', y='YearsAtCompany', palette='coolwarm')
plt.title('Years at Company by Attrition Status')
plt.xlabel('Attrition')
plt.ylabel('Years at Company')
plt.show()
# Insight: Employees who left the company tend to have a shorter period at the company compared to those who stayed
# Employees who left have spent fewer years (typically less than 5) at the company

# Attrition vs. Total Working Years
plt.figure(figsize=(12, 6))
sns.boxplot(data=hr, x='Attrition', y='TotalWorkingYears', palette='coolwarm')
plt.title('Total Working Years by Attrition Status')
plt.xlabel('Attrition')
plt.ylabel('Total Working Years')
plt.show()
# Insight: Employees with a longer career history tend to stay and those with fewer total working years are more likely to leave

# 3.3.3 Pairwise Plots for Numerical Variables

sns.pairplot(hr, hue='Attrition', vars=['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'], palette='viridis')
plt.show()

# 3.3.4 Correlation Analysis

# Select only the numeric columns from the DataFrame
hr_numeric = hr.select_dtypes(include=['number'])

plt.figure(figsize=(16, 12))
sns.heatmap(hr_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of All Variables')
plt.show()
# Feature Selection - Given the high correlation:
# JobLevel and MonthlyIncome (0.95): Exclude MonthlyIncome
# YearsInCurrentRole and YearsAtCompany (0.79): Exclude YearsAtCompany
# Check the others and decide together 

# 3.3.5 Feature Selection

#Each test applied will be ran three times

skf = StratifiedKFold(n_splits = 3, random_state = 99, shuffle = True)

#To check which variables to remove from the Correlation Analysis we'll apply a Decision Tree to see feature importance

X = hr_numeric.drop('Attrition', axis = 1)
y = hr_numeric['Attrition'].copy()

def plot_importance(variables,name):
    imp_features = variables.sort_values()
    plt.figure(figsize=(4,5))
    imp_features.plot(kind = "barh")
    plt.title("Feature importance using " + name + " Model")
    plt.show()

def apply_dt(X_train, y_train):
    dt = DecisionTreeClassifier(random_state = 99).fit(X_train, y_train)
    feature_importances = pd.Series(dt.feature_importances_, index = X_train.columns)
    plot_importance(feature_importances, 'DT')

def select_best_features_dt(X, y):
    count = 1
    for train_index, val_index in skf.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        ######################################### SELECT FEATURES #################################################
        print('_________________________________________________________________________________________________\n')
        print('                                     SPLIT ' + str(count) + '                                    ')
        print('_________________________________________________________________________________________________')

        # check which features to use using decision Tree
        apply_dt(X_train, y_train)

        count+=1


select_best_features_dt(X, y)

#According to the three splits, the variables that should be removed are JobLevel and YearsInCurrentRole as these have less importance considering their counterparts

X = X.drop(['JobLevel','YearsInCurrentRole'], axis = 1)

#Now we'll run three tests for feature selection: RFE, Lasso and Decision Trees:
def apply_rfe(X_train, y_train):
    rfe = RFE(estimator = LogisticRegression(), n_features_to_select = 5)
    rfe.fit_transform(X = X_train, y = y_train)
    selected_features = pd.Series(rfe.support_, index = X_train.columns)
    print(selected_features)

def apply_lasso(X_train, y_train):
    lasso = LassoCV().fit(X_train, y_train)
    coef = pd.Series(lasso.coef_, index = X_train.columns)
    plot_importance(coef,'Lasso')

def select_best_features(X,y):
    count = 1
    for train_index, val_index in skf.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        ########################################### SCALE DATA ####################################################
        scaler = MinMaxScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)

        ######################################### SELECT FEATURES #################################################
        print('_________________________________________________________________________________________________\n')
        print('                                     SPLIT ' + str(count) + '                                    ')
        print('_________________________________________________________________________________________________')

        # Check which features to use using RFE
        print('')
        print('----------------- RFE ----------------------')
        apply_rfe(X_train_scaled, y_train)

        # check which features to use using lasso
        print('')
        print('----------------- LASSO ----------------------')
        apply_lasso(X_train_scaled, y_train)

        # check which features to use using lasso
        print('')
        print('----------------- DT ----------------------')
        apply_dt(X_train_scaled, y_train)

        count+=1

#Given the information on this test:
# Remove: Education, PerformanceRating, RelationshipSatisfaction and YearsAtCompany
# Test with and without: HourlyRate, JobSatisfaction, MonthlyRate, PercentSalaryHike, TrainingTimesLastYear, WorkLifeBalance
#The other numerical and categorical variables, KEEP. 




# -----------------------------------------------------------------
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

# ==========================================
# SECTION 3.4: Feature Engineering
# ==========================================

