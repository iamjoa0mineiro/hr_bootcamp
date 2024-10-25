# ===========================================
# SECTION 1: Import needed libraries
# ===========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.feature_selection import RFE # wrapper method
from sklearn.linear_model import LogisticRegression # (This is one possible model to apply inside RFE)
from sklearn.linear_model import LassoCV # embedded method
from sklearn.tree import DecisionTreeClassifier # embedded method
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2 
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter


# =================================================
# SECTION 2: Data Collection and Initial Processing
# =================================================

# ------------------------------------------
# 2.1. Dataset Overview
# ------------------------------------------
# Loading the dataset
hr = pd.read_csv('HR_Attrition_Dataset.csv')
hr.head()

# ------------------------------------------
# 2.2. Data Description
# ------------------------------------------
# Information about the DataFrame
data_types = hr.info()

# Descriptive statistics for numerical columns
data_description = hr.describe().transpose() 
print(data_description)

# Checking unique values for categorical variables
# Identify categorical columns using their data types (dtype = object for string values)
categorical_columns = hr.select_dtypes(include='object').columns

# Displaying unique values for each categorical column
for column in categorical_columns:
    print(f"Unique values for '{column}': {hr[column].unique()}")

# ------------------------------------------
# 2.3. Data Preprocessing
# ------------------------------------------

# Dropping Columns with Limited Utility (constant value or irrelevant information)
hr.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# Changing the Target Variable from Categorical to Numeric
hr['Attrition'] = hr['Attrition'].apply(lambda x: 1 if x == "Yes" else 0)

# Creating Age Group Variable and Encoding
def age_group(age):
    if age < 31:
        return "Young Adults"
    elif age < 51:
        return "Mid Age"
    else:
        return "Elderly"

hr['AgeGroup'] = hr['Age'].apply(age_group)

# Encode 'AgeGroup' using ordinal mapping
age_group_mapping = {'Young Adults': 1, 'Mid Age': 2, 'Elderly': 3}
hr['AgeGroup'] = hr['AgeGroup'].map(age_group_mapping)

# Verify the dataset after encoding
print(hr.info())
print(hr.head())

# =================================================
# SECTION 3: Exploratory Data Analysis (EDA)
# =================================================

# ------------------------------------------
# 3.1. Visualization
# ------------------------------------------

# 3.1.1. Company's Age Analysis
# ------------------------------------------
# Calculate the Percentage of Each Age Group in the Dataset
age_group_percentage = hr['AgeGroup'].value_counts(normalize=True) * 100
age_group_df = age_group_percentage.reset_index()
age_group_df.columns = ['AgeGroup', 'Percentage']

# Plotting the Age Group Percentage as a Bar Chart
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='AgeGroup', y='Percentage', data=age_group_df, order=[1, 2, 3], palette='viridis')
ax.set_xticklabels(['Young Adults', 'Mid Age', 'Elderly'])
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.title('Percentage of Each Age Group in the Dataset')
plt.ylabel('Percentage')
plt.xlabel('Age Group')
plt.show()

# ------------------------------------------
# 3.1.2. Percentage of Employees by Marital Status
# ------------------------------------------
# Calculate the percentage for each marital status
marital_status_percentage = hr['MaritalStatus'].value_counts(normalize=True) * 100
marital_status_df = marital_status_percentage.reset_index()
marital_status_df.columns = ['MaritalStatus', 'Percentage']

# Plotting the Percentage of Employees by Marital Status
plt.figure(figsize=(8, 6))
sns.barplot(y='MaritalStatus', x='Percentage', data=marital_status_df, orient='h', palette='Blues')
for index, value in enumerate(marital_status_df['Percentage']):
    plt.text(value, index, f'{value:.2f}%', va='center')
plt.title('Percentage of Employees by Marital Status')
plt.ylabel('Marital Status')
plt.xlabel('Percentage (%)')
plt.show()

# ------------------------------------------
# 3.1.3. Number of Employees by Department
# ------------------------------------------
# Using original categorical values
department_percentage = hr['Department'].value_counts(normalize=True) * 100
department_df = department_percentage.reset_index()
department_df.columns = ['Department', 'Percentage']

# Squarify Plot for Number of Employees by Department
plt.figure(figsize=(10, 6))
squarify.plot(sizes=department_df['Percentage'],
              label=department_df.apply(lambda x: f"{x['Department']}\n{x['Percentage']:.2f}%", axis=1),
              color=sns.color_palette('Pastel2', len(department_df)),
              alpha=0.8,
              pad=True)
plt.title('Percentage Distribution of Departments')
plt.axis('off')
plt.show()

# ------------------------------------------
# 3.1.4. Attrition Evaluation
# ------------------------------------------
attrition_count = hr['Attrition'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=attrition_count.index, y=attrition_count.values, palette='pastel')
for i, value in enumerate(attrition_count.values):
    plt.text(i, value + 5, str(value), ha='center')
plt.title('Attrition Distribution in the Company')
plt.xlabel('Attrition')
plt.ylabel('Number of Employees')
plt.show()

# ==========================================
# SECTION 3.2: Univariate Data Analysis
# ==========================================

# ------------------------------------------
# 3.2.1. Visualize Data Distribution (Histograms)
# ------------------------------------------

# Set the style for better visualization
sns.set_theme(style="whitegrid")

# Drop 'AgeGroup' after it is used and retain 'Age' for future analysis
hr.drop(columns=['AgeGroup'], inplace=True)

# Select Numerical Columns
numerical_columns = hr.select_dtypes(include=np.number).columns

# Determine the Layout for Enhanced Histograms
num_cols = 5  # Reduced the number of columns for better plot size and spacing
num_rows = int(np.ceil(len(numerical_columns) / num_cols))
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(25, num_rows * 5))
axes = axes.flatten()

# Plotting Enhanced Histograms for Each Numerical Column
for i, column in enumerate(numerical_columns):
    sns.histplot(data=hr, x=column, ax=axes[i], bins=20, color='#7E57C2', edgecolor='black')
    axes[i].set_title(f'{column} Distribution', fontsize=14, weight='bold')
    axes[i].set_xlabel(column, fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(True, linestyle='--', alpha=0.6)

# Remove unused subplots for clarity
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust Layout and Display the Enhanced Histograms
plt.tight_layout()
plt.show()

# ------------------------------------------
# 3.2.2. Boxplot Analysis for Outlier Detection
# ------------------------------------------

# Update numerical columns after dropping 'AgeGroup'
numerical_columns = hr.select_dtypes(include=np.number).columns

# Determine the Layout for Enhanced Boxplots
num_cols = 5  # Reduce the number of columns for better spacing
num_rows = int(np.ceil(len(numerical_columns) / num_cols))
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(25, num_rows * 5))
axes = axes.flatten()

# Loop Through Numerical Columns and Plot Enhanced Boxplots
for i, column in enumerate(numerical_columns):
    sns.boxplot(data=hr, y=column, ax=axes[i], color='#7E57C2', fliersize=5, linewidth=1.5)  # Updated box color to '#FF7043'
    axes[i].set_title(f'{column} Box Plot', fontsize=14, weight='bold')
    axes[i].set_xlabel(column, fontsize=12)
    axes[i].set_ylabel('Values', fontsize=12)
    axes[i].grid(axis='y', linestyle='--', alpha=0.6)

# Remove unused subplots for clarity
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust Layout and Display the Enhanced Boxplots
plt.tight_layout()
plt.show()

# ------------------------------------------
# 3.2.3. Outlier Detection Using Z-Score Method
# ------------------------------------------
variables = ['MonthlyIncome', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Calculate Z-Scores for Each Variable
hr_zscores = pd.DataFrame()
for column in variables:
    hr_zscores[column + '_zscore'] = stats.zscore(hr[column])

# Describing the z-scores to verify range and distribution
print("Z-score Summary for Outlier Detection:")
print(hr_zscores.describe())

# Identify rows with z-scores beyond ±3
outliers = hr_zscores[(hr_zscores > 3) | (hr_zscores < -3)].dropna(how='all')
print(f"Number of detected outliers: {len(outliers)}")

# Remove the outliers based on z-score analysis
outlier_employee_numbers = outliers.index
hr_nout = hr.drop(index=outlier_employee_numbers)
print(f"Dataframe after removing outliers: {hr_nout.shape[0]} rows remaining")

# ==========================================
# 3.3. Bivariate Data Analysis
# ==========================================

# ------------------------------------------
# 3.3.1 Attrition vs. Categorical Variables
# ------------------------------------------

# Function to create percentage bar plots for categorical variables vs. Attrition
def perc_barplots(column, colname=None):
    if not colname:
        colname = column

    # Group by the column and Attrition, then calculate the count for each combination
    attrition = hr.groupby([column, 'Attrition']).size().reset_index(name='Count')
    
    # Calculate the total count for each value of the specified column
    total_count_by_status = attrition.groupby(column)['Count'].transform('sum')
    
    # Calculate the percentage of attrition for each group
    attrition['Percentage'] = attrition['Count'] / total_count_by_status * 100
    
    # Filter to keep only rows with Attrition == 1 and sort by percentage
    attrition = attrition[attrition['Attrition'] == 1].drop(columns=['Attrition', 'Count']).sort_values(by='Percentage')
    
    # Plot the percentage of attrition for each value of the specified column
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=attrition, x=column, y='Percentage', palette='viridis')
    plt.title(f'% of Attrition per {colname}')
    plt.xlabel(colname)
    plt.xticks(rotation=45)
    plt.ylabel('% of Attrition')
    
    # Annotate each bar with the percentage value
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), textcoords='offset points')
    
    plt.show()

# Apply the function to the original categorical columns

# Attrition vs. Marital Status
perc_barplots('MaritalStatus', 'Marital Status')
# Insight: Single employees have a higher likelihood of attrition compared to those who are married or divorced.

# Attrition vs. Business Travel
perc_barplots('BusinessTravel', 'Business Travel')
# Insight: Employees who travel frequently show a significantly higher attrition rate compared to those who rarely or do not travel.

# Attrition vs. OverTime
perc_barplots('OverTime', 'Overtime')
# Insight: Employees who work overtime have a substantially higher level of attrition compared to those who do not.

# Attrition vs. JobRole
perc_barplots('JobRole', 'Job Role')
# Insight: Sales Representatives, Laboratory Technicians, and Research Scientists show higher attrition compared to Managers and Research Directors.

# Attrition vs. Department
perc_barplots('Department', 'Department')
# Insight: Employees from different departments may have different rates of attrition.

# Attrition vs. EducationField
perc_barplots('EducationField', 'Education Field')
# Insight: Education background may impact attrition rates.

# Attrition vs. Gender
perc_barplots('Gender', 'Gender')
# Insight: Observe if there is any significant difference between male and female attrition rates.

# ------------------------------------------
# 3.3.2 Attrition vs. Numerical Variables (Using Boxplots)
# ------------------------------------------

# Function to create boxplots for numerical variables vs. Attrition
def attrition_boxplot(y_column):
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=hr, x='Attrition', y=y_column, palette='coolwarm')
    plt.title(f'{y_column} by Attrition Status')
    plt.xlabel('Attrition')
    plt.ylabel(y_column)
    
    # Annotate the median on each boxplot for better understanding
    medians = hr.groupby('Attrition')[y_column].median()
    for tick, label in enumerate(ax.get_xticklabels()):
        ax.text(tick, medians[tick] + 0.05 * medians[tick], f'{medians[tick]:.2f}', 
                horizontalalignment='center', size='medium', color='black', weight='semibold')

    plt.show()

# Boxplot analysis for selected numerical variables

# Attrition vs. Monthly Income
attrition_boxplot('MonthlyIncome')
# Insight: Employees with lower monthly incomes tend to leave the company, while higher incomes are associated with staying.

# Attrition vs. Years at Company
attrition_boxplot('YearsAtCompany')
# Insight: Employees who leave tend to have fewer years of experience at the company.

# Attrition vs. Total Working Years
attrition_boxplot('TotalWorkingYears')
# Insight: Employees with longer career experience tend to stay longer, while those with fewer total working years are more likely to leave.

# Attrition vs. YearsInCurrentRole
attrition_boxplot('YearsInCurrentRole')
# Insight: Employees who have been in their current role for a shorter time tend to leave more often.

# Attrition vs. YearsSinceLastPromotion
attrition_boxplot('YearsSinceLastPromotion')
# Insight: Employees who have not been promoted recently are more likely to leave.

# Attrition vs. YearsWithCurrManager
attrition_boxplot('YearsWithCurrManager')
# Insight: Employees with fewer years with their current manager show a higher tendency to leave.

 #==========================================
# SECTION 4: Feature Selection
# ==========================================

#-------------------------------------------
# 4.1. Splitting the Dataset
#-------------------------------------------

# Define Features (X) and Target (y)
X = hr.drop(columns=['Attrition'])
y = hr['Attrition']

# Split into Train and Test Sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

#-------------------------------------------
# 4.2. Chi-Square for Categorical Variables
#-------------------------------------------

# Define StratifiedKFold for Cross-Validation
skf = StratifiedKFold(n_splits=10, random_state=99, shuffle=True)

# Select categorical features for Chi-Square test
categorical_features = X_train.select_dtypes(include='object').columns

def apply_chisquare(X, y, var, alpha=0.05):
    dfObserved = pd.crosstab(y, X)
    chi2, p, _, _ = stats.chi2_contingency(dfObserved.values)
    if p < alpha:
        result = f"{var} is IMPORTANT for Prediction"
    else:
        result = f"{var} is NOT an important predictor (Discard {var} from model)"
    print(result)
    return p < alpha

def select_best_cat_features(X, y):
    selected_features = []
    count = 1
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        print(f'----- CHI-SQUARE SPLIT {count} -----')
        X_train_cat = X_train[categorical_features].copy()
        
        for var in X_train_cat:
            if apply_chisquare(X_train_cat[var], y_train, var):
                if var not in selected_features:
                    selected_features.append(var)
        count += 1
    
    return selected_features

# Apply Chi-Square and get selected features
selected_categorical_features = select_best_cat_features(X_train, y_train)

# Keep only selected categorical features
X_train = X_train[selected_categorical_features + list(X_train.select_dtypes(include=[np.number]).columns)]
X_test = X_test[selected_categorical_features + list(X_test.select_dtypes(include=[np.number]).columns)]

#-------------------------------------------
# 4.3. Encoding Categorical Variables
#-------------------------------------------

# One-Hot Encode the selected categorical features
X_train_encoded = pd.get_dummies(X_train, columns=selected_categorical_features, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=selected_categorical_features, drop_first=True)

# Align the test set with the training set after encoding
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Update features to encoded versions
X_train, X_test = X_train_encoded, X_test_encoded

#-------------------------------------------
# 4.4. Applying SMOTE to Training Data
#-------------------------------------------

from imblearn.over_sampling import SMOTE

# Define SMOTE object and Apply SMOTE to training data to balance the classes
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Print Resampled Data Sizes
print(f'Original Training Set Size: {X_train.shape}')
print(f'SMOTE Resampled Training Set Size: {X_train_smote.shape}')

#-------------------------------------------
# 4.5. Spearman Correlation Analysis
#-------------------------------------------

# Update numerical columns after SMOTE
numerical_columns = X_train_smote.select_dtypes(include=[np.number]).columns

# Apply Spearman Correlation to detect multicollinearity among numerical variables
def cor_heatmap_v2(cor, threshold=0.3):
    # Filter out low correlations based on the threshold
    mask = np.abs(cor) < threshold
    cor_filtered = cor.copy()
    cor_filtered[mask] = 0

    plt.figure(figsize=(20, 16))  # Increase figure size
    sns.heatmap(data=cor_filtered, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f', 
                linewidths=0.5, 
                linecolor='gray',
                mask=(cor_filtered == 0),  # Only show correlations above the threshold
                cbar_kws={'shrink': 0.8}, 
                square=True,
                annot_kws={'size': 8})  # Reduce font size for annotations
    
    plt.xticks(rotation=90, fontsize=10)  # Rotate and reduce font size of x labels
    plt.yticks(fontsize=10)                # Reduce font size of y labels
    plt.title("Spearman Correlation Heatmap (Filtered for Correlations > |0.3|)", fontsize=16, weight='bold')
    plt.show()

def apply_correlation_v2(X_train):
    correlation_data = X_train.copy()
    # Compute Spearman Correlation
    matrix = correlation_data.corr(method='spearman', numeric_only=True)
    # Plot Correlation Heatmap with threshold filtering
    cor_heatmap_v2(matrix)

# Apply correlation analysis to identify highly correlated features
apply_correlation_v2(X_train_smote)

# INSIGHT: The features 'YearsInCurrentRole', 'YearsWithCurrManager', and 'JobLevel' were dropped to reduce multicollinearity:
# - 'YearsInCurrentRole' (correlation with 'YearsAtCompany': 0.88)
# - 'YearsWithCurrManager' (correlation with 'YearsAtCompany': 0.86, correlation with 'YearsInCurrentRole': 0.78)
# - 'JobLevel' (correlation with 'TotalWorkingYears': 0.74)
# 'YearsAtCompany' was chosen over 'YearsInCurrentRole' and 'YearsWithCurrManager' because it provides a more comprehensive measure of an employee's tenure.
# 'TotalWorkingYears' was preferred over 'JobLevel' as it gives a more detailed view of an employee's experience, allowing the model to capture more nuanced relationships.

# Drop correlated features
X_train_smote = X_train_smote.drop(columns=['YearsInCurrentRole', 'YearsWithCurrManager', 'JobLevel'])

# ------------------------------------------
# 4.6. Variance
# ------------------------------------------