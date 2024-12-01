# :::::::::::::::::::::::::::::::::::::::::::::::::
# SECTION 1: Import needed libraries
# :::::::::::::::::::::::::::::::::::::::::::::::::

# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

# Statistics and Scipy
from scipy import stats

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score, 
    precision_recall_curve, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
)
from sklearn.feature_selection import mutual_info_classif, RFE, VarianceThreshold
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

# Utilities
from collections import Counter

# SHAP
import shap

# :::::::::::::::::::::::::::::::::::::::::::::::::
# SECTION 2: Data Collection and Initial Processing
# :::::::::::::::::::::::::::::::::::::::::::::::::

# =================================================
# 2.1 Dataset Overview
# =================================================

# Loading the dataset
hr = pd.read_csv('HR_Attrition_Dataset.csv')
hr.head()

# =================================================
# 2.2 Data Description
# =================================================

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

# =================================================
# 2.3 Data Preprocessing
# =================================================

# Dropping columns with limited utility (constant value or irrelevant information)
hr.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# Encoding 'Attrition' as a binary variable
hr['Attrition'] = hr['Attrition'].apply(lambda x: 1 if x == "Yes" else 0)

# Creating Age Group Variable
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

# :::::::::::::::::::::::::::::::::::::::::::::::::
# SECTION 3: Exploratory Data Analysis (EDA)
# :::::::::::::::::::::::::::::::::::::::::::::::::

# =================================================
# 3.1 Visualization
# =================================================

# -------------------------------------------------
# 3.1.1 Company's Age Analysis
# -------------------------------------------------

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

# -------------------------------------------------
# 3.1.2 Percentage of Employees by Marital Status
# -------------------------------------------------

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

# -------------------------------------------------
# 3.1.3 Number of Employees by Department
# -------------------------------------------------

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

# -------------------------------------------------
# 3.1.4 Attrition Evaluation
# -------------------------------------------------

# Attrition Distribution Plot
attrition_count = hr['Attrition'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=attrition_count.index, y=attrition_count.values, palette='pastel')

# Annotate bars
for i, value in enumerate(attrition_count.values):
    plt.text(i, value + 5, str(value), ha='center')

plt.title('Attrition Distribution in the Company')
plt.xlabel('Attrition')
plt.ylabel('Number of Employees')
plt.show()

# -------------------------------------------------
# 3.1.5 Attrition Distribution
# -------------------------------------------------

# -------------------------------------------------
# 3.1.5.1 Attrition Distribution on Numerical
# -------------------------------------------------

# Separate numerical and categorical columns
numerical_columns = hr.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = hr.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove the target column 'Attrition' from these lists if present
numerical_columns = [col for col in numerical_columns if col != 'Attrition']
categorical_columns = [col for col in categorical_columns if col != 'Attrition']

# Create histograms for numerical variables
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    data = hr.copy()
    # Create bins
    bins = 10 if data[col].nunique() > 10 else data[col].nunique()
    data['bin'] = pd.cut(data[col], bins=bins)
    attrition_dist = (
        data.groupby('bin')['Attrition']
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    attrition_dist = attrition_dist.reindex(columns=[0, 1])  # Ensure 0 and 1 order
    
    # Plot the histogram
    attrition_dist.plot(kind='bar', stacked=True, figsize=(10, 6), width=0.8, alpha=0.8)
    plt.title(f'Attrition Distribution for {col}')
    plt.xlabel(f'{col} (binned)')
    plt.ylabel('Proportion')
    plt.legend(title='Attrition', labels=['0 (No)', '1 (Yes)'])
    plt.xticks(rotation=45)
    plt.show()

# -------------------------------------------------
# 3.1.5.2 Attrition Distribution on Categorical
# -------------------------------------------------

# Create bar plots for categorical variables
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    attrition_dist = (
        hr.groupby(col)['Attrition']
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    attrition_dist = attrition_dist.reindex(columns=[0, 1])  # Ensure 0 and 1 order
    
    # Plot the bar chart
    attrition_dist.plot(kind='bar', stacked=True, figsize=(10, 6), width=0.8, alpha=0.8)
    plt.title(f'Attrition Distribution by {col}')
    plt.xlabel(col)
    plt.ylabel('Proportion')
    plt.legend(title='Attrition', labels=['0 (No)', '1 (Yes)'])
    plt.xticks(rotation=45)
    plt.show()

# =================================================
# 3.2 Univariate Data Analysis
# =================================================

# -------------------------------------------------
# 3.2.1 Visualize Data Distribution (Histograms)
# -------------------------------------------------

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

# -------------------------------------------------
# 3.2.2 Boxplot Analysis for Outlier Detection
# -------------------------------------------------

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

# -------------------------------------------------
# 3.2.3 Outlier Detection Using Z-Score Method
# -------------------------------------------------

# Detect outliers using the Z-score method
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

# =================================================
# 3.3 Bivariate Data Analysis
# =================================================

# -------------------------------------------------
# 3.3.1 Attrition vs. Categorical Variables
# -------------------------------------------------

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
# INSIGHT: Single employees have a higher likelihood of attrition compared to those who are married or divorced.

# Attrition vs. Business Travel
perc_barplots('BusinessTravel', 'Business Travel')
# INSIGHT: Employees who travel frequently show a significantly higher attrition rate compared to those who rarely or do not travel.

# Attrition vs. OverTime
perc_barplots('OverTime', 'Overtime')
# INSIGHT: Employees who work overtime have a substantially higher level of attrition compared to those who do not.

# Attrition vs. JobRole
perc_barplots('JobRole', 'Job Role')
# INSIGHT: Sales Representatives, Laboratory Technicians, and Research Scientists show higher attrition compared to Managers and Research Directors.

# Attrition vs. Department
perc_barplots('Department', 'Department')
# INSIGHT: Employees from different departments may have different rates of attrition.

# Attrition vs. EducationField
perc_barplots('EducationField', 'Education Field')
# INSIGHT: Education background may impact attrition rates.

# Attrition vs. Gender
perc_barplots('Gender', 'Gender')
# INSIGHT: Observe if there is any significant difference between male and female attrition rates.

# -------------------------------------------------
# 3.3.2 Attrition vs. Numerical Variables (Using Boxplots)
# -------------------------------------------------

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

# :::::::::::::::::::::::::::::::::::::::::::::::::
# SECTION 4: Feature Selection
# :::::::::::::::::::::::::::::::::::::::::::::::::

# =================================================
# 4.1 Splitting the Dataset
# =================================================

# Define Features (X) and Target (y)
X = hr.drop(columns=['Attrition'])
y = hr['Attrition']

# Split into Train and Test Sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Set global random state for reproducibility
r_state = 99

# Define StratifiedKFold for Cross-Validation
skf = StratifiedKFold(n_splits=10, random_state=r_state, shuffle=True)

# =================================================
# 4.2 Categorical Variables
# =================================================

# -------------------------------------------------
# 4.2.1 Chi-Square
# -------------------------------------------------

# Select categorical features for Chi-Square test
categorical_features = X_train.select_dtypes(include='object').columns
X_train_categorical = X_train[categorical_features]

# Function to apply the Chi-Square test on a given feature
def apply_chisquare(X, y, var, alpha=0.05):
    # Create a contingency table and compute Chi-Square test statistics
    dfObserved = pd.crosstab(y, X)
    chi2, p, _, _ = stats.chi2_contingency(dfObserved.values)
    # Determine if the feature is significant based on p-value
    if p < alpha:
        result = f"{var} is IMPORTANT for Prediction"
    else:
        result = f"{var} is NOT an important predictor (Discard {var} from model)"
    print(result)
    return p < alpha

# Function to select the best categorical features based on Chi-Square test
def select_best_cat_features(X, y):
    selected_features = []
    count = 1
    # Perform Chi-Square test across splits for better generalization
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        print(f'----- CHI-SQUARE SPLIT {count} -----')
        X_train_cat = X_train[categorical_features].copy()
        
        # Apply Chi-Square test for each categorical feature
        for var in X_train_cat:
            if apply_chisquare(X_train_cat[var], y_train, var):
                if var not in selected_features:
                    selected_features.append(var)
        count += 1
    
    return selected_features

# Apply Chi-Square
selected_categorical_features = select_best_cat_features(X_train, y_train)

# Insight: Drop the following columns --> ['Gender', 'EducationField']

# -------------------------------------------------
# 4.2.2 MIC
# -------------------------------------------------

# Function to calculate MIC for categorical features with 10-fold cross-validation
def calculate_mic_with_cv(X, y, skf):
    selected_features_mic = {}
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        # Split the data into training and validation sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Label encode categorical features before calculating MIC
        le = LabelEncoder()
        X_train_encoded = X_train.apply(lambda col: le.fit_transform(col) if col.dtypes == 'object' else col)
        # Compute MIC scores
        mic_scores = mutual_info_classif(X_train_encoded, y_train, discrete_features='auto',random_state=r_state)
        
        # Collect the MIC scores for each feature
        if fold == 1:
            # Initialize the dictionary to store scores for each feature
            selected_features_mic = {feature: [] for feature in X.columns}
        # Co_features_mic = {feature: [] for feature in X.columns}
        
        for feature, score in zip(X.columns, mic_scores):
            selected_features_mic[feature].append(score)
        
        print(f"MIC Scores for Fold {fold}: {dict(zip(X.columns, mic_scores))}")
    
    # Calculate the average MIC score for each feature across all folds
    average_mic_scores = {feature: np.mean(scores) for feature, scores in selected_features_mic.items()}
    
    # Create a DataFrame to display average MIC scores
    mic_df = pd.DataFrame(list(average_mic_scores.items()), columns=['Feature', 'Average MIC Score'])
    mic_df.sort_values(by='Average MIC Score', ascending=False, inplace=True)
    
    print("\nAverage MIC Scores Across 10 Folds for Categorical Variables:")
    print(mic_df)
    return mic_df

# Use the defined StratifiedKFold for 10-fold cross-validation
mic_scores_df = calculate_mic_with_cv(X_train[categorical_features], y_train, skf)

# Insight: Drop the following columns --> ['Gender', 'Department', 'EducationField'] because treshold MIC score is lower than 0.1

# -------------------------------------------------
# 4.2.3 Results - Categorical Features to drop
# -------------------------------------------------

# Drop the categorical features that were determined as less relevant for prediction
X_train_categorical.drop(['Gender','Department', 'EducationField'], axis=1, inplace=True)

# =================================================
# 4.3 Numerical Features
# =================================================

# -------------------------------------------------
# 4.3.1 Spearman Correlation
# -------------------------------------------------

# Select numerical columns only for Spearman Correlation analysis
numerical_columns = X_train.select_dtypes(include=[np.number]).columns

def cor_heatmap(cor, threshold=0.3):
    # Filter out correlations below the specified threshold
    mask = np.abs(cor) < threshold
    cor_filtered = cor.copy()
    cor_filtered[mask] = 0

    # Plot the heatmap
    plt.figure(figsize=(20, 16)) 
    sns.heatmap(data=cor_filtered,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                linecolor='gray',
                mask=(cor_filtered == 0), 
                cbar_kws={'shrink': 0.8},
                square=True,
                annot_kws={'size': 8}) 
    
    plt.xticks(rotation=90, fontsize=10) 
    plt.yticks(fontsize=10)                
    plt.title("Spearman Correlation Heatmap (Filtered for Correlations > |0.3|)", fontsize=16, weight='bold')
    plt.show()

def apply_correlation(X_train):
    correlation_data = X_train.copy()
    # Compute Spearman Correlation
    matrix = correlation_data.corr(method='spearman', numeric_only=True)
    # Plot Correlation Heatmap with threshold filtering
    cor_heatmap(matrix)

# Apply Spearman Correlation analysis on X_train
apply_correlation(X_train)

# To confirm the features to remove from the correlation analysis, we will run a Decision Tree to select them based on importance
correlated_features = ["JobLevel","MonthlyIncome","YearsAtCompany","YearsInCurrentRole","YearsWithCurrManager"]
X_correlated = X_train[correlated_features]

# List to store feature importances for each split
feature_importances_all_splits = []

# Function to collect feature importances for each split
def collect_feature_importance(X_correlated, y):
    for train_index, val_index in skf.split(X_correlated, y):
        X_train, X_val = X_correlated.iloc[train_index], X_correlated.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Fit Decision Tree on the split data
        dt = DecisionTreeClassifier(random_state=99)
        dt.fit(X_train, y_train)
        
        # Collect feature importances
        feature_importances = pd.Series(dt.feature_importances_, index=X_correlated.columns)
        feature_importances_all_splits.append(feature_importances)

# Run the function to collect importances across splits
collect_feature_importance(X_correlated, y_train)

# Create a DataFrame to store all splits' importances and calculate the mean
feature_importance_df = pd.DataFrame(feature_importances_all_splits)
mean_feature_importances = feature_importance_df.mean().sort_values(ascending=False)

# Plot the average feature importance
plt.figure(figsize=(10, 8))
mean_feature_importances.plot(kind="barh")
plt.title("Average DT Feature Importance Across 10 Splits")
plt.xlabel("Average Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Insight: Drop the following columns --> ['YearsInCurrentRole', 'JobLevel', 'YearsAtCompany']
X_train.drop(['YearsInCurrentRole', 'JobLevel', 'YearsWithCurrManager'], axis=1, inplace=True)

# Update numerical columns after dropping correlated features
numerical_columns = X_train.select_dtypes(include=[np.number]).columns
X_train_numerical = X_train[numerical_columns]

# -------------------------------------------------
# 4.3.2 Variance Threshold for Low Variance Features
# -------------------------------------------------

# Function to apply variance threshold using cross-validation on numerical variables
def select_features_variance(X, y, threshold=0.01):
    count = 1
    low_variance_features_per_split = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]

        print(f'----- VARIANCE SPLIT {count} -----')

        # Initialize VarianceThreshold selector with the specified threshold
        selector = VarianceThreshold(threshold=threshold)

        # Fit the selector on the training data
        selector.fit(X_train)

        # Get variances for all features in the current training split
        variances = pd.Series(selector.variances_, index=X_train.columns)
        pd.options.display.float_format = '{:.6f}'.format  # Disable scientific notation

        # Print variances for all features in decimal notation
        print("Feature Variances:")
        print(variances)

        # Identify features with variance below the threshold
        low_variance_features = variances[variances < threshold].index.tolist()
        if low_variance_features:
            print("Features with Low Variance (Below Threshold):")
            print(low_variance_features)
        else:
            print("No features found with variance below the threshold.")

        low_variance_features_per_split.append(low_variance_features)

        count += 1

    # Summarize the most frequently occurring low variance features across all splits
    feature_counter = Counter()
    for features in low_variance_features_per_split:
        feature_counter.update(features)

    feature_freq_df = pd.DataFrame.from_dict(feature_counter, orient='index', columns=['Frequency']).sort_values(by='Frequency', ascending=False)
    print("\nFrequency of Features Considered Low Variance Across Splits:")
    print(feature_freq_df)

# Apply variance threshold selection with cross-validation on numerical features only
select_features_variance(X_train_numerical, y_train, threshold=0.03)

# -------------------------------------------------
# 4.3.3 RFE, Lasso, DT Models with 10 Splits
# -------------------------------------------------

# 4.3.3.1 RFE
#..................................................

def apply_rfe(X_train, y_train, n_features_to_select=5):
    # Applying RFE with Logistic Regression as the base model
    rfe = RFE(estimator=LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    # Convert True/False to 1/0
    return pd.Series(rfe.support_, index=X_train.columns).astype(int)

# 4.3.3.2 Lasso 
#..................................................

def apply_lasso(X_train, y_train):
    # Applying Lasso for feature selection
    lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    # Convert non-zero coefficients to 1 and zero coefficients to 0
    return pd.Series(lasso.coef_, index=X_train.columns).apply(lambda x: 1 if x != 0 else 0)

# 4.3.3.3 Decision Tree
#..................................................

def apply_dt(X_train, y_train):
    # Applying Decision Tree for feature importance
    dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    # Get feature importances as a Series
    feature_importances = pd.Series(dt.feature_importances_, index=X_train.columns)
    # Identify the top 10 features by importance
    top_10_features = feature_importances.nlargest(10).index
    # Mark top 10 features as 1 and others as 0
    return feature_importances.apply(lambda x: 1 if feature_importances.index[feature_importances == x][0] in top_10_features else 0)

# 4.3.3.4 Result
#..................................................

def select_best_features(X, y):
    all_features = X.columns
    results = pd.DataFrame(index=all_features)

    # Use the predefined skf for splits
    for count, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Scale data
        scaler = RobustScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

        # Apply feature selection methods
        rfe_features = apply_rfe(X_train_scaled, y_train)
        lasso_features = apply_lasso(X_train_scaled, y_train)
        dt_features = apply_dt(X_train_scaled, y_train)

        # Store results in the DataFrame with unique columns for each method and split
        results[f'RFE_Split_{count}'] = rfe_features
        results[f'Lasso_Split_{count}'] = lasso_features
        results[f'DT_Split_{count}'] = dt_features

    return results

# Apply the feature selection process and aggregate results
numerical_ranking = select_best_features(X_train_numerical,y_train)
# Summing results across splits to compute total votes for each feature
numerical_ranking['Total_Sum'] = numerical_ranking.sum(axis=1) 

# Classify features based on their importance
numerical_ranking['Decision'] = np.select(
    [
        numerical_ranking['Total_Sum'] < 15,
        (numerical_ranking['Total_Sum'] >= 15) & (numerical_ranking['Total_Sum'] < 20),
        numerical_ranking['Total_Sum'] >= 20
    ],
    ['remove', 'try', 'keep'],
    default='remove'  # Ensure the default is a string matching the type of choices
)

print(numerical_ranking)

# Define lists of variables to keep or try based on the ranking
variables_to_keep = numerical_ranking[numerical_ranking['Decision'] == 'keep'].index.tolist() + X_train_categorical.columns.tolist()
variables_to_try_and_keep = numerical_ranking[numerical_ranking['Decision'].isin(['keep', 'try'])].index.tolist() + X_train_categorical.columns.tolist()

# Create new datasets with selected features
X_train_keep = X_train[variables_to_keep]
X_train_try_and_keep = X_train[variables_to_try_and_keep]


# :::::::::::::::::::::::::::::::::::::::::::::::::
# SECTION 5: Model & Assessment
# :::::::::::::::::::::::::::::::::::::::::::::::::

# =================================================
# 5.1 Class Imbalance and Data Normalization
# =================================================

# -------------------------------------------------
# 5.1.1 Keep + Try Dataset
# -------------------------------------------------

# Create a copy of the dataset containing features marked as "keep" and "try"
X_to_train = X_train_try_and_keep.copy()

# Apply one-hot encoding to categorical features before applying SMOTENC
categorical_columns = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'OverTime']
X_categorical_features = X_to_train[categorical_columns]
# Fit the OneHotEncoder to transform categorical variables into binary indicators
encoder = OneHotEncoder(drop='first', sparse_output=False).fit(X_categorical_features)
X_train_encoded_cat = pd.DataFrame(
    encoder.transform(X_categorical_features),
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X_categorical_features.index
)

# Convert encoded categorical features to integers
X_train_encoded_cat = X_train_encoded_cat.astype(int)

# Separate numerical features and merge with encoded categorical features
X_no_categorical = X_to_train.drop(columns=categorical_columns)
X_train_final_keep = pd.concat([X_no_categorical, X_train_encoded_cat], axis=1)

# Define the indices of categorical columns for SMOTENC
categorical_column_indices=[14,15,16,17,18,19,20,21,22,23,24,25,26]

# Apply SMOTENC to address class imbalance in the target variable 
smote_nc = SMOTENC(categorical_features=categorical_column_indices, random_state=99)
X_resampled, y_resampled = smote_nc.fit_resample(X_train_final_keep, y_train)

# Normalize numerical features using MinMaxScaler
numerical_features = ['Age', 'DailyRate','DistanceFromHome','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobSatisfaction','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany']
X_numerical = X_resampled[numerical_features]
scaler = MinMaxScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)
# Convert scaled numerical features to a DataFrame
X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features)

# Separate binary features from numerical features and combine them after scaling
X_binary = X_resampled.drop(columns=numerical_features)
X_resampled_scaled = pd.concat([X_numerical_scaled_df, X_binary], axis=1)

# =================================================
# 5.2 Model Selection and Assessment
# =================================================

# -------------------------------------------------
# 5.2.1 Create Function Show Results
# -------------------------------------------------

# Function to evaluate models using cross-validation
def select_best_models(xdata, ydata,model):
    skf = StratifiedKFold(n_splits = 5, random_state = 99, shuffle = True)
    X = xdata
    y = ydata


    score_train, score_val = [],[]

    # Perform stratified cross-validation
    for train_index, val_index in skf.split(X,y):
        # Split data into training and validation sets for this fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Predict on training and validation data
        predictions_train = model.predict(X_train)
        predictions_val = model.predict(X_val)

        # Calculate F1 scores for training and validation sets
        score_train.append(f1_score(y_train, predictions_train))
        score_val.append(f1_score(y_val, predictions_val))
    
    # Calculate average and standard deviation of F1 scores for training and validation sets
    avg_train = round(np.mean(score_train),3)
    avg_val = round(np.mean(score_val),3)
    std_train = round(np.std(score_train),2)
    std_val = round(np.std(score_val),2)

    return avg_train, std_train, avg_val, std_val

# Function to display results for multiple models
def show_results(df, xdata,ydata, *args):
    count = 0
    # Iterate over each model passed as an argument
    for arg in args:
        # Evaluate model using cross-validation
        avg_train, std_train, avg_val, std_val = select_best_models(xdata,ydata, arg)
        # Store results in the provided DataFrame
        df.iloc[count] = str(avg_train) + '+/-' + str(std_train), str(avg_val) + '+/-' + str(std_val)
        count+=1
    return df

# -------------------------------------------------
# 5.2.2 Grid Search
# -------------------------------------------------

# This section applies GridSearchCV to tune hyperparameters for various models.
# For each model, we define a parameter grid and perform cross-validation to find the best parameters.

'''
# 1) Logistic Regression
param_grid_logr = {
    'penalty': ['l2'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['lbfgs', 'newton-cg', 'sag'],
    'max_iter': [5000, 10000]
}

log_model = LogisticRegression(random_state=99)
clf_logr = GridSearchCV(log_model, param_grid=param_grid_logr, scoring='f1', return_train_score=True, cv=5)
best_logr = clf_logr.fit(X_resampled_scaled, y_resampled)
print("Best Logistic Regression Hyperparameters: ", best_logr.best_params_)
print("Best Logistic Regression Score: ", best_logr.best_score_)

# 2) Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': list(range(1, 200))
}

dt_model = DecisionTreeClassifier(random_state=99)
clf_dt = GridSearchCV(dt_model, param_grid=param_grid_dt, scoring='f1', return_train_score=True, cv=5)
best_dt = clf_dt.fit(X_resampled_scaled, y_resampled)
print("Best Decision Tree Hyperparameters: ", best_dt.best_params_)
print("Best Decision Tree Score: ", best_dt.best_score_)

# 3) SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(probability=True, random_state=99)
clf_svm = GridSearchCV(svm_model, param_grid=param_grid_svm, scoring='f1', return_train_score=True, cv=5)
best_svm = clf_svm.fit(X_resampled_scaled, y_resampled)
print("Best SVM Hyperparameters: ", best_svm.best_params_)
print("Best SVM Score: ", best_svm.best_score_)

# 4) Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30, 50],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=99)
clf_rf = GridSearchCV(rf_model, param_grid=param_grid_rf, scoring='f1', return_train_score=True, cv=5)
best_rf = clf_rf.fit(X_resampled_scaled, y_resampled)
print("Best Random Forest Hyperparameters: ", best_rf.best_params_)
print("Best Random Forest Score: ", best_rf.best_score_)

# 5) Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}

gb_model = GradientBoostingClassifier(random_state=99)
clf_gb = GridSearchCV(gb_model, param_grid=param_grid_gb, scoring='f1', return_train_score=True, cv=5)
best_gb = clf_gb.fit(X_resampled_scaled, y_resampled)
print("Best Gradient Boosting Hyperparameters: ", best_gb.best_params_)
print("Best Gradient Boosting Score: ", best_gb.best_score_)

# 6) Naive Bayes
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

nb_model = GaussianNB()
clf_nb = GridSearchCV(nb_model, param_grid=param_grid_nb, scoring='f1', return_train_score=True, cv=5)
best_nb = clf_nb.fit(X_resampled_scaled, y_resampled)
print("Best Naive Bayes Hyperparameters: ", best_nb.best_params_)
print("Best Naive Bayes Score: ", best_nb.best_score_)

# 7) Neural Network
param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.0005, 0.0001],
    'max_iter': [2000]
}

nn_model = MLPClassifier(random_state=99)
clf_nn = GridSearchCV(nn_model, param_grid=param_grid_nn, scoring='f1', return_train_score=True, cv=5)
best_nn = clf_nn.fit(X_resampled_scaled, y_resampled)
print("Best Neural Network Hyperparameters: ", best_nn.best_params_)
print("Best Neural Network Score: ", best_nn.best_score_)
'''

# Best Model Results:
# Logistic Regression: {'C': 4.281332398719396, 'max_iter': 5000, 'penalty': 'l2', 'solver': 'lbfgs'}, Score: 0.838
# Decision Tree: {'criterion': 'entropy', 'max_depth': 15}, Score: 0.835
# SVM: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}, Score: 0.863
# Random Forest: {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}, Score: 0.900
# Gradient Boosting: {'learning_rate': 0.2, 'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200, 'subsample': 0.8}, Score: 0.918
# Naive Bayes: {'var_smoothing': 1e-06}, Score: 0.764
# Neural Network: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_iter': 2000, 'solver': 'adam'}, Score: 0.869

# -------------------------------------------------
# 5.2.3 Final Models
# -------------------------------------------------

# Define final models with the best hyperparameters determined during Grid Search
final_models = {
    'LogR': LogisticRegression(C=4.281332398719396, max_iter=5000, penalty='l2', solver='lbfgs', random_state=99),
    'DT': DecisionTreeClassifier(criterion='log_loss', max_depth=102, random_state=99),
    'SVM': SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=99),
    'RF': RandomForestClassifier(
        criterion='gini', max_depth=6, min_samples_leaf=2, min_samples_split=5, n_estimators=100, random_state=99
    ),
    'GB': GradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, min_samples_leaf=7, min_samples_split=5, n_estimators=100, subsample=0.8, random_state=99
    ),
    'NB': GaussianNB(var_smoothing=1e-06),
    'NN': MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=(128, 64), learning_rate='adaptive', 
                        learning_rate_init=0.001, max_iter=2000, solver='adam', early_stopping=True, 
                        validation_fraction=0.15, random_state=99)
}
                      
# Evaluate all models using the Keep+Try dataset and display train/validation F1 scores
df_all_models = pd.DataFrame(columns=['Train', 'Validation'], index=final_models.keys())
show_results(df_all_models, X_resampled_scaled, y_resampled, *final_models.values())
print("Train/Validation scores for all models on Keep+Try dataset:")
print(df_all_models)

# Train/Validation scores for all models on Keep+Try dataset:
#              Train    Validation
# LogR  0.859+/-0.01  0.849+/-0.02
# DT       1.0+/-0.0  0.828+/-0.02
# SVM    0.921+/-0.0  0.883+/-0.01
# RF       1.0+/-0.0   0.91+/-0.01
# GB       1.0+/-0.0   0.93+/-0.01
# NB    0.765+/-0.01  0.758+/-0.02
# NN     0.995+/-0.0  0.887+/-0.01

# -------------------------------------------------
# 5.2.4 ROC Curve Analysis for All Models (Keep+Try Dataset)
# -------------------------------------------------

# Split the resampled and scaled dataset into training (80%) and validation (20%) sets for ROC curve analysi
X_train, X_val, y_train, y_val = train_test_split(X_resampled_scaled, y_resampled, train_size=0.8, random_state=99, stratify=y_resampled)

# Initialize a plot for the ROC curves
plt.figure(figsize=(10, 8))

# Iterate through each model in the final models dictionary
for model_name, model in final_models.items():
    # Fit the model on training data
    model.fit(X_train, y_train)

    # Generate predicted probabilities for the positive class (Attrition = 1)
    if hasattr(model, "predict_proba"):
        # Use predict_proba if the model supports it
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        # For models without predict_proba (e.g., SVM), use decision_function for scores
        y_prob = model.decision_function(X_val)

     # Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) for the ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    
    # Calculate the Area Under the Curve (AUC) score
    roc_auc = roc_auc_score(y_val, y_prob)

    # Plot the ROC curve for the current model
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

# Finalize the ROC plot
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for All Models (Keep+Try Dataset)')
plt.legend()
plt.show()

# -------------------------------------------------
# 5.2.5 Adjusting Threshold for Logistic Regression
# -------------------------------------------------

# Split the resampled and scaled dataset into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled_scaled,
    y_resampled,
    train_size=0.8,
    random_state=99,
    stratify=y_resampled
)

# Train the Logistic Regression model on the training set
final_logr_model = final_models['LogR'].fit(X_train, y_train)

# Obtain the probability predictions for the validation set
predict_proba_logr = final_logr_model.predict_proba(X_val)

# Calculate precision, recall, and thresholds using the precision_recall_curve function
precision, recall, thresholds = precision_recall_curve(y_val, predict_proba_logr[:, 1])

# Compute F1 score for each threshold and find the index of the maximum F1 score
fscore = np.where((precision + recall) > 0, (2 * precision * recall) / (precision + recall), 0)
ix = np.argmax(fscore)

# Output the best threshold and corresponding F1 Score
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

# Plot the Precision-Recall curve
plt.plot(recall, precision, marker='.', label='Gradient Boosting')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve for Logistic Regression (Keep+Try Dataset)')
plt.show()

# Display the best threshold and F1 Score for Logistic Regression
print(f"Best Threshold: {thresholds[ix]:.6f}")
print(f"F-Score at Best Threshold: {fscore[ix]:.3f}")

# Best Threshold: 0.315325
# F-Score at Best Threshold: 0.850

# :::::::::::::::::::::::::::::::::::::::::::::::::
# SECTION 6: Deploy
# :::::::::::::::::::::::::::::::::::::::::::::::::

# =================================================
# 6.1 Creating a new train and test set with only the keep+try features
# =================================================

# Define Features (X) and Target (y)
X = hr[variables_to_try_and_keep]
y = hr['Attrition']

# Split into Train and Test Sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Applying the same transformations as previously on train and test set: SMOTE NC and MinMaxScaler

# -------------------------------------------------
# 6.1.1 Transformations on the Train Set
# -------------------------------------------------

# One-Hot Encode categorical features
X_categorical_featuresf = X_train[categorical_columns]
encoderf = OneHotEncoder(drop='first', sparse_output=False).fit(X_categorical_featuresf)
X_train_encoded_catf = pd.DataFrame(
    encoderf.transform(X_categorical_featuresf),
    columns=encoderf.get_feature_names_out(categorical_columns),
    index=X_categorical_featuresf.index
)
X_train_encoded_catf = X_train_encoded_catf.astype(int)  # Convert encoded features to integers
X_no_categoricalf = X_train.drop(columns=categorical_columns)  # Drop original categorical columns
X_train_final_keepf = pd.concat([X_no_categoricalf, X_train_encoded_catf], axis=1)  # Combine numeric and encoded data

# SMOTENC to solve class imbalance in the train set
categorical_column_indices = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # Indices of categorical features
smote_nc = SMOTENC(categorical_features=categorical_column_indices, random_state=99)
X_resampledf, y_resampledf = smote_nc.fit_resample(X_train_final_keepf, y_train)

# Scale numerical features using MinMaxScaler
X_numericalf = X_resampledf[numerical_features]  # Extract numerical columns
scaler = MinMaxScaler()
X_numerical_scaledf = scaler.fit_transform(X_numericalf)  # Apply MinMaxScaler
X_numerical_scaled_dff = pd.DataFrame(X_numerical_scaledf, columns=numerical_features)  # Convert to DataFrame
X_binaryf = X_resampledf.drop(columns=numerical_features)  # Extract binary/categorical data
X_resampled_scaledf = pd.concat([X_numerical_scaled_dff, X_binaryf], axis=1)  # Combine scaled numerical and binary data

# -------------------------------------------------
# 6.1.2 Transformations on the Test Set
# -------------------------------------------------

# One-Hot Encode categorical features
X_categorical_featurest = X_test[categorical_columns]
encodert = OneHotEncoder(drop='first', sparse_output=False).fit(X_categorical_featurest)
X_test_encoded_cat = pd.DataFrame(
    encodert.transform(X_categorical_featurest),
    columns=encodert.get_feature_names_out(categorical_columns),
    index=X_categorical_featurest.index
)
X_test_encoded_cat = X_test_encoded_cat.astype(int)  # Convert encoded features to integers
X_no_categoricalt = X_test.drop(columns=categorical_columns)  # Drop original categorical columns
X_test_final_keep = pd.concat([X_no_categoricalt, X_test_encoded_cat], axis=1)  # Combine numeric and encoded data

# SMOTENC to solve class imbalance in the test set
categorical_column_indices = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # Indices of categorical features
smote_nc = SMOTENC(categorical_features=categorical_column_indices, random_state=99)
X_resampledt, y_resampledt = smote_nc.fit_resample(X_test_final_keep, y_test)

# Scale numerical features using MinMaxScaler
X_numericalt = X_resampledt[numerical_features]  # Extract numerical columns
scaler = MinMaxScaler()
X_numerical_scaledt = scaler.fit_transform(X_numericalt)  # Apply MinMaxScaler
X_numerical_scaled_dft = pd.DataFrame(X_numerical_scaledt, columns=numerical_features)  # Convert to DataFrame
X_binaryt = X_resampledt.drop(columns=numerical_features)  # Extract binary/categorical data
X_resampled_scaledt = pd.concat([X_numerical_scaled_dft, X_binaryt], axis=1)  # Combine scaled numerical and binary data

# =================================================
# 6.2 Creating the final model and a column with the final prediction
# =================================================

# Train the final logistic regression model on the processed training set
final_model = final_logr_model.fit(X_resampled_scaledf, y_resampledf)

# Generate probability predictions for the test set
predict_proba_test = final_model.predict_proba(X_resampled_scaledt)

# Initialize an empty list to store the final predictions
final_pred = []  # This will hold binary predictions (0 or 1) based on a threshold

# =================================================
# 6.3 Measuring different metrics based on the threshold given on 5.2.5
# =================================================

# Apply the threshold determined in Section 5.2.5 to convert probabilities into binary predictions
for value in predict_proba_test[:, 1]:
    final_pred.append(1 if value >= 0.315325 else 0)

# Calculate performance metrics
metrics = {
    "F1 Score": f1_score(y_true=y_resampledt, y_pred=final_pred),
    "Accuracy Score": accuracy_score(y_true=y_resampledt, y_pred=final_pred),
    "Precision Score": precision_score(y_true=y_resampledt, y_pred=final_pred),
    "Recall Score": recall_score(y_true=y_resampledt, y_pred=final_pred),
}

# Display metrics in a DataFrame
metrics_table = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
print(metrics_table)

# Displaying example results for Logistic Regression and other models:
# LogR Results:
#               Metric         Value
# 0            F1 Score     0.816754
# 1      Accuracy Score     0.810811
# 2     Precision Score     0.791878
# 3        Recall Score     0.843243

# Example results for comparison:
# SVM Results:
#               Metric         Value
# 0            F1 Score     0.786026
# 1      Accuracy Score     0.801351
# 2     Precision Score     0.851735
# 3        Recall Score     0.729730

# GB Results:
#               Metric         Value
# 0            F1 Score     0.801027
# 1      Accuracy Score     0.790541
# 2     Precision Score     0.762836
# 3        Recall Score     0.843243

# RF Results:
#               Metric         Value
# 0            F1 Score     0.778846
# 1      Accuracy Score     0.751351
# 2     Precision Score     0.701299
# 3        Recall Score     0.875676

# NN Results:
#               Metric         Value
# 0            F1 Score     0.812332
# 1      Accuracy Score     0.810811
# 2     Precision Score     0.805851
# 3        Recall Score     0.818919

# =================================================
# 6.4 Confusion Matrix
# =================================================

# Generate the confusion matrix
cm = confusion_matrix(y_true=y_resampledt, y_pred=final_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Attrition", "Attrition"])
disp.plot(cmap="Blues")
disp.ax_.set_title("Confusion Matrix for Final Model")
disp.ax_.set_xlabel("Predicted Labels")
disp.ax_.set_ylabel("True Labels")
plt.show()

# =================================================
# 6.5 Applying SHAP to LogR Model
# =================================================

# -------------------------------------------------
# 6.5.1 Removing MinMaxScaler for better readability from X_Test
# -------------------------------------------------

# Inverse MinMaxScaler
X_numerical_original = scaler.inverse_transform(X_resampled_scaledt[numerical_features])
X_numerical_original_t = pd.DataFrame(X_numerical_original, columns=numerical_features)

# Extract encoded categorical features
X_encoded_categorical_t = X_resampled_scaledt.drop(columns=numerical_features)

# Combine both into a dataframe. Now, we have a df resampled, but not scaled
X_resampled_t = pd.concat([X_numerical_original_t, X_encoded_categorical_t], axis=1)

# -------------------------------------------------
# 6.5.2 Applying SHAP
# -------------------------------------------------

# Apply SHAP to explain model predictions using the test scaled data
# Note: data needs to be scaled for SHAP
explainer = shap.Explainer(final_model, X_resampled_scaledt, model_output="probability")
shap_values = explainer(X_resampled_scaledt)

# Visualize SHAP values
shap.plots.violin(shap_values, max_display=10)

# -------------------------------------------------
# 6.5.3 Insights
# -------------------------------------------------

# 6.5.3.1 Why is it so important to address overtime?
# .................................................

# Note: Dependence plots will use the values from not scaled data, to check the real values

# See how job satisfaction is amplified by whether employees work overtime
shap.dependence_plot("JobSatisfaction", shap_values.values, X_resampled_t)

# See how people who work overtime will have less time to train
shap.dependence_plot("TrainingTimesLastYear", shap_values.values, X_resampled_t)

# 6.5.3.2 Additional Insights
# .................................................

# Plotting the effect of distance from home on attrition
shap.dependence_plot("DistanceFromHome", shap_values.values, X_resampled_t)

# Most Important Features :
    # - People who are a Research Scientist tend to have a much lesser chance to leave the company
    # - People who work overtime have a much higher probability in leaving the company
    # - Higher stock option levels (red dots) are associated with a decreased likelihood of attrition.
    # - Lower job satisfaction/environemnt satisfaction/job involvement (blue dots) increases 
    # attrition, while higher values (red dots) reduces it.
    # - MANY MORE CONCLUSIONS TO BE TAKEN

# Suggestions:
# Environment Satisfaction relacionado com Distance from Home; JobInvolvement com JobSatisfaction; JobSatisfaction com Overtime
# Estratégia para reduzir Distance From Home -> Teletrabalho, o que melhora o EnvironmentSatisfaction ; Grupo de boleias ou empresa oferecer transporte
# Estratégia para reduzir OverTime que aumenta JobSatisfaction que aumenta JobInvolvement -> Introduzir metodologia Agile de modo a que cada tarefa esteja partida em tarefas mais pequenas e melhora a organização dentro de equipas
# Estratégia para aumentar StockOptionLevel e JobInvolvement -> Prémio por performance elevada ser ações da empresa.

# -------------------------------------------------
# 6.5.4 Explaining Individual Employee Predictions
# -------------------------------------------------

# 6.5.4.1 Identifying the Most Extreme Examples
# .................................................

# Predict probabilities for attrition (an array with two columns, one for attrition = 0, one for attrition = 1)
attrition_probabilities_all = final_model.predict_proba(X_resampled_scaledt)

# Extract probabilities for Attrition = 1 (using the second column, index 1)
attrition_probabilities = attrition_probabilities_all[:, 1]

# Identify indices for the most extreme cases
most_likely_attrition_idx = attrition_probabilities.argmax()  # Highest probability index
least_likely_attrition_idx = attrition_probabilities.argmin()  # Lowest probability index

# Get probabilities for the extreme cases (convert to percentages)
most_likely_attrition = attrition_probabilities[most_likely_attrition_idx]*100  # Highest probability
least_likely_attrition = attrition_probabilities[least_likely_attrition_idx]*100  # Lowest probability

# Retrieve the corresponding real data points (unscaled) for interpretation
extreme_high = X_resampled_t.iloc[most_likely_attrition_idx]
extreme_low = X_resampled_t.iloc[least_likely_attrition_idx]

# 6.5.4.2 Computing SHAP Values for Extreme Examples
# .................................................

# Compute SHAP values for the most extreme cases (using scaled data for SHAP)
shap_values_high = explainer(X_resampled_scaledt.iloc[[most_likely_attrition_idx]])
shap_values_low = explainer(X_resampled_scaledt.iloc[[least_likely_attrition_idx ]])

# 6.5.4.3 SHAP Visualization for Most Likely Attrition Example
# .................................................

# Force plot for the most likely attrition (Attrition = 1)
# Note: f(x) is the log-odd, and E[g(x)] is the base log-odd. Since we are trying to increase recall, the base log-odd may be slightly negative.
shap.force_plot(
    explainer.expected_value,
    shap_values_high.values[0],  # Scaled Data to quantify the contribution of each feature correctly
    extreme_high,  # Not Scaled Data in a human-readable format for the visualization,
    link = "logit"
)

# Waterfall plot for the most likely attrition (Attrition = 1)
# Create SHAP Explanation object with original values for better interpretability
shap_values_high_original = shap.Explanation(
    values=shap_values_high.values[0],
    base_values=shap_values_high.base_values[0],
    data=extreme_high.values,
    feature_names=extreme_high.index
)

# Waterfall plot for the most likely attrition example
shap.plots.waterfall(shap_values_high_original)


# 6.5.4.4 SHAP Visualization for Least Likely Attrition Example
# .................................................

# Force plot for the least likely attrition (Attrition = 0)
shap.force_plot(
    explainer.expected_value,
    shap_values_low.values[0],
    extreme_low,
    link="logit"
)

print("Least Likely Probability:", least_likely_attrition)

# Waterfall plot for the least likely attrition (Attrition = 0)
# Create SHAP Explanation object with original values for better interpretability
shap_values_low_original = shap.Explanation(
    values=shap_values_low.values[0],
    base_values=shap_values_low.base_values[0],
    data=extreme_low.values,
    feature_names=extreme_low.index
)

# Waterfall plot for the least likely attrition example
shap.plots.waterfall(shap_values_low_original)
