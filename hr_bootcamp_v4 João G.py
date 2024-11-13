# ===========================================
# SECTION 1: Import needed libraries
# ===========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LassoCV 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, chi2, VarianceThreshold, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from itertools import chain, combinations

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

# Set global random state for reproducibility
r_state = 99

# Define StratifiedKFold for Cross-Validation
skf = StratifiedKFold(n_splits=10, random_state=r_state, shuffle=True)

#-------------------------------------------
#-------------------------------------------
# 4.2. Categorical Variables
#-------------------------------------------
#-------------------------------------------

#-------------------------------------------
# 4.2.1 Chi-Square
#-------------------------------------------

# Select categorical features for Chi-Square test
categorical_features = X_train.select_dtypes(include='object').columns
X_train_categorical = X_train[categorical_features]

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

# Apply Chi-Square
selected_categorical_features = select_best_cat_features(X_train, y_train)

# INSIGHTS: Drop the following columns --> ['Gender', 'EducationField']

#-------------------------------------------
# 4.2.2 MIC
#-------------------------------------------
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

# INSIGHTS: Drop the following columns --> ['Gender', 'Department', 'EducationField'] because treshold MIC score lower than 0.1

#-------------------------------------------
# 4.2.3 Results - Categorical Features to drop
#-------------------------------------------

X_train_categorical.drop(['Gender','Department', 'EducationField'], axis=1, inplace=True)

#-------------------------------------------
#-------------------------------------------
# 4.3. Numerical Features
#-------------------------------------------
#-------------------------------------------

#-------------------------------------------
# 4.3.1 Spearman Correlation
#-------------------------------------------

# Select numerical columns only for Spearman Correlation analysis
numerical_columns = X_train.select_dtypes(include=[np.number]).columns

def cor_heatmap(cor, threshold=0.3):
    # Filter out low correlations based on the threshold
    mask = np.abs(cor) < threshold
    cor_filtered = cor.copy()
    cor_filtered[mask] = 0

    plt.figure(figsize=(20, 16))  # Increase figure size for better readability
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

# INSIGHTS: Drop the following columns --> ['YearsInCurrentRole', 'JobLevel', 'YearsAtCompany']
X_train.drop(['YearsInCurrentRole', 'JobLevel', 'YearsWithCurrManager'], axis=1, inplace=True)

# Update numerical columns after dropping correlated features
numerical_columns = X_train.select_dtypes(include=[np.number]).columns
X_train_numerical = X_train[numerical_columns]

#-------------------------------------------
# 4.3.2 Variance Threshold for Low Variance Features
#-------------------------------------------

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

# ------------------------------------------
# 4.3.3 RFE, Lasso, DT Models with 10 Splits
# ------------------------------------------

# 4.3.3.1 RFE
def apply_rfe(X_train, y_train, n_features_to_select=5):
    # Applying RFE with Logistic Regression as the base model
    rfe = RFE(estimator=LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    # Convert True/False to 1/0
    return pd.Series(rfe.support_, index=X_train.columns).astype(int)

# 4.3.3.2 Lasso 
def apply_lasso(X_train, y_train):
    # Applying Lasso for feature selection
    lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    # Convert non-zero coefficients to 1 and zero coefficients to 0
    return pd.Series(lasso.coef_, index=X_train.columns).apply(lambda x: 1 if x != 0 else 0)

# 4.3.3.3 Decision Tree
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

numerical_ranking = select_best_features(X_train_numerical,y_train)
numerical_ranking['Total_Sum'] = numerical_ranking.sum(axis=1) 
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

# Defining new X_train based on X_train_numerical to keep/try and remaining categorical variables
variables_to_keep = numerical_ranking[numerical_ranking['Decision'] == 'keep'].index.tolist() + X_train_categorical.columns.tolist()
variables_to_try_and_keep = numerical_ranking[numerical_ranking['Decision'].isin(['keep', 'try'])].index.tolist() + X_train_categorical.columns.tolist()

X_train_keep = X_train[variables_to_keep]
X_train_try_and_keep = X_train[variables_to_try_and_keep]

#variables_to_try = numerical_ranking[numerical_ranking['Decision'].isin(['keep', 'try'])].index.tolist() + X_train_categorical.columns.tolist()

#all_combinations = []
#for r in range(len(consider_indices) + 1):
    #comb = list(itertools.combinations(consider_indices, r))
    #all_combinations.extend(comb)
 
#final_combinations = [keep_indices + list(comb) for comb in all_combinations]
 
#final_combinations


X_to_train = X_train_try_and_keep.copy()

# Apply one hot enconding before SMOTE NC

categorical_columns = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'OverTime']
X_categorical_features = X_to_train[categorical_columns]
encoder = OneHotEncoder(drop='first', sparse_output=False).fit(X_categorical_features)
X_train_encoded_cat = pd.DataFrame(encoder.transform(X_categorical_features),
                                   columns=encoder.get_feature_names_out(categorical_columns),
                                   index=X_categorical_features.index)
X_train_encoded_cat = X_train_encoded_cat.astype(int)
X_no_categorical = X_to_train.drop(columns=categorical_columns)
X_train_final_keep = pd.concat([X_no_categorical, X_train_encoded_cat], axis=1)
categorical_column_indices=[14,15,16,17,18,19,20,21,22,23,24,25,26]

#SMOTE NC to solve class imbalance 
smote_nc = SMOTENC(categorical_features=categorical_column_indices)
X_resampled, y_resampled = smote_nc.fit_resample(X_train_final_keep, y_train)

#MinMaxScaler 
numerical_features = ['Age', 'DailyRate','DistanceFromHome','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobSatisfaction','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany']
X_numerical = X_resampled[numerical_features]
scaler = MinMaxScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)
X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features)
X_binary = X_resampled.drop(columns=numerical_features)
X_resampled_scaled = pd.concat([X_numerical_scaled_df, X_binary], axis=1)


#Now for only the keep data 

categorical_columns2 = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'OverTime']
X_categorical_features2 = X_train_keep[categorical_columns2]
encoder2 = OneHotEncoder(drop='first', sparse_output=False).fit(X_categorical_features2)
X_train_encoded_cat2 = pd.DataFrame(encoder2.transform(X_categorical_features2),
                                   columns=encoder2.get_feature_names_out(categorical_columns2),
                                   index=X_categorical_features2.index)
X_train_encoded_cat2 = X_train_encoded_cat2.astype(int)
X_no_categorical2 = X_train_keep.drop(columns=categorical_columns2)
X_train_final_keep2 = pd.concat([X_no_categorical2, X_train_encoded_cat2], axis=1)
categorical_column_indices2=[7,8,9,10,11,12,13,14,15,16,17,18,19]

#SMOTE NC to solve class imbalance 
smote_nc = SMOTENC(categorical_features=categorical_column_indices2)
X_resampled2, y_resampled2 = smote_nc.fit_resample(X_train_final_keep2, y_train)

#MinMaxScaler 
numerical_features2 = ['Age','EnvironmentSatisfaction','HourlyRate','JobSatisfaction','MonthlyIncome','StockOptionLevel','TotalWorkingYears']
X_numerical2 = X_resampled2[numerical_features2]
scaler2 = MinMaxScaler()
X_numerical_scaled2 = scaler.fit_transform(X_numerical2)
X_numerical_scaled_df2 = pd.DataFrame(X_numerical_scaled2, columns=numerical_features2)
X_binary2 = X_resampled2.drop(columns=numerical_features2)
X_resampled_scaled2 = pd.concat([X_numerical_scaled_df2, X_binary2], axis=1)

# Models -> Logistic Regression + Decision Trees

def select_best_models(xdata, ydata,model):
    skf = StratifiedKFold(n_splits = 5, random_state = 99, shuffle = True)
    X = xdata
    y = ydata


    score_train, score_val = [],[]

    # perform the cross-validation
    for train_index, val_index in skf.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Apply model
        model.fit(X_train, y_train)
        predictions_train = model.predict(X_train)
        predictions_val = model.predict(X_val)
        score_train.append(f1_score(y_train, predictions_train))
        score_val.append(f1_score(y_val, predictions_val))

    avg_train = round(np.mean(score_train),3)
    avg_val = round(np.mean(score_val),3)
    std_train = round(np.std(score_train),2)
    std_val = round(np.std(score_val),2)

    return avg_train, std_train, avg_val, std_val

def show_results(df, xdata,ydata, *args):
    count = 0
    # for each instance of model passed as argument
    for arg in args:
        avg_train, std_train, avg_val, std_val = select_best_models(xdata,ydata, arg)
        # store the results in the right row
        df.iloc[count] = str(avg_train) + '+/-' + str(std_train), str(avg_val) + '+/-' + str(std_val)
        count+=1
    return df

model_DT = DecisionTreeClassifier(max_depth = 3, random_state = 99)
model_LogR = LogisticRegression(random_state=99)

df_all = pd.DataFrame(columns = ['Train','Validation'], index = ['DT','LogR'])

show_results(df_all, X_resampled_scaled, y_resampled, model_LogR, model_DT)

df_keep = pd.DataFrame(columns = ['Train','Validation'], index = ['DT','LogR'])

show_results(df_keep, X_resampled_scaled2, y_resampled2, model_LogR, model_DT)

#Best model based on f1score is Decision Tree. Based on overfitting, they're tied.

#GridSearch 

# Logistic Regression
#keep + try data
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]
logModel = LogisticRegression()       
clf = GridSearchCV(logModel, param_grid = param_grid, scoring = 'f1', return_train_score = True, cv = 5)
best_clf = clf.fit(X_resampled_scaled,y_resampled)
print("Best Hyperparameters: ", best_clf.best_params_)
print("Best Score: ", best_clf.best_score_)

#Best Hyperparameters:  {'C': 1.623776739188721, 'max_iter': 100, 'penalty': 'l2', 'solver': 'lbfgs'}
#Best Score:  0.8303727529725732

#keep data
clf2 = GridSearchCV(logModel, param_grid = param_grid, scoring = 'f1', return_train_score = True, cv = 5)
best_clf2 = clf.fit(X_resampled_scaled2,y_resampled2)
print("Best Hyperparameters: ", best_clf2.best_params_)
print("Best Score: ", best_clf2.best_score_)

#Best Hyperparameters:  {'C': 11.288378916846883, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}
#Best Score:  0.7898557133400309

#Decision Tree
#keep + try
DTModel = DecisionTreeClassifier()
tree_param={'criterion':['gini','entropy','log_loss'],'max_depth':list(range(200))}
clf3 = GridSearchCV(DTModel, param_grid = tree_param, scoring = 'f1', return_train_score = True, cv = 5)
best_clf3 = clf3.fit(X_resampled_scaled,y_resampled)
print("Best Hyperparameters: ", best_clf3.best_params_)
print("Best Score: ", best_clf3.best_score_)

#Best Hyperparameters:  {'criterion': 'gini', 'max_depth': 116}
#Best Score:  0.8328544387333017
#keep
clf4 = GridSearchCV(DTModel, param_grid = tree_param, scoring = 'f1', return_train_score = True, cv = 5)
best_clf4 = clf4.fit(X_resampled_scaled2,y_resampled2)
print("Best Hyperparameters: ", best_clf4.best_params_)
print("Best Score: ", best_clf4.best_score_)

#Best Hyperparameters:  {'criterion': 'entropy', 'max_depth': 11}
#Best Score:  0.8416895936131228

#Based on the previous results, both models were ran with keep features and keep+try features
#Creating models
finalkeeptry_dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 116)
finalkeeptry_logr = LogisticRegression(C= 1.623776739188721, max_iter= 100, penalty= 'l2', solver= 'lbfgs')
finalkeep_dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= 11)
finalkeep_logr = LogisticRegression(C= 11.288378916846883, max_iter= 100, penalty= 'l1', solver= 'liblinear')
#Running models
df_final_models1 = pd.DataFrame(columns = ['Train','Validation'], index = ['Best LogR','Best DT'])
show_results(df_final_models1, X_resampled_scaled, y_resampled, finalkeeptry_logr, finalkeeptry_dt)

df_final_models2 = pd.DataFrame(columns = ['Train','Validation'], index = ['Best LogR','Best DT'])
show_results(df_final_models2, X_resampled_scaled2, y_resampled2, finalkeep_logr, finalkeep_dt)

#LogR on the keep+try dataset had the best validation f1 score with the least amount of overfitting. 

#Now let's try to choose based on a ROC Curve
#keep+try data
X_train, X_val, y_train, y_val = train_test_split(X_resampled_scaled, y_resampled,
                                                  train_size = 0.8,
                                                  random_state = 99,
                                                  stratify = y_resampled)
modelkeeptry_dt = finalkeeptry_dt.fit(X_train, y_train)
modelkeeptry_logr = finalkeeptry_logr.fit(X_train, y_train)
prob_modelkeeptryDT =  modelkeeptry_dt.predict_proba(X_val)
prob_modelkeeptryLogR =  modelkeeptry_logr.predict_proba(X_val)
fpr_modelkeeptryDT, tpr_modelkeeptryDT, thresholds_modelkeeptryDT = roc_curve(y_val, prob_modelkeeptryDT[:,1])
fpr_modelkeeptrylogr, tpr_modelkeeptrylogr, thresholds_modelkeeptrylogr = roc_curve(y_val, prob_modelkeeptryLogR[:,1])
plt.plot(fpr_modelkeeptryDT, tpr_modelkeeptryDT,label="ROC Curve DT")
plt.plot(fpr_modelkeeptrylogr, tpr_modelkeeptrylogr, label="ROC Curve LogR")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
roc_auc_modelkeeptryDT = roc_auc_score(y_val, prob_modelkeeptryDT[:, 1])
roc_auc_modelkeeptryLogR = roc_auc_score(y_val, prob_modelkeeptryLogR[:, 1])
print(roc_auc_modelkeeptryDT)
print(roc_auc_modelkeeptryLogR)

#0.7919075144508672 - DT
#0.9272277723946674 - LogR


#keep data
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_resampled_scaled2, y_resampled2,
                                                  train_size = 0.8,
                                                  random_state = 99,
                                                  stratify = y_resampled2)
modelkeep_dt = finalkeep_dt.fit(X_train2, y_train2)
modelkeep_logr = finalkeep_logr.fit(X_train2, y_train2)
prob_modelkeepDT =  modelkeep_dt.predict_proba(X_val2)
prob_modelkeepLogR =  modelkeep_logr.predict_proba(X_val2)
fpr_modelkeepDT, tpr_modelkeepDT, thresholds_modelkeepDT = roc_curve(y_val2, prob_modelkeepDT[:,1])
fpr_modelkeeplogr, tpr_modelkeeplogr, thresholds_modelkeeplogr = roc_curve(y_val, prob_modelkeepLogR[:,1])
plt.plot(fpr_modelkeepDT, tpr_modelkeepDT,label="ROC Curve DT")
plt.plot(fpr_modelkeeplogr, tpr_modelkeeplogr, label="ROC Curve LogR")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
roc_auc_modelkeepDT = roc_auc_score(y_val2, prob_modelkeepDT[:, 1])
roc_auc_modelkeepLogR = roc_auc_score(y_val2, prob_modelkeepLogR[:, 1])
print(roc_auc_modelkeepDT)
print(roc_auc_modelkeepLogR)

#0.8456847873300144 -> DT
#0.890641184135788 -> LogR

#Best model is LogR on keep+try features 

#Adjusting threshold

final_modeljg = modelkeeptry_logr.fit(X_resampled_scaled, y_resampled)
predict_proba = final_modeljg.predict_proba(X_val)
precision, recall, thresholds = precision_recall_curve(y_val, predict_proba[:,1])

# Compute F1 score, avoid division by zero
fscore = np.where((precision + recall) > 0, (2 * precision * recall) / (precision + recall), 0)
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

plt.plot(recall, precision, marker='.', label='DT')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

#Best Threshold=0.485693, F-Score=0.883

# SVM
model_SVM = SVC(probability=True, random_state=99)

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(probability=True)
clf_svm = GridSearchCV(svm_model, param_grid=param_grid_svm, scoring='f1', return_train_score=True, cv=5)
best_svm = clf_svm.fit(X_resampled_scaled, y_resampled)

print("Best SVM Hyperparameters: ", best_svm.best_params_)
print("Best SVM Score: ", best_svm.best_score_)

# Best SVM Hyperparameters:  {'C': 1, 'gamma': 'scale', 'kernel': 'poly'}
# Best SVM Score:  0.856913314750406

final_svm = SVC(probability=True, **best_svm.best_params_)

# Random Forest (RF)
model_RF = RandomForestClassifier(random_state=99)

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

print("Best RF Hyperparameters: ", best_rf.best_params_)
print("Best RF Score: ", best_rf.best_score_)

# Best RF Hyperparameters:  {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# Best RF Score:  0.9067569124667912

final_rf = RandomForestClassifier(random_state=99, **best_rf.best_params_)

# Train and Evaluate Models on Keep+Try Dataset
df_final_models1 = pd.DataFrame(columns=['Train', 'Validation'], index=['Best SVM', 'Best RF'])
show_results(df_final_models1, X_resampled_scaled, y_resampled, final_svm, final_rf)
print(df_final_models1)

# 0.877 -> SVM
# 0.909 -> RF
# Best model on Keep+Try dataset is RF

# Train and Evaluate Models on Keep Dataset
df_final_models2 = pd.DataFrame(columns=['Train', 'Validation'], index=['Best SVM', 'Best RF'])
show_results(df_final_models2, X_resampled_scaled2, y_resampled2, final_svm, final_rf)
print(df_final_models2)

# 0.848 -> SVM
# 0.897 -> RF
# Best model on Keep dataset is RF

# Threshold Adjustment for SVM
X_train, X_val, y_train, y_val = train_test_split(X_resampled_scaled, y_resampled, train_size=0.8, random_state=99, stratify=y_resampled)
final_model_svm = final_svm.fit(X_train, y_train)
predict_proba_svm = final_model_svm.predict_proba(X_val)

precision, recall, thresholds = precision_recall_curve(y_val, predict_proba_svm[:, 1])
fscore = np.where((precision + recall) > 0, (2 * precision * recall) / (precision + recall), 0)
ix = np.argmax(fscore)
print('Best Threshold (SVM)=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

plt.plot(recall, precision, marker='.', label='SVM')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Best Threshold (SVM)=0.599842, F-Score=0.896

# Threshold Adjustment for RF
final_model_rf = final_rf.fit(X_train, y_train)
predict_proba_rf = final_model_rf.predict_proba(X_val)

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_val, predict_proba_rf[:, 1])
fscore_rf = np.where((precision_rf + recall_rf) > 0, (2 * precision_rf * recall_rf) / (precision_rf + recall_rf), 0)
ix_rf = np.argmax(fscore_rf)
print('Best Threshold (RF)=%f, F-Score=%.3f' % (thresholds_rf[ix_rf], fscore_rf[ix_rf]))

plt.plot(recall_rf, precision_rf, marker='.', label='Random Forest')
plt.scatter(recall_rf[ix_rf], precision_rf[ix_rf], marker='o', color='black', label='Best')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Best Threshold (RF)=0.653636, F-Score=0.917

# ROC-AUC Comparison
fpr_svm, tpr_svm, _ = roc_curve(y_val, predict_proba_svm[:, 1])
roc_auc_svm = roc_auc_score(y_val, predict_proba_svm[:, 1])

fpr_rf, tpr_rf, _ = roc_curve(y_val, predict_proba_rf[:, 1])
roc_auc_rf = roc_auc_score(y_val, predict_proba_rf[:, 1])

plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.3f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

print("SVM ROC-AUC:", roc_auc_svm)
print("Random Forest ROC-AUC:", roc_auc_rf)

# Random Forest Feature Importance
feature_names = X_resampled_scaled.columns
importances = final_model_rf.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("Feature Importances (Random Forest):")
print(importance_df)

# The Random Forest model is the best choice here because:
# 1. Higher AUC (RF AUC = 0.964 > SVM AUC = 0.948): Indicates better performance in distinguishing between classes across all thresholds.
# 2. Flexibility: Random Forest generally handles feature importance and noisy data better than SVM.