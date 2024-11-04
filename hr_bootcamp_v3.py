# ===========================================
# SECTION 1: Import needed libraries
# ===========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import RFE # wrapper method
from sklearn.linear_model import LogisticRegression # (This is one possible model to apply inside RFE)
from sklearn.linear_model import LassoCV # embedded method
from sklearn.tree import DecisionTreeClassifier # embedded method
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, mutual_info_classif
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
        scaler = MinMaxScaler().fit(X_train)
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
    ['remove', 'try', 'keep']
)
numerical_ranking

# Defining new X_train based on X_train_numerical to keep/try and remaining categorical variables
variables_to_keep = numerical_ranking[numerical_ranking['Decision'] == 'keep'].index.tolist() + X_train_categorical.columns.tolist()
variables_to_try = numerical_ranking[numerical_ranking['Decision'].isin(['keep', 'try'])].index.tolist() + X_train_categorical.columns.tolist()

X_train_keep = X_train[variables_to_keep]
X_train_try = X_train[variables_to_try]