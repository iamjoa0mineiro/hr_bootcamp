# hr_bootcamp
HR Bootcamp 2024

# Instalações a serem realizadas no terminal:
pip install matplotlib seaborn squarify scikit-learn scipy
pip install polars
pip install scipy polars

# Versões:
matplotlib 3.9.2
pandas 2.2.3
seaborn 0.13.2
numpy 2.1.2

# Notas para o modelo:
Outliers - correr código com e sem outliers

12/10/2024 - João Mineiro

Boas malta, tentei deixar o código documentado ao máximo com comentários. Se puderem façam o mesmo!
Apenas algumas notas importantes:
1. Data Collection and Initial Processing:
- Dataset Overview - DONE
- Data Overview - DONE
- Data Description - DONE
- Preprocessing - DONE

2. Exploratory Data Analysis (EDA):
- Unvariate Analysis - DONE - Removeu-se 83 outliers depois de fazer análise com z-score
- Bivariate and Multivariate Analysis - DONE -  Fiz apenas heatmap. Raysa: Adicionei alguns gráficos e deixei o que estava embaixo, mas ainda vou terminar. João Grade: Corri DecisionTrees para ver quais remover face a correlation analysis e corri embedded e wrapped methods para ver quais as variáveis a remover. 
- Visualization - DONE
- Feature Engeneering - !!!NOT DONE!!!

26/10/2024
1.	Split test & train sets
2.	Feature importance no training set – aplicar remoção de variáveis em ambos os sets – train e test 
3.	Aplicar SMOTE NC no train set – Resultado: Train set balanced ready for data modeling

26/10/2024 - Joao M. & Joao G.
Feature selection based on Chi-Square, Spearman Corr. + DT, RFE, Lasso, DT --> Final decision was 2 train dfs:
1) X_train1 --> Dropping just the variables to remove
2) X_train2 --> Dropping variables to remove and to try

27/10/2024 - Bruno (Nova versão - não estava a conseguir perceber o código)
Feature selection organizada em numerical e categorical features
Spearman Correlation Analysis: DT summary table para perceber melhor que variáveis correlacionadas devem ser removidas
Final Analysis - adicionei summary table e mudei uma beca o codigo. Usei muito ChatGPT, por isso se conseguirem revejam o código só para o caso.
Recomendação: 
    remover variance analysis (não nos traz nada, não precisamos de manter no código)
    usar MIC

3/11/2024 - Joao M.
Adicionei o código do MIC - falta pensarmos num critério de variables to keep/exclude

04/11/2024 - Bruno
MIC DONE
Feature Selection for Categorical Variables is finished
New variable for the random state r_state = 99.

08/11_2024 - Raysa
Modelling

Next Steps:
- Use X_to_train
- SMOTE or SMOTE NC - FEITO
- One Hot Encoding (create dummies) - FEITO
- Scale the variables (use MinMaxScaler) - we can use more models - FEITO
- Apply the models 

11/11/2024 - João Grade

One Hot Enconding : Teve que se fazer antes do Smote NC porque ele não compreende variáveis categóricas com strings, têm que se integers. O próprio SMOTE NC faz, mas optei por fazer antes para poder ver o que ele estava a fazer
SMOTE NC feito: para não haver problemas de versões, corram isto: pip install --upgrade imbalanced-learn scikit-learn numpy
MinMax Scaler: Tem que se fazer só às numerical variables, não se pode fazer a variáveis binárias. O MinMax Scaler pega em variáveis numéricas e transforma-as em variáveis entre 0 e 1, o que faz sentido em variáveis numéricas. Ora, se fizermos isso em variáveis binárias, elas perdem o seu significado, porque o valor ou é (1) ou não é (0), não faz sentido ter o valor de 0,5. Do que pesquisei tanto em artigos como no stackoverflow a opinião global é que standardizar dados só se faz em variáveis numéricas, não se faz em variáveis binárias por perderem o significado. 

- Modelos de classificação que podemos usar tendo em conta que temos variáveis numéricas e binárias:
    - Logistic Regression
    - Decision Trees
    - Random Forests
    - Gradient Boosting
    - Support Vector Machine (SVM)
    - Neural Networks 
    - Naive Bayes


Modeling the Data: NEXT

Next step --> Modeling with both X_train1 and X_train2, and decide with which one to proceed with --> according to the final decision, drop the columns on test set as well.