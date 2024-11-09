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
- SMOTE or SMOTE NC
- One Hot Encoding (create dummies)
- Scale the variables (use MinMaxScaler) - we can use more models
- Apply the models 

Modeling the Data: NEXT

Next step --> Modeling with both X_train1 and X_train2, and decide with which one to proceed with --> according to the final decision, drop the columns on test set as well.