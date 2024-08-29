import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("D:\\Documents\\CP322\\final-project-200740070\\src\\data")
sys.path.append("D:\\Documents\\CP322\\final-project-200740070\\src\\models")
sys.path.append("D:\\Documents\\CP322\\final-project-200740070\\src\\visualization")

from data.pre_processing import DatasetProcessing
from models.train_model import ModelFit, ModelDataset
from models.model_evalutation import model_evaluation
from visualization.visualize import visualize

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, r2_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('always')

RANDOM_STATE = 42

project_root = Path(__file__).parent.parent
raw_data_path = project_root / 'data' / 'raw' / 'steam' / 'steam.csv'
save_data_path = project_root / 'data' / 'processed' / 'processed_dataset.csv'
print(save_data_path)

processed_dataset_exists = save_data_path.exists()
if not processed_dataset_exists:
    print("Creating processed dataset from raw")
    games = DatasetProcessing(raw_path=raw_data_path, save_path=save_data_path)
    games.process_dataset()
    del games
else:
    print("processed file exists, continuing to training models")


#Visualize data
visualize()
# Train models
get_data_models = ModelDataset(data_path=save_data_path)
X_train, X_test, y_train, y_test = get_data_models.get_dataframes()

models = ModelFit()

# Linear Discriminant Analysis
fit_lda = LinearDiscriminantAnalysis()
models.fit_model(fit_lda, X_train, y_train)
lda_pred = models.predict_model(fit_lda, X_test)
lda_acc = accuracy_score(y_test, lda_pred)


# Logistic Regression
fit_logit = LogisticRegression(random_state=RANDOM_STATE)
models.fit_model(fit_logit, X_train, y_train)
logit_pred = models.predict_model(fit_logit, X_test)
logit_acc = accuracy_score(y_test, logit_pred)


# Gaussian Naive Bayes
fit_gnb = LinearDiscriminantAnalysis()
models.fit_model(fit_gnb, X_train, y_train)
gnb_pred = models.predict_model(fit_gnb, X_test)
gnb_acc = accuracy_score(y_test, gnb_pred)

# KNN Regressor
fit_knn = KNeighborsRegressor()
models.fit_model(fit_knn, X_train, y_train)
knn_pred = models.predict_model(fit_knn, X_test)
knn_acc = r2_score(y_test, knn_pred)

# KNN Classifier 
fit_knn_c = KNeighborsClassifier()
models.fit_model(fit_knn_c, X_train, y_train)
knn_c_pred = models.predict_model(fit_knn_c, X_test)
knn_c_acc = accuracy_score(y_test, knn_c_pred)

# Decision Tree Regressor
fit_dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
models.fit_model(fit_dt, X_train, y_train)
dt_pred = models.predict_model(fit_dt, X_test)
dt_acc = r2_score(y_test, dt_pred)


# Decision Tree Classifier 
fit_dt_c = DecisionTreeClassifier(random_state=RANDOM_STATE)
models.fit_model(fit_dt_c, X_train, y_train)
dt_c_pred = models.predict_model(fit_dt_c, X_test)
dt_c_acc = accuracy_score(y_test, dt_c_pred)

# Lasso 
fit_lasso = Lasso(alpha=0.1, random_state=42)
models.fit_model(fit_lasso, X_train, y_train)
lasso_pred = models.predict_model(fit_lasso, X_test)
lasso_acc = r2_score(y_test, lasso_pred)

print(f"LDA prediction r2 score: {lda_acc}")
print(f"Logistic regression prediction r2 score: {logit_acc}")
print(f"Gaussian NB prediction accuracy: {gnb_acc}")
print(f"KNN Regressor prediction r2 score: {knn_acc}")
print(f"KNN Classifier prediction accuracy: {knn_c_acc}")
print(f"Decision Tree Regressor prediction r2 score: {dt_acc}")
print(f"Decision Tree Classifier prediction accuracy: {dt_c_acc}")
print(f"Lasso Regression prediction r2 score: {lasso_acc}")

lda_r2, lda_f1, lda_recall, lda_precision = model_evaluation.output_lda(y_test, lda_pred)
logit_r2, logit_f1, logit_recall, logit_precision = model_evaluation.output_logit(y_test, logit_pred)
knn_r_r2, knn_r_f1, knn_r_recall, knn_r_precision = model_evaluation.output_knn_r(y_test, knn_pred)
knn_c_r2, knn_c__f1, knn_c__recall, knn_c__precision = model_evaluation.output_knn_c(y_test, knn_c_pred)
dt_r_r2, dt_r_f1, dt_r_recall, dt_r_precision = model_evaluation.output_decisionTree_r(y_test, dt_pred)
dt_c_r2, dt_c_f1, dt_c_recall, dt_c_precision = model_evaluation.output_decisionTree_c(y_test, dt_c_pred)
gnb_r2, gnb_f1, gnb_recall, gnb_precision = model_evaluation.output_gnb(y_test, gnb_pred)



#model_evaluation.output_Lasso(y_test, lasso_pred)




# graph for r2
fig, ax = plt.subplots()
ax.bar(['lda', 'logit', 'knn_r', 'knn_c', 'decisionTree_r','decisionTree_c', 'gnb'],[lda_r2,logit_r2,knn_r_r2,knn_c_r2,dt_r_r2,dt_c_r2,gnb_r2])
ax.set_ylabel('r2 Score')
ax.set_title('r2 Score of Models')
plt.show()

# graph for f1
fig, ax = plt.subplots()
ax.bar(['lda', 'logit', 'knn_r',  'decisionTree_r','decisionTree_c', 'gnb'],[lda_f1,logit_f1,knn_r_f1, dt_r_f1,dt_c_f1,gnb_f1])
ax.set_ylabel('f1 Score')
ax.set_title('f1 Score of Models')
plt.show()

# graph for recall
fig, ax = plt.subplots()
ax.bar(['lda', 'logit', 'knn_r',  'decisionTree_r','decisionTree_c', 'gnb'],[lda_recall,logit_recall,knn_r_recall,dt_r_recall,dt_c_recall,gnb_recall])
ax.set_ylabel('recall Score')
ax.set_title('recall Score of Models')
plt.show()


# graph for precision
fig, ax = plt.subplots()
ax.bar(['lda', 'logit', 'knn_r',  'decisionTree_r','decisionTree_c', 'gnb'],[lda_precision,logit_precision,knn_r_precision,dt_r_precision,dt_c_precision,gnb_precision])
ax.set_ylabel('precision Score')
ax.set_title('precision Score of Models')
plt.show()