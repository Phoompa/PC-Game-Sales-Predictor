import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('always')



#models to run
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.preprocessing import LabelBinarizer, SplineTransformer, PolynomialFeatures

#train_test_split
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV

#for cycle
from itertools import cycle

#metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.2

class model_evaluation:

    def model_fit(model, X_train, y_train):
        """Fit classification model using sklearn library. Returns: model predictions and probabilities"""
        model.fit(X_train, y_train)

    def model_predict(model, X_test):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        return y_pred, y_pred_proba
    
        def get_performance_metrics(y_test, model_predictions, model_predictions_probability):
        

            model_accuracy = sum(y_test == model_predictions) / len(y_test)
            model_precision = precision_score(y_test,model_predictions, average='weighted')
            model_recall = recall_score(y_test, model_predictions, average='weighted')
            model_f1 = f1_score(y_test,model_predictions, average='weighted')
            model_kappa = cohen_kappa_score(y_test,model_predictions)

            # Confusion matrix
            model_confusion_matrix = confusion_matrix(y_test,model_predictions)

            # Return as dictionary
            return {'Model_Accuracy': model_accuracy, 'Model_Precision': model_precision, 'Model_Recall': model_recall, 'Model_F1_Score': model_f1, \
            'Model_Kappa': model_kappa, 'Confusion_Matrix': model_confusion_matrix}


    def output_lda(y_test, pred):
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test,pred, average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating lda:  \n")
        print(" accuracy: {}\n".format(accuracy))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        return accuracy, f1, recall,precision
    
    def output_qda(y_test,pred):
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test,pred,average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating qda:  \n")
        print(" accuracy: {}\n".format(accuracy))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        return accuracy, f1, recall,precision
    
    def output_logit(y_test,pred):
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test,pred,average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating Logistic Regression:  \n")
        print(" accuracy: {}\n".format(accuracy))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        return accuracy, f1, recall,precision
    
    def output_knn_c(y_test,pred):
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test,pred,average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating KNN Classifier:  \n")
        print(" accuracy: {}\n".format(accuracy))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        return accuracy, f1, recall,precision
    
    def output_knn_r(y_test,pred):
        r2 = r2_score(y_test, pred)
        f1 = f1_score(y_test,pred,average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating KNN Regressor:  \n")
        print(" r2_score: {}\n".format(r2))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        return r2, f1, recall,precision
    
    def output_decisionTree_c(y_test,pred):
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test,pred,average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating Decision Tree Classifier:    \n")
        print(" accuracy: {}\n".format(accuracy))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        return accuracy, f1, recall,precision
    
    def output_decisionTree_r(y_test,pred):
        r2 = r2_score(y_test, pred)
        f1 = f1_score(y_test,pred,average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating Decision Tree Regressor:    \n")
        print(" r2_score: {}\n".format(r2))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        return r2, f1, recall,precision
    
    def output_Lasso(y_test, pred):
        r2 = r2_score(y_test, pred)
        f1 = f1_score(y_test,pred,average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating Lasso:    \n")
        print(" r2_score: {}\n".format(r2))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        
        
        return r2, f1, recall,precision

    def output_gnb(y_test,pred):
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test,pred, average = 'weighted')
        recall = recall_score(y_test,pred,average = 'weighted')
        precision = precision_score(y_test,pred,average = 'weighted')
        print("Evaluating GaussianNB:  \n")
        print(" accuracy: {}\n".format(accuracy))
        print(" f1_score: {}\n".format(f1))
        print(" recall: {}\n".format(recall))
        print(" precision: {}\n\n".format(precision))
        return accuracy, f1, recall, precision