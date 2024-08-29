import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV


class ModelDataset:
    """Class to get split training data for use with models"""
    def __init__(self, data_path):
        self.path = data_path
    
    def load_data(self):
        self.dataset = pd.read_csv(self.path)
    
    def split_data(self):
        game_owners = self.dataset["owners"]
        self.dataset.drop(["owners", "name", "appid", "release_date"], axis=1, inplace=True)
        self.dataset.dropna(inplace=True)

        RANDOM_STATE = 42
        TEST_SIZE = 0.2

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, game_owners, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    def get_dataframes(self):
        self.load_data()
        self.split_data()
        
        return self.X_train, self.X_test, self.y_train, self.y_test

class ModelFit:
    """This class provides methods to fit and evaluate models"""
    def __init__(self):
        return

    def fit_model(self, model,  X_train, y_train):
        model.fit(X_train, y_train)
    
    def predict_model(self, model, X_test):
        """Returns predictions of model"""
        y_pred = model.predict(X_test)
        return y_pred
