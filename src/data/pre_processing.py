import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

class DatasetProcessing:
    """Data processing class

    This class takes a raw database, steam.csv and transforms it into a final 
    encoded and feature selected dataset ready to be used by machine learning models.
    """
    def __init__(self, raw_path, save_path):
        self.raw_path = raw_path
        self.save_path = save_path
    
    def read_data(self):
        self.dataset = pd.read_csv(self.raw_path, encoding='UTF-8')
        return
    
    def print_info(self):
        print(self.dataset.info)
        return
    
    def make_ascii(self):
        self.dataset.name.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        self.dataset.developer.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        self.dataset.publisher.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        self.dataset = self.dataset.dropna()
        return
    
    def remove_before_date(self, date='2013-01-01'):
        self.dataset['release_date'] = pd.to_datetime(self.dataset['release_date'])
        self.dataset = self.dataset.loc[self.dataset['release_date'] >= date]
        return
    
    def one_hot_tag_encoding(self):
        # Cuts the dataset to [0:10,000] 
        # Performs one hot encoding on developer, publisher, platforms, genre, tags
        temp = self.dataset.iloc[0:10000]
        devs = temp['developer'].str.get_dummies(';')
        devs = devs.add_prefix('dev-')
        devs.reset_index(drop=True, inplace=True)

        pubs = temp['publisher'].str.get_dummies(';')
        pubs = pubs.add_prefix('pub-')
        pubs.reset_index(drop=True, inplace=True)

        plats = temp['platforms'].str.get_dummies(';')
        plats = plats.add_prefix('plat-')
        plats.reset_index(drop=True, inplace=True)

        cats = temp['categories'].str.get_dummies(';')
        cats = cats.add_prefix('cat-')
        cats.reset_index(drop=True, inplace=True)

        genr = temp['genres'].str.get_dummies(';')
        genr = genr.add_prefix('genr-')
        genr.reset_index(drop=True, inplace=True)

        tags = temp['steamspy_tags'].str.get_dummies(';')
        tags = tags.add_prefix('tag-')
        tags.reset_index(drop=True, inplace=True)

        self.dataset = pd.concat([temp, devs, pubs, plats, cats, genr, tags], axis=1, join='inner')
        self.dataset.reset_index(drop=True, inplace=True)
        return
    
    def drop_after_encoding(self):
        self.dataset = self.dataset.drop(['developer', 'publisher', 'platforms', 'categories', 'genres', 'steamspy_tags', 'achievements'], axis=1)
        return
    
    def process_owners(self):
        self.dataset[['owners_min', 'owners_max']] = self.dataset['owners'].str.split('-', expand=True)
        self.dataset['owners'] = self.dataset['owners_min']
        self.dataset = self.dataset.drop(['owners_min','owners_max'], axis = 1)
        self.dataset['owners'] = pd.to_numeric(self.dataset['owners'], errors='coerce')
        return
    
    def feature_selection(self, number_features=500):
        games_selection = self.dataset
        selection = SelectKBest(f_regression, k=number_features)
        new_array = selection.fit_transform(self.dataset.iloc[::, 11:], self.dataset["owners"])
        names = selection.get_feature_names_out()

        games_trimmed = self.dataset.iloc[::, 0:11]
        tag_df=self.dataset
        unpopular_tags = [tag for tag in tag_df.columns if tag not in names]
        tag_df.drop(columns=unpopular_tags, inplace=True)
        
        self.dataset = pd.concat([games_trimmed, tag_df], axis=1)
        return

    def save_dataset(self):
        self.dataset.to_csv(self.save_path, index=False)

    def process_dataset(self):
        """Run this function after initializing to generate a finalized dataset"""
        self.read_data()
        print("\n")
        print(f"Using dataset {self.raw_path}")
        self.print_info()
        self.make_ascii()
        self.remove_before_date()
        print("Performing one hot encoding")
        self.one_hot_tag_encoding()
        self.drop_after_encoding()
        self.process_owners()
        print("Performing feature selection")
        self.feature_selection()
        print("Saving final dataset")
        self.save_dataset()
