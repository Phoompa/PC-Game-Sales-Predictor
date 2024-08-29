import os
import numpy as np
import pandas as pd
from pathlib import Path


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

class visualize:
    def __init__(self):
        sns.set(style = "dark", 
                color_codes = True,
                font_scale = 1.5)
        plt.style.use('fivethirtyeight')


        #loading data
        print(os.getcwd())
        URL = Path("./data/raw/steam/steam.csv")
        games_original = pd.read_csv(URL, encoding='UTF-8')
        games = games_original

        games_english = games
        games_english.name.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        games_english.developer.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        games_english.publisher.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        #decided to get rid of games released before 2013 as it doesn't accurately reflect the current market.
        games_english['release_date'] = pd.to_datetime(games_english['release_date'])
        games_english = games_english.loc[games_english['release_date'] >= '2013-01-01']
        print(games_english.head(100))

        temp = games_english.iloc[0:5000]
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

        temp = pd.concat([temp, devs, pubs, plats, cats, genr, tags], axis=1)
        print(temp.head(5000))

        # creates dataframe for singleplayer and multiplayer games along with their estimated owners.
        singleplayer_owners = games_english.loc[games_english['categories'].str.contains('Single-player'), 'owners']
        multiplayer_owners = games_english.loc[games_english['categories'].str.contains('Multiplayer'),'owners']
        #for the sake of graphing data, change the owner column and take the minimum amount of owners.
        singleplayer_owners = singleplayer_owners.apply(lambda x: int(x.split('-')[0]))
        multiplayer_owners = multiplayer_owners.apply(lambda x: int(x.split('-')[0]))

        fig, ax = plt.subplots()
        ax.bar(['Single-player', 'Multiplayer'], [singleplayer_owners.sum(),multiplayer_owners.sum()])
        ax.set_ylabel('Total number of units owned')
        ax.set_title('Units owned: Singleplayer vs Multiplayer')
        plt.show()

        #Compare how many owners of a game depending on price point. for this example: free to play, <= $10, <=$30, <= $60
        #games_english['price'] = pd.to_numeric(games_english['price'], errors = 'coerce')
        games_english = games_english.dropna(subset=['price'])

        temp_price = games_english
        #for the sake of the graph, take the minimum number of units owned.
        temp_price.head()
        #temp_price = temp_price.drop(['owners_min','owners_max'], axis = 1)
        temp_price[['owners_min', 'owners_max']] = temp_price['owners'].str.split('-', expand=True)
        temp_price['owners'] = temp_price['owners_min']
        temp_price = temp_price.drop(['owners_min','owners_max'], axis = 1)
        temp_price['owners'] = pd.to_numeric(temp_price['owners'], errors='coerce')


        price_ranges = [0,15,30,60,temp_price['price'].max()]
        price_labels = ['$0.00', '$0 - $15', '$15 - $30', '$30 - $60', '$60+']

        price_range_1 = 0 # price == 0
        price_range_2 = 0 # price > 0 and <= 15
        price_range_3 = 0 # price > 15 and <= 30
        price_range_4 = 0 # price > 30 and <=60
        price_range_5 = 0 # price > 60
        temp_price.head(100)

        for index,row in temp_price.iterrows():
            if row['price'] == 0:
                price_range_1 += row['owners']
            elif row['price']>0 and row['price'] <= 15:
                price_range_2 += row['owners']
            elif row['price'] > 15 and row['price'] <= 30:
                price_range_3 += row['owners']
            elif row['price'] > 30 and row['price'] <= 60:
                price_range_4 += row['owners']
            elif row['price'] > 60:
                price_range_5 += row['owners']
            
        owners_by_price_range = [price_range_1,price_range_2,price_range_3,price_range_4,price_range_5]
        ax = plt.bar(price_labels,owners_by_price_range)
        plt.xlabel('Price Range')
        plt.ylabel('Number of Owners')
        plt.title('Number of Owners by Price Range')
        plt.show()

        temp_devs = games_english

        temp_devs.head()
        #temp_price = temp_price.drop(['owners_min','owners_max'], axis = 1)
        #temp_devs[['owners_min', 'owners_max']] = temp_devs['owners'].str.split('-', expand=True)
        temp_devs['owners'] = temp_devs['owners_min']
        temp_devs = temp_devs.drop(['owners_min','owners_max'], axis = 1)
        temp_devs['owners'] = pd.to_numeric(temp_devs['owners'], errors='coerce')

        developers = temp_devs.groupby('developer').agg({'owners':'sum'})
        top_developers = developers.sort_values('owners',ascending = False)[:5]
        ax = plt.bar(top_developers.index, top_developers['owners'])
        plt.xlabel('Developer')
        plt.ylabel('Number of Owners')
        plt.title('Top Developers by Number of owners.')
        plt.show()

        top_developers.head()

        #trying the same thing again but removing free to play games from dataset.
        temp_devs = games_english

        temp_devs.head()
        #temp_price = temp_price.drop(['owners_min','owners_max'], axis = 1)
        #temp_devs[['owners_min', 'owners_max']] = temp_devs['owners'].str.split('-', expand=True)
        temp_devs['owners'] = temp_devs['owners_min']
        temp_devs = temp_devs.drop(['owners_min','owners_max'], axis = 1)
        temp_devs['owners'] = pd.to_numeric(temp_devs['owners'], errors='coerce')
        #remove free games
        temp_devs = temp_devs[temp_devs['price'] != 0.00]

        developers = temp_devs.groupby('developer').agg({'owners':'sum'})
        top_developers = developers.sort_values('owners',ascending = False)[:5]
        ax = plt.bar(top_developers.index, top_developers['owners'])
        plt.xlabel('Developer')
        plt.ylabel('Number of Owners')
        plt.title('Top Developers by Number of owners excluding free to play.')
        plt.show()

        top_developers.head()

        URL = Path("./data/raw/steam/steam.csv")
        games_original = pd.read_csv(URL, encoding='UTF-8')
        games = games_original

        games.info()
        games.head()

        games.describe()

        games["price"].unique().sum()
        sns.set_color_codes()
        fig = plt.figure()

        data = games
        plot_ = sns.countplot(data['price'].apply(int))

        fig.canvas.draw()
        new_ticks = [i.get_text() for i in plot_.get_xticklabels()]
        plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10])
        plot_.set_title("Count of games by price")

        sns.set_color_codes()
        games.plot(x="price", y="median_playtime", kind="scatter")
        games.plot(x="price", y="median_playtime", kind="scatter")
        plot_.set_title('Median Playtime vs Price')
        plt.show()

        games_english = games
        games_english.name.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        games_english.developer.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        games_english.publisher.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        games_english = games_english.dropna()
        #games_english = games_english.fillna(0).astype(int)
        games_english.head(100)

        games_english['release_date'] = pd.to_datetime(games_english['release_date'])
        games_english = games_english.loc[games_english['release_date'] >= '2013-01-01']

        temp = games_english.iloc[0:10000]
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

        temp = pd.concat([temp, devs, pubs, plats, cats, genr, tags], axis=1, join='inner')
        temp.reset_index(drop=True, inplace=True)
        temp.head()

        # These columns are one-hot encoded now so we dont need them anymore
        games_english = temp.drop(['developer', 'publisher', 'platforms', 'categories', 'genres', 'steamspy_tags'], axis=1)
        print(games_english.head())
        #compression_opts = dict(method='zip',archive_name='out.csv')  
        #games_english.to_csv('out.zip', index=False,compression=compression_opts)  
        #games_english.to_csv('pre_processed.csv', index=False)  