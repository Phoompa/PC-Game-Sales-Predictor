# Abstract - PC Game Sales Predictor

Today the video game market offers thousands of products of many different genres, budgets, and prices. It is difficult for developers of games to get noticed on online platforms when they lack marketing budget due to the sheer size of these platforms, and the time it takes for users to determine if they want to purchase a game. The objective of this project is to provide a tool that can determine the success of a PC game based on data that is available before the game is released, such as categories and platforms. A dataset that is extracted from Steam and SteamSpy will be used to get information about games for training and testing. Data cleanup and dimensionality reduction techniques will be necessary in order to get good data to train the models with. Since we are only concerned with the current market, the database will be reduced to only include relevant titles that have been released in the last couple of years. Determining which model to use for the dataset can be a complex process as many different factors such as performance, complexity, explainability and the size of the dataset need to be taken into account. Machine learning models including Gaussian Naive Bayes, Lasso regression, KNN, Decision Trees, Polynomial Regression, and Smoothing splines will be compared to determine which model best suits the dataset. the best suited model will generally be decided based upon the results of metrics like accuracy, f1-score, recall and precision as well how explainable the results are.

[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10491390&assignment_repo_type=AssignmentRepo)

Project Instructions
==============================

This repo contains the instructions for a machine learning project.

# Project Organization

### Modified files
- main.py - runs the project
- pre_processing.py - data processing and feature engineering
- train_model.py - functions for model training
- model_evaluation.py - functions for model evaluation and performance metrics
### Removed files
- make_dataset.py
- predict_model.py
- build_features.py
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       ├── main.py        <- Main file that runs the project
       │
       ├── preprocessing data           <- Scripts to download or generate data and pre-process the data
       │   └── pre-processing.py
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   └── train_model.py
       │   └── model_evaluation.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py # used to visualize data before training.

