# XBoost Classifier Optimization with Optuna
This repo accompanies the Medium article [A Quick Guide to Hyperparameter Optimization with Optuna](https://medium.com/@cris.lincoleo/a-quick-guide-to-hyperparameter-optimization-with-optuna-1980f1d185dc).  
It includes the example notebook used in the post so you can explore the code, run the experiments, and tweak the hyperparameter search yourself.  
Optuna was used to optimize **XGBoost** classifier hyperparameters, especifically, the sampling was applieded to `max_depth`, `n_estimators`, `eta` and `reg_lambda`.  

You can view the rendered notebook here: [![nbviewer](https://img.shields.io/badge/view%20in-nbviewer-orange)](https://nbviewer.org/github/crisbebop/optuna-hyperparam-guide/blob/main/xgboost-optuna-example.ipynb) or in Kaggle (The model was developed in a Kaggle environment) [XGBoost_Optuna_example](https://www.kaggle.com/code/crisbebop/xgboost-optuna-example?scriptVersionId=202840668)

## The dataset  
We used the Kaggle [Loan Approval Prediction dataset](https://www.kaggle.com/competitions/playground-series-s4e10/data) that contains person key features. 
### Dataset features
| #  | Column                         | Description                                                                  |
| -- | ------------------------------ | ---------------------------------------------------------------------------- |
| 0  | id                             | Unique identifier for each record                                            |
| 1  | person\_age                    | Age of the applicant (years)                                                 |
| 2  | person\_income                 | Annual income of the applicant (in USD)                                      |
| 3  | person\_home\_ownership        | Type of home ownership (e.g., rent, own, mortgage, other)                    |
| 4  | person\_emp\_length            | Length of current employment (years)                                         |
| 5  | loan\_intent                   | Purpose of the loan (e.g., education, medical, personal, etc.)               |
| 6  | loan\_grade                    | Credit grade assigned to the loan application                                |
| 7  | loan\_amnt                     | Requested loan amount (in USD)                                               |
| 8  | loan\_int\_rate                | Interest rate assigned to the loan (%)                                       |
| 9  | loan\_percent\_income          | Ratio of loan amount to the applicant’s annual income (%)                    |
| 10 | cb\_person\_default\_on\_file  | Whether the applicant has a history of default in their credit bureau record |
| 11 | cb\_person\_cred\_hist\_length | Length of the applicant’s credit history (years)                             |
| 12 | loan\_status                   | Target variable: whether the loan was approved (Yes/No)                      |

The shape of training dataset is (58645, 13).

## The model
We use the dataset features to build a model that determines whether a person should get a loan or not. In this case, we trained an XGBoost classifier in two ways:
* Baseline model without adjusting hyperparameters
* Optimized model, based on an Optuna search

In both cases, we used scikit-learn’s `cross_val_score` to compare the average `ROC_AUC` metric.  
The goal is to find the best hyperparameters to improve the score.  

## Results
The baseline model achieved a `ROC_AUC` score of **0.9479**, while the optimized model reached **0.9605** after 30 trials.  
In the search, the most important hyperparameter was `max_depth`.  


