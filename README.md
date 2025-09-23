# XBoost Classifier Optimization with Optuna
This is an case study as an basic example of how to use Optuna to optimize XGBoost classifier hyperparameters, especifically, the sampling was applieded to `max_depth`, `n_estimators`, `eta` and `reg_lambda`.  
## The dataset  
We used the Kaggle [Loan Approval Prediction dataset](https://www.kaggle.com/competitions/playground-series-s4e10/data) that contains person key features.  

|#  |Column                    |Description                   |
|---|--------------------------|------------------------------|
|0  |id                        |                              |
|1  |person_age                |                              |
|2  |person_income             |                              |
|3  |person_home_ownership     |                              |
|4  |person_emp_length         |                              |
|5  |loan_intent               |                              |
|6  |loan_grade                |                              |
|7  |loan_amnt                 |                              |
|8  |loan_int_rate             |                              |
|9  |loan_percent_income       |                              |
|10 |cb_person_default_on_file |                              |
|11 |cb_person_cred_hist_length|                              |
|12 |loan_status               | Whether loan aprroved (Yes/No)|

The shape of training dataset is (58645, 13).

## The model
We use the dataset features to build a model that determines whether a person should get a loan or not. In this case, we trained an XGBoost classifier in two ways:
* Baseline model without adjusting hyperparameters
* Optimized model, based on an Optuna search

In both cases, we used scikit-learn’s cross_val_score to compare the average ROC_AUC metric.  
The goal is to find the best hyperparameters to improve the score.

## Results
The baseline model achieved a ROC_AUC score of 0.9479, while the optimized model reached 0.9605 after 30 trials.  
In the search, the most important hyperparameter was max_depth.  
<img width="828" height="525" alt="hyperpareters_importance" src="https://github.com/user-attachments/assets/e720dda9-0df7-47bf-937b-5a6d0712e4e0" />
