import numpy as np
import pandas as pd
from typing import Tuple
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

SEED = 578

def read_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	# read the training and testing datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

# Usage:
train_path = 'train.csv'
test_path = 'test.csv'
train_df, test_df = read_datasets(train_path, test_path)

train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

train_df['is_original'] = 0
test_df['is_original'] = 0

# Merge the current train data with the original dataset...
original_path = 'train_dataset.csv'
original_df = pd.read_csv(original_path)
original_df['is_original'] = 1
train_df = pd.concat([train_df, original_df])

def remove_duplicates(df):

    duplicates = df[df.duplicated()]
    print(f"Number of duplicates found and removed: {len(duplicates)}")
    df_no_duplicates = df.drop_duplicates()
    return df_no_duplicates

train_df = remove_duplicates(train_df)
# outputs: Number of duplicates found and removed: 5517

ignore_list = ['id', 'smoking', 'is_original']
features = [feat for feat in train_df.columns if feat not in ignore_list]

categorical_features = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']
numerical_features = [feat for feat in train_df.columns if feat not in categorical_features and feat not in ['smoking']]

model_features = [col for col in train_df.columns if col not in ['id', 'smoking']]


'''
# Used to optimize the hyperparameters
def objective(trial):
    # Load the dataset and split it into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df[model_features], train_df['smoking'], test_size=0.25, random_state=SEED)

    # Define the hyperparameters to be optimized
    param = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "subsample": trial.suggest_float("subsample", 0.01, 1.0, step = 0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, step = 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "n_estimators": trial.suggest_int("n_estimators", 256, 4096),
        "eta": trial.suggest_float("eta", 0.01, 0.5, step = 0.01),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "tree_method": "gpu_hist",
    }

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    #Train the XGBoost model with the current hyperparameters
    model = xgb.train(param, xgb.DMatrix(X_train, label = y_train))

    # Evaluate the model on the test set
    y_pred = model.predict(xgb.DMatrix(X_test))
    loss = log_loss(y_test, y_pred)

    return loss

def optimize_xgboost_hyperparameters(num_trials=10):
    study = optuna.create_study(direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(objective, n_trials=num_trials)

    best_params = study.best_params
    return best_params

# Run the optimization

optimal_params = optimize_xgboost_hyperparameters()
print('.' * 25, '\n')
print(optimal_params)
'''

def fit_xgboost_with_kfold(df, features, target_variable, parameters, n_splits=10,  random_state=SEED):

    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    model = xgb.XGBClassifier(**parameters)

    fold_rocs = []
    fold_loglosses = []
    fold_predictions = []
    fold = 1

    for train_index, test_index in kfold.split(X[features], y):
        print(f'Training Fold: {fold} ...')
        X_train, X_test = X[features].iloc[train_index], X[features].iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train,
                  y_train,
                  eval_set = [(X_test, y_test)],
                  verbose = 512,)

        best_iteration = model.best_iteration

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        fold_logloss = log_loss(y_test, y_pred_proba)
        fold_roc = roc_auc_score(y_test, y_pred_proba)
        fold_rocs.append(fold_roc)
        fold_loglosses.append(fold_logloss)
        fold += 1

        test_pred = model.predict_proba(test_df[features])[:,1]
        fold_predictions.append(test_pred)

        print('....', '\n')

    predictions = np.mean(fold_predictions, axis=0)

    print("Fold Accuracies:", fold_rocs)
    print("Fold Log Losses:", fold_loglosses)
    print("Mean AUC:", sum(fold_rocs) / len(fold_rocs))
    print("Mean Log Loss:", sum(fold_loglosses) / len(fold_loglosses))

    return model, predictions


# Best Model Parameters...
'''
params = {'n_estimators'          : 2048,
          'max_depth'             : 9,
          'learning_rate'         : 0.05,
          'booster'               : 'gbtree',
          'subsample'             : 0.75,
          'colsample_bytree'      : 0.30,
          'reg_lambda'            : 1.00,
          'reg_alpha'             : 1.00,
          'gamma'                 : 1.00,
          'random_state'          : SEED,
          'objective'             : 'binary:logistic',
          'tree_method'           : 'gpu_hist',
          'eval_metric'           : 'auc',
          'early_stopping_rounds' : 256,
          'n_jobs'                : -1,
         }
'''


params = {'n_estimators'          : 2048,
          'max_depth'             : 9,
          'learning_rate'         : 0.045,
          'booster'               : 'gbtree',
          'subsample'             : 0.75,
          'colsample_bytree'      : 0.30,
          'reg_lambda'            : 1.00,
          'reg_alpha'             : 0.80,
          'gamma'                 : 0.80,
          'random_state'          : SEED,
          'objective'             : 'binary:logistic',
          'tree_method'           : 'gpu_hist',
          'eval_metric'           : 'auc',
          'early_stopping_rounds' : 256,
          'n_jobs'                : -1,
         }

xgboost_model, xgboost_predictions = fit_xgboost_with_kfold(train_df,
                                                            model_features,
                                                            target_variable='smoking',
                                                            parameters = params,
                                                            random_state=SEED,
                                                            n_splits = 10)


train_pred = xgboost_model.predict_proba(train_df[model_features])[:,1]
train_df['pred'] = train_pred
train_df[(train_df['smoking'] == 1) & (train_df['pred'] > 0.9)][model_features].sample(10).T

'''
submission = pd.read_csv('sample_submission.csv')
submission['smoking'] = xgboost_predictions
submission.to_csv('xgb_opt_submission.csv', index = False)
'''

test_df['predictions'] = xgboost_predictions
test_df.head()

cutoff = 0.95 # Probability CutOff...
pseudo_set_1 = test_df[test_df['predictions'] > cutoff]
pseudo_set_1['smoking'] = 1
pseudo_set_1.drop(columns=['predictions'], axis = 1, inplace=True)

pseudo_set_2 = test_df[test_df['predictions'] < 1 - cutoff]
pseudo_set_2['smoking'] = 0
pseudo_set_2.drop(columns=['predictions'], axis = 1, inplace=True)

pseudo_set = pd.concat([pseudo_set_1,pseudo_set_2])
pseudo_set.shape

pseudo_train_df = pd.concat([train_df, pseudo_set])

params = {'n_estimators': 2048,
          'max_depth': 9,
          'learning_rate': 0.045,
          'booster': 'gbtree',
          'subsample': 0.75,
          'colsample_bytree': 0.30,
          'reg_lambda': 1.00,
          'reg_alpha': 0.80,
          'gamma': 0.80,
          'random_state': SEED,
          'objective': 'binary:logistic',
          'tree_method': 'gpu_hist',
          'eval_metric': 'auc',
          'early_stopping_rounds': 256,
          'n_jobs': -1,
         }

xgboost_model, xgboost_predictions = fit_xgboost_with_kfold(pseudo_train_df,
                                                            model_features,
                                                            target_variable='smoking',
                                                            parameters=params,
                                                            random_state=SEED,
                                                            n_splits=10)

submission = pd.read_csv('sample_submission.csv')
submission['smoking'] = xgboost_predictions
submission.to_csv('xgb_pseudo_opt_submission.csv', index = False)



