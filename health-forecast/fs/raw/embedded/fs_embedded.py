import numpy as np
import csv
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from plot_functions import plot_confusion_matrix, plot_roc
from utils_functions import save_object, load_object, save_dict, load_dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
import os
import math


def select_features(model, X_train, y_train):
    select_from_model = SelectFromModel(model, threshold=0.25)
    select_from_model.fit(X_train, y_train)

    selected_features = np.zeros(X_train.shape[1])
    selected_features[select_from_model.get_support()] = 1

    return selected_features


def regularized_logistic_regression(X_train, y_train, C_OPTIONS, PENALTY_OPTIONS, seed):
    param_grid = {
        'C': C_OPTIONS,
        'penalty': PENALTY_OPTIONS
    }

    reg_log_reg = LogisticRegression(random_state=seed, class_weight="balanced")

    reg_log_reg_gridsearch = GridSearchCV(estimator=reg_log_reg, cv=StratifiedKFold(n_splits=5, random_state=seed),
                                          n_jobs=11, param_grid=param_grid)
    reg_log_reg_gridsearch.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(reg_log_reg_gridsearch.best_params_)
    print()

    return select_features(reg_log_reg_gridsearch.best_estimator_, X_train, y_train)

if __name__ == '__main__':
    num_experiments = 10
    seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    PENALTY_OPTIONS = ['l1', 'l2']

    print("Loading variable names...")
    print()
    with open('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    feature_ranking = np.zeros((len(variable_names), num_experiments))

    for i in range(0, num_experiments):
        print("Loading experiment " + str(i) + " data...")
        print()
        raw_train_data = np.genfromtxt('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/experiment_' + str(i) + '_train.csv',
                                   delimiter=',')
        raw_train_data = raw_train_data[1:, :]

        # print("Train dataset:")
        raw_train_data_n_rows = raw_train_data.shape[0]
        # print('nrows: ' + str(raw_train_data_n_rows))

        raw_train_data_n_cols = raw_train_data.shape[1]
        # print('ncols: ' + str(raw_train_data_n_cols))
        # print()

        X_train = raw_train_data[:, 1:]
        y_train = raw_train_data[:, 0]

        print("Performing embedded FS...")
        print()
        feature_ranking[:, i] = regularized_logistic_regression(X_train, y_train, C_OPTIONS, PENALTY_OPTIONS, 0)

    final_ranking = np.sum(feature_ranking, axis=1)

    print("Variables selected in ALL experiments:")
    print(variable_names[final_ranking == num_experiments])
    print()

    print("Variables selected in 80% of experiments:")
    print(variable_names[final_ranking == math.floor(0.8 * num_experiments)])
    print()

    print("Variables selected in 50% of experiments:")
    print(variable_names[final_ranking == math.floor(0.8 * num_experiments)])
    print()
