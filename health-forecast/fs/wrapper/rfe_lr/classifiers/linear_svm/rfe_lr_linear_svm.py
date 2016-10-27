import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from utils_functions import save_object, performance_metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def stability(main_path, dataset_type, sampling):
    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/' + sampling + '/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    num_experiments = 10
    seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    feature_ranking = np.zeros((len(variable_names), num_experiments))

    for i in range(0, num_experiments):
        print("Loading experiment " + str(i) + " data...")
        print()
        raw_train_data = np.genfromtxt(main_path + dataset_type + '/' + sampling + '/experiment_' + str(i) + '_train.csv', delimiter=',')
        raw_train_data = raw_train_data[1:, :]

        X_train = raw_train_data[:, 1:]
        y_train = raw_train_data[:, 0]

        lr = LogisticRegression(random_state=seeds[i], class_weight="balanced", penalty='l1', dual=False, n_jobs=-1)

        rfe_lr_wrapper = RFE(lr, n_features_to_select=2000, step=0.25)

        linear_svm = SVC(kernel='linear', random_state=seeds[i], probability=True, class_weight='balanced')

        rfe_lr_linear_svm_pipe = Pipeline([('rfe_lr', rfe_lr_wrapper), ('linear_svm', linear_svm)])

        param_grid = {
            'linear_svm__C': C_OPTIONS
        }

        print("Performing gridsearch...")
        print()

        pipe_gridsearch = GridSearchCV(rfe_lr_linear_svm_pipe, param_grid=param_grid, n_jobs=12, scoring='f1_weighted',
                                       cv=StratifiedKFold(n_splits=5, random_state=seeds[i]))
        pipe_gridsearch.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(pipe_gridsearch.best_params_)
        print()

        selected_features = np.zeros(X_train.shape[1])
        selected_features[pipe_gridsearch.best_estimator_.named_steps['rfe_lr'].get_support()] = 1

        feature_ranking[:, i] = selected_features

    print("Calculating final feature ranking")
    print()

    final_ranking = np.sum(feature_ranking, axis=1)

    save_object(feature_ranking, os.getcwd() + '/' + sampling + '/' + dataset_type + '/feature_ranking.pkl')

    features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                             dtype=[('names', 'S120'), ('stability', '>i4')])
    features_info['names'] = variable_names
    features_info['stability'] = final_ranking

    with open(os.getcwd() + '/' + sampling + '/' + dataset_type + '/stability_features_info.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['names', 'stability'])
        w.writerows(features_info)


def general_performance(main_path, dataset_type, sampling):
    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/' + sampling + '/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    print("Loading experiment data...")
    print()
    raw_train_data = np.genfromtxt(main_path + dataset_type + '/' + sampling + '/raw_train.csv', delimiter=',')
    raw_train_data = raw_train_data[1:, :]

    X_train = raw_train_data[:, 1:]
    y_train = raw_train_data[:, 0]

    raw_test_data = np.genfromtxt(main_path + dataset_type + '/' + sampling + '/raw_test.csv', delimiter=',')
    raw_test_data = raw_test_data[1:, :]

    X_test = raw_test_data[:, 1:]
    y_test = raw_test_data[:, 0]

    experiment_results = dict()

    lr = LogisticRegression(random_state=123456, class_weight="balanced", penalty='l1', dual=False, n_jobs=-1)

    rfe_lr_wrapper = RFE(lr, n_features_to_select=2000, step=0.25)

    linear_svm = SVC(kernel='linear', random_state=123456, probability=True, class_weight='balanced')

    rfe_lr_linear_svm_pipe = Pipeline([('rfe_lr', rfe_lr_wrapper), ('linear_svm', linear_svm)])

    param_grid = {
        'linear_svm__C': C_OPTIONS
    }

    print("Performing gridsearch...")
    print()

    pipe_gridsearch = GridSearchCV(rfe_lr_linear_svm_pipe, param_grid=param_grid, n_jobs=12, scoring='f1_weighted',
                                   cv=StratifiedKFold(n_splits=5, random_state=123456))
    pipe_gridsearch.fit(X_train, y_train)

    rfe_lr_linear_svm_cv_results = dict()
    rfe_lr_linear_svm_cv_results['mean_test_score'] = pipe_gridsearch.cv_results_['mean_test_score']
    rfe_lr_linear_svm_cv_results['std_test_score'] = pipe_gridsearch.cv_results_['std_test_score']
    rfe_lr_linear_svm_cv_results['mean_train_score'] = pipe_gridsearch.cv_results_['mean_train_score']
    rfe_lr_linear_svm_cv_results['std_train_score'] = pipe_gridsearch.cv_results_['std_train_score']
    rfe_lr_linear_svm_cv_results['params'] = pipe_gridsearch.cv_results_['params']

    experiment_results['cv_results'] = rfe_lr_linear_svm_cv_results

    print("Best parameters set found on development set:")
    print()
    print(pipe_gridsearch.best_params_)
    print()

    # print("Estimator classes: ")
    # print()
    # print(pipe_gridsearch.best_estimator_.named_steps['linear_svm'].classes_)
    # print()

    performance_metrics(experiment_results, pipe_gridsearch.best_estimator_, 'rfe_lr', 'linear_svm', X_train, y_train,
                        X_test, y_test, dataset_type, variable_names, sampling)


if __name__ == '__main__':
    # main_path = '/home/mgvaldes/devel/MIRI/master-thesis/miri-master-thesis/health-forecast/datasets/'
    main_path = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast-project/health-forecast/datasets/'

    # sampling_types = ["raw", "down_sample", "up_sample", "smote_sample"]
    sampling_types = ["up_sample"]
    dataset_types = ["genomic", "genomic_epidemiological"]

    for sampling in sampling_types:
        for dataset_type in dataset_types:
            general_performance(main_path, dataset_type, sampling)
            stability(main_path, dataset_type, sampling)