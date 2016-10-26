import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.svm import SVC
from utils_functions import save_object, performance_metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def stability(main_path, dataset_type):
    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/raw/raw_train.csv', 'r') as csvfile:
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
        raw_train_data = np.genfromtxt(main_path + dataset_type + '/raw/experiment_' + str(i) + '_train.csv', delimiter=',')
        raw_train_data = raw_train_data[1:, :]

        X_train = raw_train_data[:, 1:]
        y_train = raw_train_data[:, 0]

        anova_filter = SelectPercentile(f_classif, percentile=10)

        linear_svm = SVC(kernel='linear', random_state=seeds[i], probability=True, class_weight='balanced')

        anova_linear_svm_pipe = Pipeline([('anova', anova_filter), ('linear_svm', linear_svm)])

        param_grid = {
            'linear_svm__C': C_OPTIONS
        }

        print("Performing gridsearch...")
        print()

        pipe_gridsearch = GridSearchCV(anova_linear_svm_pipe, param_grid=param_grid,
                                       cv=StratifiedKFold(n_splits=5, random_state=seeds[i]),
                                       n_jobs=12, scoring='f1_weighted')
        pipe_gridsearch.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(pipe_gridsearch.best_params_)
        print()

        selected_features = np.zeros(X_train.shape[1])
        selected_features[pipe_gridsearch.best_estimator_.named_steps['anova'].get_support()] = 1

        feature_ranking[:, i] = selected_features

    print("Calculating final feature ranking")
    print()

    final_ranking = np.sum(feature_ranking, axis=1)

    save_object(feature_ranking, os.getcwd() + '/' + dataset_type + '/feature_ranking.pkl')

    features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                             dtype=[('names', 'S120'), ('stability', '>i4')])
    features_info['names'] = variable_names
    features_info['stability'] = final_ranking

    with open(os.getcwd() + '/' + dataset_type + '/stability_features_info.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['names', 'stability'])
        w.writerows(features_info)


def general_performance(main_path, dataset_type):
    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/raw/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    print("Loading experiment data...")
    print()
    raw_train_data = np.genfromtxt(main_path + dataset_type + '/raw/raw_train.csv', delimiter=',')
    raw_train_data = raw_train_data[1:, :]

    X_train = raw_train_data[:, 1:]
    y_train = raw_train_data[:, 0]

    raw_test_data = np.genfromtxt(main_path + dataset_type + '/raw/raw_test.csv', delimiter=',')
    raw_test_data = raw_test_data[1:, :]

    X_test = raw_test_data[:, 1:]
    y_test = raw_test_data[:, 0]

    experiment_results = dict()

    anova_filter = SelectPercentile(f_classif, percentile=10)

    linear_svm = SVC(kernel='linear', random_state=123456, probability=True, class_weight='balanced')

    anova_linear_svm_pipe = Pipeline([('anova', anova_filter), ('linear_svm', linear_svm)])

    param_grid = {
        'linear_svm__C': C_OPTIONS
    }

    print("Performing gridsearch...")
    print()

    pipe_gridsearch = GridSearchCV(anova_linear_svm_pipe, param_grid=param_grid, cv=StratifiedKFold(n_splits=5, random_state=123456),
                                   n_jobs=12, scoring='f1_weighted')
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

    performance_metrics(experiment_results, pipe_gridsearch.best_estimator_, 'anova', 'linear_svm', X_train, y_train, X_test, y_test, dataset_type,
                        variable_names)


if __name__ == '__main__':
    # main_path = '/home/mgvaldes/devel/MIRI/master-thesis/miri-master-thesis/health-forecast/datasets/'
    main_path = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/datasets/'
    dataset_type = 'genomic'

    general_performance(main_path, dataset_type)
    # stability(main_path, dataset_type)

    dataset_type = 'genomic_epidemiological'

    general_performance(main_path, dataset_type)
    # stability(main_path, dataset_type)

# import numpy as np
# import csv
# import os
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import f_classif, SelectPercentile
# from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import roc_curve, auc, f1_score
# from plot_functions import plot_confusion_matrix, plot_roc
# from utils_functions import save_object
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
#
# experiments_results = dict()
#
# num_experiments = 10
#
# seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# PERCENTILE_OPTIONS = [10]
#
# print("Loading variable names...")
# print()
# with open('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_train.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row in reader:
#         variable_names = np.array(list(row))
#         break
#
# variable_names = variable_names[1:]
#
# feature_ranking = np.zeros((len(variable_names), num_experiments))
# F1s = []
# coefficients = np.zeros((len(variable_names), num_experiments))
#
# for i in range(0, num_experiments):
#     experiment_results = dict()
#
#     print("Loading experiment " + str(i) + " data...")
#     print()
#     raw_train_data = np.genfromtxt(
#         '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/experiment_' + str(i) + '_train.csv',
#         delimiter=',')
#     raw_train_data = raw_train_data[1:, :]
#
#     X_train = raw_train_data[:, 1:]
#     y_train = raw_train_data[:, 0]
#
#     raw_test_data = np.genfromtxt(
#         '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/experiment_' + str(i) + '_test.csv',
#         delimiter=',')
#     raw_test_data = raw_test_data[1:, :]
#
#     X_test = raw_test_data[:, 1:]
#     y_test = raw_test_data[:, 0]
#
#     anova_filter = SelectPercentile(f_classif)
#
#     # linear_svm = LinearSVC(penalty='l1', random_state=seeds[i], dual=False)
#     linear_svm = SVC(kernel='linear', random_state=seeds[i], probability=True, class_weight='balanced')
#
#     anova_linear_svm_pipe = Pipeline([('anova', anova_filter), ('linear_svm', linear_svm)])
#
#     param_grid = {
#         'anova__percentile': PERCENTILE_OPTIONS,
#         'linear_svm__C': C_OPTIONS
#     }
#
#     skf = StratifiedKFold(n_splits=5, random_state=seeds[i])
#
#     linear_svm_gridsearch = GridSearchCV(anova_linear_svm_pipe, param_grid=param_grid, cv=skf, n_jobs=11, scoring='f1_weighted')
#
#     print("Performing automatic gridsearch over C parameter, including feature selection over each fold")
#     print()
#     linear_svm_gridsearch.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(linear_svm_gridsearch.best_params_)
#     print()
#
#     anova_filter = SelectPercentile(f_classif, percentile=10)
#
#     # linear_svm = LinearSVC(penalty='l1', random_state=seeds[i], dual=False, C=linear_svm_gridsearch.best_params_['linear_svm__C'])
#     linear_svm = SVC(kernel='linear', random_state=seeds[i], probability=True, class_weight='balanced', C=linear_svm_gridsearch.best_params_['linear_svm__C'])
#
#     anova_linear_svm_pipe = Pipeline([('anova', anova_filter), ('linear_svm', linear_svm)])
#
#     anova_linear_svm_pipe.fit(X_train, y_train)
#
#     experiment_results['fs'] = anova_linear_svm_pipe.named_steps['anova']
#
#     selected_features = np.zeros(X_train.shape[1])
#     selected_features[anova_linear_svm_pipe.named_steps['anova'].get_support()] = 1
#
#     feature_ranking[:, i] = selected_features
#
#     experiment_results['model'] = anova_linear_svm_pipe.named_steps['linear_svm']
#
#     linear_svm_coefficients = np.zeros(X_train.shape[1])
#     linear_svm_coefficients[anova_linear_svm_pipe.named_steps['anova'].get_support()] = np.absolute(anova_linear_svm_pipe.named_steps['linear_svm'].coef_)
#
#     coefficients[:, i] = linear_svm_coefficients
#
#     print("Predicting y_test with reduced X_test")
#     print()
#     # Use only selected features of test set
#     y_pred = anova_linear_svm_pipe.predict(X_test)
#
#     # print("Decision function:")
#     # y_score = anova_linear_svm_pipe.decision_function(X_test)
#     # experiment_results['y_score'] = y_score
#     # print(y_score)
#     # print()
#
#     print("Probabilities:")
#     y_prob = anova_linear_svm_pipe.predict_proba(X_test)
#     experiment_results['y_prob'] = y_prob
#     print(y_prob)
#     print()
#
#     print()
#     print("Accuracy:")
#     linear_svm_accuracy = accuracy_score(y_test, y_pred)
#     experiment_results['accuracy'] = linear_svm_accuracy
#     print(linear_svm_accuracy)
#     print()
#
#     print("Confusion matrix:")
#     print()
#     linear_svm_confusion_matrix = confusion_matrix(y_test, y_pred)
#     experiment_results['confusion_matrix'] = linear_svm_confusion_matrix
#     print(linear_svm_confusion_matrix)
#     print()
#
#     plot_confusion_matrix(linear_svm_confusion_matrix, classes=["Positive", "Negative"],
#                           filename=os.getcwd() + '/experiment_' + str(
#                               i) + '_confusion_matrix.png')
#
#     linear_svm_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred)
#     experiment_results['precision_recall_f1'] = linear_svm_precision_recall_fscore_support
#
#     pos_precision = linear_svm_precision_recall_fscore_support[0][0]
#     neg_precision = linear_svm_precision_recall_fscore_support[0][1]
#     pos_recall = linear_svm_precision_recall_fscore_support[1][0]
#     neg_recall = linear_svm_precision_recall_fscore_support[1][1]
#     pos_f1 = linear_svm_precision_recall_fscore_support[2][0]
#     neg_f1 = linear_svm_precision_recall_fscore_support[2][1]
#
#     print("Positive precision:")
#     print(pos_precision)
#     print()
#
#     print("Positive recall:")
#     print(pos_recall)
#     print()
#
#     print("Positive F1:")
#     print(pos_f1)
#     print()
#
#     print("Negative precision:")
#     print(neg_precision)
#     print()
#
#     print("Negative recall:")
#     print(neg_recall)
#     print()
#
#     print("Negative F1:")
#     print(neg_f1)
#     print()
#
#     print("WEIGHTED F1:")
#     linear_svm_f1 = f1_score(y_test, y_pred, average='weighted')
#     experiment_results['weighted_F1'] = linear_svm_f1
#     F1s.append(linear_svm_f1)
#     print(linear_svm_f1)
#     print()
#
#     fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
#     pos_auc = auc(fpr, tpr)
#     experiment_results['fpr'] = fpr
#     experiment_results['tpr'] = tpr
#     experiment_results['pos_auc'] = pos_auc
#     plot_roc(fpr, tpr, pos_auc, "Positive ROC",
#              filename=os.getcwd() + '/experiment_' + str(i) + '_pos_roc.png')
#
#     print("auc pos:")
#     print(pos_auc)
#     print()
#
#     fnr, tnr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)
#     neg_auc = auc(fnr, tnr)
#     experiment_results['fnr'] = fnr
#     experiment_results['tnr'] = tnr
#     experiment_results['neg_auc'] = neg_auc
#     plot_roc(fnr, tnr, neg_auc, "Negative ROC",
#              filename=os.getcwd() + '/experiment_' + str(i) + '_neg_roc.png')
#
#     print("auc neg:")
#     print(neg_auc)
#     print()
#
#     experiments_results[i] = experiment_results
#
# final_ranking = np.sum(feature_ranking, axis=1)
#
# print("Variables selected in ALL experiments:")
# print(variable_names[final_ranking == num_experiments])
# print()
#
# save_object(experiments_results, os.getcwd() + '/linear_svm_results.pkl')
# save_object(feature_ranking, os.getcwd() + '/feature_ranking.pkl')
#
# features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)), np.repeat(0, len(variable_names)), np.repeat(0, len(variable_names)))), dtype=[('names', 'S12'), ('stability', '>i4'), ('performance (F1)', '>f4'), ('coefficients', '>f4')])
# features_info['names'] = variable_names
# features_info['stability'] = final_ranking
# features_info['performance (F1)'] = np.repeat(np.mean(np.array(F1s)), len(variable_names))
# features_info['coefficients'] = np.mean(coefficients, axis=1)
#
# with open(os.getcwd() + '/features_info.csv', 'wb') as f:
#     w = csv.writer(f)
#     w.writerow(['names', 'stability', 'performance (F1)', 'linear SVM coefficients'])
#     w.writerows(features_info)