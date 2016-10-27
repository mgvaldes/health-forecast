import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
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
    LR_C_OPTIONS = [0.1, 1, 10, 100, 1000]
    feature_ranking = np.zeros((len(variable_names), num_experiments))

    for i in range(0, num_experiments):
        print("Loading experiment " + str(i) + " data...")
        print()
        raw_train_data = np.genfromtxt(main_path + dataset_type + '/' + sampling + '/experiment_' + str(i) + '_train.csv', delimiter=',')
        raw_train_data = raw_train_data[1:, :]

        X_train = raw_train_data[:, 1:]
        y_train = raw_train_data[:, 0]

        reg_log_reg = LogisticRegression(random_state=seeds[i], class_weight="balanced", penalty='l1', dual=False,
                                         n_jobs=-1)

        reg_log_reg_embedded = SelectFromModel(reg_log_reg)

        linear_svm = SVC(kernel='linear', random_state=seeds[i], probability=True, class_weight='balanced')

        anova_linear_svm_pipe = Pipeline([('reg_log_reg', reg_log_reg_embedded), ('linear_svm', linear_svm)])

        param_grid = {
            'reg_log_reg__estimator__C': LR_C_OPTIONS,
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
        selected_features[pipe_gridsearch.best_estimator_.named_steps['reg_log_reg'].get_support()] = 1

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
    LR_C_OPTIONS = [10, 50, 100, 500, 1000, 1500]

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

    reg_log_reg = LogisticRegression(random_state=123456, class_weight="balanced", penalty='l1', dual=False, n_jobs=-1)

    reg_log_reg_embedded = SelectFromModel(reg_log_reg)

    linear_svm = SVC(kernel='linear', random_state=123456, probability=True, class_weight='balanced')

    anova_linear_svm_pipe = Pipeline([('reg_log_reg', reg_log_reg_embedded), ('linear_svm', linear_svm)])

    param_grid = {
        'reg_log_reg__estimator__C': LR_C_OPTIONS,
        'linear_svm__C': C_OPTIONS
    }

    print("Performing gridsearch...")
    print()

    pipe_gridsearch = GridSearchCV(anova_linear_svm_pipe, param_grid=param_grid, n_jobs=12, scoring='f1_weighted',
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

    performance_metrics(experiment_results, pipe_gridsearch.best_estimator_, 'reg_log_reg', 'linear_svm', X_train, y_train,
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


# import numpy as np
# import csv
# import os
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectFromModel, SelectPercentile
# from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import roc_curve, auc, f1_score
# from plot_functions import plot_confusion_matrix, plot_roc
# from utils_functions import save_object
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV, cross_val_score
# from sklearn.linear_model import LogisticRegression
#
# C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# LR_C_OPTIONS = [0.1, 1, 10, 100, 1000]
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
# experiment_results = dict()
#
# print("Loading experiment data...")
# print()
# raw_train_data = np.genfromtxt('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_train.csv',
#     delimiter=',')
# raw_train_data = raw_train_data[1:, :]
#
# X_train = raw_train_data[:, 1:]
# y_train = raw_train_data[:, 0]
#
# raw_test_data = np.genfromtxt('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_test.csv',
#     delimiter=',')
# raw_test_data = raw_test_data[1:, :]
#
# X_test = raw_test_data[:, 1:]
# y_test = raw_test_data[:, 0]
#
# experiment_results = dict()
#
# reg_log_reg = LogisticRegression(random_state=123456, class_weight="balanced", penalty='l1', dual=False, n_jobs=-1)
#
# fs_reg_log_reg = SelectFromModel(reg_log_reg)
#
# linear_svm = SVC(kernel='linear', random_state=123456, probability=True, class_weight='balanced')
#
# fs_reg_log_reg_linear_svm_pipe = Pipeline([('fs_reg_log_reg', fs_reg_log_reg), ('linear_svm', linear_svm)])
#
# param_grid = {
#     'fs_reg_log_reg__estimator__C': LR_C_OPTIONS,
#     'linear_svm__C': C_OPTIONS
# }
#
# pipe_gridsearch = GridSearchCV(estimator=fs_reg_log_reg_linear_svm_pipe, cv=StratifiedKFold(n_splits=5, random_state=123456),
#                                n_jobs=12, param_grid=param_grid, scoring='f1_weighted')
# pipe_gridsearch.fit(X_train, y_train)
#
# reg_log_reg_cv_results = dict()
# reg_log_reg_cv_results['mean_test_score'] = pipe_gridsearch.cv_results_['mean_test_score']
# reg_log_reg_cv_results['std_test_score'] = pipe_gridsearch.cv_results_['std_test_score']
# reg_log_reg_cv_results['mean_train_score'] = pipe_gridsearch.cv_results_['mean_train_score']
# reg_log_reg_cv_results['std_train_score'] = pipe_gridsearch.cv_results_['std_train_score']
# reg_log_reg_cv_results['params'] = pipe_gridsearch.cv_results_['params']
#
# experiment_results['cv_results'] = reg_log_reg_cv_results
#
# print("Best parameters set found on development set:")
# print()
# print(pipe_gridsearch.best_params_)
# print()
#
# cv_score = np.mean(cross_val_score(pipe_gridsearch.best_estimator_, X_train, y_train, n_jobs=12,
#                                    cv=StratifiedKFold(n_splits=5, random_state=789012), scoring='f1_weighted'))
# experiment_results['cv_score'] = cv_score
#
# print("CV score:")
# print()
# print(cv_score)
# print()
#
# print("Selected variables: %s" % (len(variable_names[pipe_gridsearch.best_estimator_.named_steps['fs_reg_log_reg'].get_support()])))
# print()
# print(variable_names[pipe_gridsearch.best_estimator_.named_steps['fs_reg_log_reg'].get_support()])
# print()
#
# train_score = pipe_gridsearch.best_estimator_.score(X_train, y_train)
# experiment_results['train_score'] = train_score
#
# print("Train score:")
# print()
# print(train_score)
# print()
#
# y_pred = pipe_gridsearch.best_estimator_.predict(X_test)
#
# print("Predicting y_test with reduced X_test")
# print()
#
# y_prob = pipe_gridsearch.best_estimator_.predict_proba(X_test)
# experiment_results['y_prob'] = y_prob
#
# print("Probabilities:")
# print()
# print(y_prob)
# print()
#
# linear_svm_accuracy = accuracy_score(y_test, y_pred)
# experiment_results['accuracy'] = linear_svm_accuracy
#
# print("Accuracy:")
# print()
# print(linear_svm_accuracy)
# print()
#
# linear_svm_confusion_matrix = confusion_matrix(y_test, y_pred)
# experiment_results['confusion_matrix'] = linear_svm_confusion_matrix
#
# print("Confusion matrix:")
# print()
# print(linear_svm_confusion_matrix)
# print()
#
# plot_confusion_matrix(linear_svm_confusion_matrix, classes=["Positive", "Negative"],
#                       filename=os.getcwd() + '/confusion_matrix.png')
#
# linear_svm_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred)
# experiment_results['precision_recall_f1'] = linear_svm_precision_recall_fscore_support
#
# pos_precision = linear_svm_precision_recall_fscore_support[0][0]
# neg_precision = linear_svm_precision_recall_fscore_support[0][1]
# pos_recall = linear_svm_precision_recall_fscore_support[1][0]
# neg_recall = linear_svm_precision_recall_fscore_support[1][1]
# pos_f1 = linear_svm_precision_recall_fscore_support[2][0]
# neg_f1 = linear_svm_precision_recall_fscore_support[2][1]
#
# print("Positive precision:")
# print()
# print(pos_precision)
# print()
#
# print("Positive recall:")
# print()
# print(pos_recall)
# print()
#
# print("Positive F1:")
# print()
# print(pos_f1)
# print()
#
# print("Negative precision:")
# print()
# print(neg_precision)
# print()
#
# print("Negative recall:")
# print()
# print(neg_recall)
# print()
#
# print("Negative F1:")
# print()
# print(neg_f1)
# print()
#
# linear_svm_f1 = f1_score(y_test, y_pred, average='weighted')
# experiment_results['weighted_F1'] = linear_svm_f1
#
# print("WEIGHTED F1:")
# print()
# print(linear_svm_f1)
# print()
#
# fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
# pos_auc = auc(fpr, tpr)
# experiment_results['fpr'] = fpr
# experiment_results['tpr'] = tpr
# experiment_results['pos_auc'] = pos_auc
#
# plot_roc(fpr, tpr, pos_auc, "Positive ROC",
#          filename=os.getcwd() + '/pos_roc.png')
#
# print("auc pos:")
# print()
# print(pos_auc)
# print()
#
# fnr, tnr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)
# neg_auc = auc(fnr, tnr)
# experiment_results['fnr'] = fnr
# experiment_results['tnr'] = tnr
# experiment_results['neg_auc'] = neg_auc
#
# plot_roc(fnr, tnr, neg_auc, "Negative ROC",
#          filename=os.getcwd() + '/neg_roc.png')
#
# print("auc neg:")
# print()
# print(neg_auc)
# print()
#
# save_object(experiment_results, os.getcwd() + '/linear_svm_results.pkl')
