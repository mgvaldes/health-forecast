import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from plot_functions import plot_confusion_matrix, plot_roc
import math
from utils_functions import save_object
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

experiments_results = dict()

num_experiments = 10

seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

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
    experiment_results = dict()

    print("Loading experiment " + str(i) + " data...")
    print()
    raw_train_data = np.genfromtxt(
        '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/down_sample/experiment_' + str(i) + '_down_sample_train.csv',
        delimiter=',')
    raw_train_data = raw_train_data[1:, :]

    X_train = raw_train_data[:, 1:]
    y_train = raw_train_data[:, 0]

    raw_test_data = np.genfromtxt(
        '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/down_sample/experiment_' + str(i) + '_test.csv',
        delimiter=',')
    raw_test_data = raw_test_data[1:, :]

    X_test = raw_test_data[:, 1:]
    y_test = raw_test_data[:, 0]

    anova_filter = SelectPercentile(f_classif)

    # linear_svm = LinearSVC(penalty='l1', random_state=seeds[i], dual=False)
    linear_svm = SVC(kernel='linear', random_state=seeds[i], probability=True, class_weight='balanced')

    anova_linear_svm_pipe = Pipeline([('anova', anova_filter), ('linear_svm', linear_svm)])

    param_grid = {
        'anova__percentile': 10,
        'linear_svm__C': C_OPTIONS
    }

    skf = StratifiedKFold(n_splits=5, random_state=seeds[i])

    linear_svm_gridsearch = GridSearchCV(anova_linear_svm_pipe, param_grid=param_grid, cv=skf, n_jobs=11, scoring='f1')

    print("Performing automatic gridsearch over C parameter, including feature selection over each fold")
    print()
    linear_svm_gridsearch.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(linear_svm_gridsearch.best_params_)
    print()

    anova_filter = SelectPercentile(f_classif, percentile=10)

    # linear_svm = LinearSVC(penalty='l1', random_state=seeds[i], dual=False, C=linear_svm_gridsearch.best_params_['linear_svm__C'])
    linear_svm = SVC(kernel='linear', random_state=seeds[i], probability=True, class_weight='balanced', C=linear_svm_gridsearch.best_params_['linear_svm__C'])

    anova_linear_svm_pipe = Pipeline([('anova', anova_filter), ('linear_svm', linear_svm)])

    anova_linear_svm_pipe.fit(X_train, y_train)

    experiment_results['fs'] = anova_linear_svm_pipe.named_steps['anova']

    selected_features = np.zeros(X_train.shape[1])
    selected_features[anova_linear_svm_pipe.named_steps['anova'].get_support()] = 1

    feature_ranking[:, i] = selected_features

    experiment_results['model'] = anova_linear_svm_pipe.named_steps['linear_svm']

    print("Predicting y_test with reduced X_test")
    # Use only selected features of test set
    y_pred = anova_linear_svm_pipe.predict(X_test)

    # print("Decision function:")
    # y_score = anova_linear_svm_pipe.decision_function(X_test)
    # experiment_results['y_score'] = y_score
    # print(y_score)
    # print()

    print("Probabilities:")
    y_prob = anova_linear_svm_pipe.predict_proba(X_test)
    experiment_results['y_prob'] = y_prob
    print(y_prob)
    print()

    print()
    print("Accuracy:")
    linear_svm_accuracy = accuracy_score(y_test, y_pred)
    experiment_results['accuracy'] = linear_svm_accuracy
    print(linear_svm_accuracy)
    print()

    print("Confusion matrix:")
    print()
    linear_svm_confusion_matrix = confusion_matrix(y_test, y_pred)
    experiment_results['confusion_matrix'] = linear_svm_confusion_matrix
    print(linear_svm_confusion_matrix)
    print()

    plot_confusion_matrix(linear_svm_confusion_matrix, classes=["Positive", "Negative"],
                          filename=os.getcwd() + '/classifiers/linear_svm2/experiment_' + str(
                              i) + '_down_sample_confusion_matrix.png')

    linear_svm_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred)
    experiment_results['precision_recall_f1'] = linear_svm_precision_recall_fscore_support

    pos_precision = linear_svm_precision_recall_fscore_support[0][0]
    neg_precision = linear_svm_precision_recall_fscore_support[0][1]
    pos_recall = linear_svm_precision_recall_fscore_support[1][0]
    neg_recall = linear_svm_precision_recall_fscore_support[1][1]
    pos_f1 = linear_svm_precision_recall_fscore_support[2][0]
    neg_f1 = linear_svm_precision_recall_fscore_support[2][1]

    print("Positive precision:")
    print(pos_precision)
    print()

    print("Positive recall:")
    print(pos_recall)
    print()

    print("Positive F1:")
    print(pos_f1)
    print()

    print("Negative precision:")
    print(neg_precision)
    print()

    print("Negative recall:")
    print(neg_recall)
    print()

    print("Negative F1:")
    print(neg_f1)
    print()

    print("F1:")
    linear_svm_f1 = np.mean([pos_f1, neg_f1])
    experiment_results['F1'] = linear_svm_f1
    print(linear_svm_f1)
    print()

    print("AUC:")
    linear_svm_auc = roc_auc_score(y_test, y_prob)
    experiment_results['AUC'] = linear_svm_auc
    print(linear_svm_auc)
    print()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=0)
    pos_auc = auc(fpr, tpr)
    experiment_results['fpr'] = fpr
    experiment_results['tpr'] = tpr
    experiment_results['pos_auc'] = pos_auc
    plot_roc(fpr, tpr, pos_auc, "Positive ROC",
             filename=os.getcwd() + '/classifiers/linear_svm2/experiment_' + str(i) + '_down_sample_pos_roc.png')

    fnr, tnr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    neg_auc = auc(fnr, tnr)
    experiment_results['fnr'] = fnr
    experiment_results['tnr'] = tnr
    experiment_results['neg_auc'] = neg_auc
    plot_roc(fnr, tnr, neg_auc, "Negative ROC",
             filename=os.getcwd() + '/classifiers/linear_svm2/experiment_' + str(i) + '_down_sample_neg_roc.png')

    experiments_results[i] = experiment_results

final_ranking = np.sum(feature_ranking, axis=1)

print("Final variable selection:")
print(final_ranking)
print()

print("Variables selected in ALL experiments:")
print(variable_names[final_ranking == num_experiments])
print()

print("Variables selected in 80% of experiments:")
print(variable_names[final_ranking == math.floor(0.8 * num_experiments)])
print()

print("Variables selected in 50% of experiments:")
print(variable_names[final_ranking == math.floor(0.5 * num_experiments)])
print()

save_object(experiments_results, os.getcwd() + '/classifiers/linear_svm2/linear_svm_results.pkl')
save_object(feature_ranking, os.getcwd() + '/classifiers/linear_svm2/feature_ranking.pkl')