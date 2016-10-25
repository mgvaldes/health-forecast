import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, f1_score
from plot_functions import plot_confusion_matrix, plot_roc
from utils_functions import save_object
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression

C_OPTIONS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
LR_C_OPTIONS = [0.1, 1, 10, 100, 1000]

print("Loading variable names...")
print()
with open('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        variable_names = np.array(list(row))
        break

variable_names = variable_names[1:]

experiment_results = dict()

print("Loading experiment data...")
print()
raw_train_data = np.genfromtxt('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_train.csv',
    delimiter=',')
raw_train_data = raw_train_data[1:, :]

X_train = raw_train_data[:, 1:]
y_train = raw_train_data[:, 0]

raw_test_data = np.genfromtxt('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_test.csv',
    delimiter=',')
raw_test_data = raw_test_data[1:, :]

X_test = raw_test_data[:, 1:]
y_test = raw_test_data[:, 0]

experiment_results = dict()

reg_log_reg = LogisticRegression(random_state=123456, class_weight="balanced", penalty='l1', dual=False, n_jobs=-1)

fs_reg_log_reg = SelectFromModel(reg_log_reg)

linear_svm = SVC(kernel='linear', random_state=123456, probability=True, class_weight='balanced')

fs_reg_log_reg_linear_svm_pipe = Pipeline([('fs_reg_log_reg', fs_reg_log_reg), ('linear_svm', linear_svm)])

param_grid = {
    'fs_reg_log_reg__estimator__C': LR_C_OPTIONS,
    'linear_svm__C': C_OPTIONS
}

pipe_gridsearch = GridSearchCV(estimator=fs_reg_log_reg_linear_svm_pipe, cv=StratifiedKFold(n_splits=5, random_state=123456),
                               n_jobs=12, param_grid=param_grid, scoring='f1_weighted')
pipe_gridsearch.fit(X_train, y_train)

reg_log_reg_cv_results = dict()
reg_log_reg_cv_results['mean_test_score'] = pipe_gridsearch.cv_results_['mean_test_score']
reg_log_reg_cv_results['std_test_score'] = pipe_gridsearch.cv_results_['std_test_score']
reg_log_reg_cv_results['mean_train_score'] = pipe_gridsearch.cv_results_['mean_train_score']
reg_log_reg_cv_results['std_train_score'] = pipe_gridsearch.cv_results_['std_train_score']
reg_log_reg_cv_results['params'] = pipe_gridsearch.cv_results_['params']

experiment_results['cv_results'] = reg_log_reg_cv_results

print("Best parameters set found on development set:")
print()
print(pipe_gridsearch.best_params_)
print()

cv_score = np.mean(cross_val_score(pipe_gridsearch.best_estimator_, X_train, y_train, n_jobs=12,
                                   cv=StratifiedKFold(n_splits=5, random_state=789012), scoring='f1_weighted'))
experiment_results['cv_score'] = cv_score

print("CV score:")
print()
print(cv_score)
print()

print("Selected variables: %s" % (len(variable_names[pipe_gridsearch.best_estimator_.named_steps['fs_reg_log_reg'].get_support()])))
print()
print(variable_names[pipe_gridsearch.best_estimator_.named_steps['fs_reg_log_reg'].get_support()])
print()

train_score = pipe_gridsearch.best_estimator_.score(X_train, y_train)
experiment_results['train_score'] = train_score

print("Train score:")
print()
print(train_score)
print()

y_pred = pipe_gridsearch.best_estimator_.predict(X_test)

print("Predicting y_test with reduced X_test")
print()

y_prob = pipe_gridsearch.best_estimator_.predict_proba(X_test)
experiment_results['y_prob'] = y_prob

print("Probabilities:")
print()
print(y_prob)
print()

linear_svm_accuracy = accuracy_score(y_test, y_pred)
experiment_results['accuracy'] = linear_svm_accuracy

print("Accuracy:")
print()
print(linear_svm_accuracy)
print()

linear_svm_confusion_matrix = confusion_matrix(y_test, y_pred)
experiment_results['confusion_matrix'] = linear_svm_confusion_matrix

print("Confusion matrix:")
print()
print(linear_svm_confusion_matrix)
print()

plot_confusion_matrix(linear_svm_confusion_matrix, classes=["Positive", "Negative"],
                      filename=os.getcwd() + '/confusion_matrix.png')

linear_svm_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred)
experiment_results['precision_recall_f1'] = linear_svm_precision_recall_fscore_support

pos_precision = linear_svm_precision_recall_fscore_support[0][0]
neg_precision = linear_svm_precision_recall_fscore_support[0][1]
pos_recall = linear_svm_precision_recall_fscore_support[1][0]
neg_recall = linear_svm_precision_recall_fscore_support[1][1]
pos_f1 = linear_svm_precision_recall_fscore_support[2][0]
neg_f1 = linear_svm_precision_recall_fscore_support[2][1]

print("Positive precision:")
print()
print(pos_precision)
print()

print("Positive recall:")
print()
print(pos_recall)
print()

print("Positive F1:")
print()
print(pos_f1)
print()

print("Negative precision:")
print()
print(neg_precision)
print()

print("Negative recall:")
print()
print(neg_recall)
print()

print("Negative F1:")
print()
print(neg_f1)
print()

linear_svm_f1 = f1_score(y_test, y_pred, average='weighted')
experiment_results['weighted_F1'] = linear_svm_f1

print("WEIGHTED F1:")
print()
print(linear_svm_f1)
print()

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
pos_auc = auc(fpr, tpr)
experiment_results['fpr'] = fpr
experiment_results['tpr'] = tpr
experiment_results['pos_auc'] = pos_auc

plot_roc(fpr, tpr, pos_auc, "Positive ROC",
         filename=os.getcwd() + '/pos_roc.png')

print("auc pos:")
print()
print(pos_auc)
print()

fnr, tnr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)
neg_auc = auc(fnr, tnr)
experiment_results['fnr'] = fnr
experiment_results['tnr'] = tnr
experiment_results['neg_auc'] = neg_auc

plot_roc(fnr, tnr, neg_auc, "Negative ROC",
         filename=os.getcwd() + '/neg_roc.png')

print("auc neg:")
print()
print(neg_auc)
print()

save_object(experiment_results, os.getcwd() + '/linear_svm_results.pkl')
