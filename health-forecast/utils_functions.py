from sklearn.externals import joblib
import pickle
import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, f1_score
from plot_functions import plot_confusion_matrix, plot_roc
from sklearn.model_selection import cross_val_score


def save_object(obj, filename):
    joblib.dump(obj, filename, compress=1)


def load_object(filename):
    return joblib.load(filename)


def save_dict(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)


def load_dict(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def performance_metrics(experiment_results, best_estimator, X_train, y_train, X_test, y_test, dataset_type, variable_names):
    experiment_results['best_estimator'] = best_estimator

    cv_score = np.mean(cross_val_score(best_estimator, X_train, y_train, n_jobs=12,
                                       cv=StratifiedKFold(n_splits=5, random_state=789012), scoring='f1_weighted'))


    experiment_results['cv_score'] = cv_score

    print("CV score:")
    print()
    print(cv_score)
    print()

    y_train_pred = best_estimator.predict(X_train)
    train_score = f1_score(y_train, y_train_pred, average='weighted')
    experiment_results['train_score'] = train_score

    print("Train score:")
    print()
    print(train_score)
    print()

    y_pred = best_estimator.predict(X_test)

    print("Predicting y_test with reduced X_test")
    print()
    print(y_pred)
    print()

    y_prob = best_estimator.predict_proba(X_test)
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
                          filename=os.getcwd() + '/' + dataset_type + '/confusion_matrix.png')

    linear_svm_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred, pos_label=0)
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

    linear_svm_f1 = f1_score(y_test, y_pred, average='weighted', pos_label=0)
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
             filename=os.getcwd() + '/' + dataset_type + '/pos_roc.png')

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
             filename=os.getcwd() + '/' + dataset_type + '/neg_roc.png')

    print("auc neg:")
    print()
    print(neg_auc)
    print()

    save_object(experiment_results, os.getcwd() + '/' + dataset_type + '/linear_svm_results.pkl')

    features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                             dtype=[('names', 'S12'), ('linear SVM coefficients', '>i4')])

    features_info['names'] = variable_names

    coefficients = np.zeros(X_train.shape[1])
    coefficients[best_estimator.named_steps['rfe_lr'].get_support()] = np.absolute(best_estimator.named_steps['linear_svm'].coef_)

    features_info['linear SVM coefficients'] = coefficients

    with open(os.getcwd() + '/' + dataset_type + '/coefficients_features_info.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['names', 'linear SVM coefficients'])
        w.writerows(features_info)
