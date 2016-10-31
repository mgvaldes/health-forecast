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


def performance_metrics(experiment_results, best_estimator, fs_step_name, classifier_step_name, X_train, y_train, X_test,
                        y_test, dataset_type, variable_names, sampling):
    experiment_results['best_estimator'] = best_estimator

    cv_score = np.mean(cross_val_score(best_estimator, X_train, y_train, n_jobs=12, cv=StratifiedKFold(n_splits=5, random_state=789012), scoring='f1_weighted'))

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

    result_files_path = os.getcwd() + '/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + sampling + '/' + dataset_type

    plot_confusion_matrix(linear_svm_confusion_matrix, classes=["Positive", "Negative"],
                          filename=result_files_path + '/confusion_matrix.png')

    linear_svm_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred)
    # experiment_results['class_precision_recall_f1'] = linear_svm_precision_recall_fscore_support

    pos_precision = linear_svm_precision_recall_fscore_support[0][0]
    experiment_results['pos_precision'] = pos_precision
    neg_precision = linear_svm_precision_recall_fscore_support[0][1]
    experiment_results['neg_precision'] = neg_precision
    pos_recall = linear_svm_precision_recall_fscore_support[1][0]
    experiment_results['pos_recall'] = pos_recall
    neg_recall = linear_svm_precision_recall_fscore_support[1][1]
    experiment_results['neg_recall'] = neg_recall
    pos_f1 = linear_svm_precision_recall_fscore_support[2][0]
    experiment_results['pos_f1'] = pos_f1
    neg_f1 = linear_svm_precision_recall_fscore_support[2][1]
    experiment_results['neg_f1'] = neg_f1

    linear_svm_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # experiment_results['global_precision_recall_f1'] = linear_svm_precision_recall_fscore_support
    precision = linear_svm_precision_recall_fscore_support[0]
    experiment_results['precision'] = precision
    recall = linear_svm_precision_recall_fscore_support[1]
    experiment_results['recall'] = recall
    f1 = linear_svm_precision_recall_fscore_support[2]
    experiment_results['f1'] = f1

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

    # linear_svm_f1 = f1_score(y_test, y_pred, average='weighted')
    # experiment_results['weighted_F1'] = linear_svm_f1
    #
    # print("WEIGHTED F1:")
    # print()
    # print(linear_svm_f1)
    # print()

    print("Precision:")
    print()
    print(precision)
    print()

    print("Recall:")
    print()
    print(recall)
    print()

    print("F1:")
    print()
    print(f1)
    print()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
    pos_auc = auc(fpr, tpr)
    experiment_results['fpr'] = fpr
    experiment_results['tpr'] = tpr
    experiment_results['pos_auc'] = pos_auc

    plot_roc(fpr, tpr, pos_auc, "Positive ROC", filename=result_files_path + '/pos_roc.png')

    print("Positive AUC:")
    print()
    print(pos_auc)
    print()

    fnr, tnr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)
    neg_auc = auc(fnr, tnr)
    experiment_results['fnr'] = fnr
    experiment_results['tnr'] = tnr
    experiment_results['neg_auc'] = neg_auc

    plot_roc(fnr, tnr, neg_auc, "Negative ROC", filename=result_files_path + '/neg_roc.png')

    print("Negative AUC:")
    print()
    print(neg_auc)
    print()

    if classifier_step_name == "linear_svm":
        save_object(experiment_results, result_files_path + '/linear_svm_results.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('linear SVM coefficients', 'f4')])

        features_info['names'] = variable_names

        coefficients = np.zeros(X_train.shape[1])
        coefficients[best_estimator.named_steps[fs_step_name].get_support()] = np.absolute(best_estimator.named_steps[classifier_step_name].coef_)

        features_info['linear SVM coefficients'] = coefficients

        with open(result_files_path + '/coefficients_features_info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['names', 'linear SVM coefficients'])
            w.writerows(features_info)

    elif classifier_step_name == "rf":
        save_object(experiment_results, result_files_path + '/rf_results.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('RF importances', 'f4')])

        features_info['names'] = variable_names

        importances = np.zeros(X_train.shape[1])
        importances[best_estimator.named_steps[fs_step_name].get_support()] = best_estimator.named_steps[classifier_step_name].feature_importances_

        features_info['RF importances'] = importances

        with open(result_files_path + '/importances_features_info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['names', 'RF importances'])
            w.writerows(features_info)
