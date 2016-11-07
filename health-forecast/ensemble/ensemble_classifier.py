from sklearn.ensemble import VotingClassifier
import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
from plot_functions import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from utils_functions import save_object


def enssemble_classifier(best_estimators, main_path, dataset_type):
    print("##### Experiment Info #####")
    print("Dataset type: ", dataset_type)
    print()

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

    ensemble = VotingClassifier(estimators=best_estimators, voting='hard')

    experiment_results['estimator'] = ensemble

    cv_score = np.mean(cross_val_score(ensemble, X_train, y_train, n_jobs=12,
                                       cv=StratifiedKFold(n_splits=5, random_state=789012), scoring='f1_weighted'))

    experiment_results['cv_score'] = cv_score

    print("CV score:")
    print()
    print(cv_score)
    print()

    y_train_pred = ensemble.predict(X_train)
    train_score = f1_score(y_train, y_train_pred, average='weighted')
    experiment_results['train_score'] = train_score

    print("Train score:")
    print()
    print(train_score)
    print()

    y_pred = ensemble.predict(X_test)

    print("Predicting y_test with reduced X_test")
    print()
    print(y_pred)
    print()

    classifier_accuracy = accuracy_score(y_test, y_pred)
    experiment_results['accuracy'] = classifier_accuracy

    print("Accuracy:")
    print()
    print(classifier_accuracy)
    print()

    classifier_confusion_matrix = confusion_matrix(y_test, y_pred)
    experiment_results['confusion_matrix'] = classifier_confusion_matrix

    print("Confusion matrix:")
    print()
    print(classifier_confusion_matrix)
    print()

    result_files_path = os.getcwd() + '/' + dataset_type

    plot_confusion_matrix(classifier_confusion_matrix, classes=["Positive", "Negative"],
                          filename=result_files_path + '/confusion_matrix.png')

    classifier_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred)

    pos_precision = classifier_precision_recall_fscore_support[0][0]
    experiment_results['pos_precision'] = pos_precision
    neg_precision = classifier_precision_recall_fscore_support[0][1]
    experiment_results['neg_precision'] = neg_precision
    pos_recall = classifier_precision_recall_fscore_support[1][0]
    experiment_results['pos_recall'] = pos_recall
    neg_recall = classifier_precision_recall_fscore_support[1][1]
    experiment_results['neg_recall'] = neg_recall
    pos_f1 = classifier_precision_recall_fscore_support[2][0]
    experiment_results['pos_f1'] = pos_f1
    neg_f1 = classifier_precision_recall_fscore_support[2][1]
    experiment_results['neg_f1'] = neg_f1

    classifier_precision_recall_fscore_support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    precision = classifier_precision_recall_fscore_support[0]
    experiment_results['precision'] = precision
    recall = classifier_precision_recall_fscore_support[1]
    experiment_results['recall'] = recall
    f1 = classifier_precision_recall_fscore_support[2]
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

    save_object(experiment_results, result_files_path + '/' + 'ensemble_results.pkl')
