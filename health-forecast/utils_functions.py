from sklearn.externals import joblib
import pickle
import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, f1_score
from plot_functions import plot_confusion_matrix, plot_roc, plot_metrics_vs_data
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
                        y_test, dataset_type, variable_names, sampling, sampling_timing):
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

    result_files_path = os.getcwd() + '/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + sampling_timing + '/' + '/' + sampling + '/' + dataset_type

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
        save_object(experiment_results, result_files_path + '/' + classifier_step_name + '_results.pkl')

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
        save_object(experiment_results, result_files_path + '/' + classifier_step_name + '_results.pkl')

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
    elif classifier_step_name == "knn":
        save_object(experiment_results, result_files_path + '/' + classifier_step_name + '_results.pkl')


def manual_performance_metrics(experiment_results, feature_selector, best_estimator, fs_step_name, classifier_step_name, X_train, y_train,
                         X_test, y_test, dataset_type, variable_names, sampling, sampling_timing):
    experiment_results['feature_selector'] = feature_selector
    experiment_results['best_estimator'] = best_estimator

    cv_score = np.mean(
        cross_val_score(best_estimator, X_train, y_train, n_jobs=12, cv=StratifiedKFold(n_splits=5, random_state=789012),
                        scoring='f1_weighted'))

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

    result_files_path = os.getcwd() + '/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + sampling_timing + '/' + '/' + sampling + '/' + dataset_type

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
        save_object(experiment_results, result_files_path + '/' + classifier_step_name + '_results.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('linear SVM coefficients', 'f4')])

        features_info['names'] = variable_names

        coefficients = np.zeros(len(variable_names))
        coefficients[feature_selector.top_features[:2000]] = np.absolute(best_estimator.coef_)

        features_info['linear SVM coefficients'] = coefficients

        with open(result_files_path + '/coefficients_features_info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['names', 'linear SVM coefficients'])
            w.writerows(features_info)
    elif classifier_step_name == "rf":
        save_object(experiment_results, result_files_path + '/' + classifier_step_name + '_results.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('RF importances', 'f4')])

        features_info['names'] = variable_names

        importances = np.zeros(len(variable_names))
        importances[feature_selector.top_features[:2000]] = best_estimator.feature_importances_

        features_info['RF importances'] = importances

        with open(result_files_path + '/importances_features_info.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['names', 'RF importances'])
            w.writerows(features_info)
    elif classifier_step_name == "knn":
        save_object(experiment_results, result_files_path + '/' + classifier_step_name + '_results.pkl')


def feature_metrics(main_path, dataset_type, sampling, sampling_timing, fs_step_name, classifier_step_name):
    print("Loading best estimator...")
    print()

    result_files_path = os.getcwd() + '/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + sampling_timing + '/' + sampling + '/' + dataset_type

    experiment_results = load_object(result_files_path + '/' + classifier_step_name + '_results.pkl')

    best_estimator = experiment_results['best_estimator']

    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/' + sampling + '/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    num_experiments = 10
    feature_ranking = np.zeros((len(variable_names), num_experiments))
    coefficients = np.zeros((len(variable_names), num_experiments))

    for i in range(0, num_experiments):
        print("##### Experiment " + str(i) + " Info #####")
        print("Dataset type: ", dataset_type)
        print("Sampling: ", sampling)
        print("FS: ", fs_step_name)
        print("Classifier: ", classifier_step_name)
        print()

        raw_train_data = np.genfromtxt(main_path + dataset_type + '/' + sampling + '/experiment_' + str(i) + '_train.csv', delimiter=',')
        raw_train_data = raw_train_data[1:, :]

        X_train = raw_train_data[:, 1:]
        y_train = raw_train_data[:, 0]

        print("Re-fitting best estimator...")
        print()
        best_estimator.fit(X_train, y_train)

        selected_features = np.zeros(X_train.shape[1])
        selected_features[best_estimator.named_steps[fs_step_name].get_support()] = 1

        feature_ranking[:, i] = selected_features

        if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
            selected_coefficients = np.zeros(X_train.shape[1])

            if classifier_step_name == "linear_svm":
                selected_coefficients[best_estimator.named_steps[fs_step_name].get_support()] = \
                    best_estimator.named_steps[classifier_step_name].coef_
            elif classifier_step_name == "rf":
                selected_coefficients[best_estimator.named_steps[fs_step_name].get_support()] = \
                    best_estimator.named_steps[classifier_step_name].feature_importances_
            elif classifier_step_name == "knn":
                selected_coefficients[best_estimator.named_steps[fs_step_name].get_support()] = \
                    best_estimator.named_steps[fs_step_name].estimator_.coef_

            coefficients[:, i] = selected_coefficients

    print("Calculating final feature ranking")
    print()

    final_ranking = np.sum(feature_ranking, axis=1)

    save_object(feature_ranking, result_files_path + '/feature_stability.pkl')

    if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
        if classifier_step_name == "linear_svm":
            mean_name = "coefficients_mean"
            abs_mean_name = "abs_coefficients_mean"
            scaled_name = "scaled_coefficients"

            save_object(coefficients, result_files_path + '/feature_coefficients.pkl')
        elif classifier_step_name == "rf":
            mean_name = "importances_mean"
            abs_mean_name = "abs_importances_mean"
            scaled_name = "scaled_importances"

            save_object(coefficients, result_files_path + '/feature_importances.pkl')
        elif classifier_step_name == "knn":
            mean_name = "coefficients_mean"
            abs_mean_name = "abs_coefficients_mean"
            scaled_name = "scaled_coefficients"

            save_object(coefficients, result_files_path + '/feature_coefficients.pkl')

        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)),
                                          np.repeat(0, len(variable_names)), np.repeat(0, len(variable_names)),
                                          np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('stability', '>i4'), (mean_name, '>f8'),
                                        (abs_mean_name, '>f8'), (scaled_name, '>f8')])
    else:
        features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                                 dtype=[('names', 'S120'), ('stability', '>i4')])

    features_info['names'] = variable_names
    features_info['stability'] = final_ranking

    if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
        features_info[mean_name] = np.mean(coefficients, axis=1)
        features_info[abs_mean_name] = np.mean(np.abs(coefficients), axis=1)
        features_info[scaled_name] = np.mean((coefficients - np.min(coefficients)) / (np.max(coefficients) - np.min(coefficients)), axis=1)

    with open(result_files_path + '/general_features_info.csv', 'w') as f:
        w = csv.writer(f)

        header = list(['names', 'stability'])

        if classifier_step_name != "knn" or (classifier_step_name == "knn" and fs_step_name == "rlr_l1"):
            header.append(list([mean_name, abs_mean_name, scaled_name]))

        w.writerow(header)
        w.writerows(features_info)


def performance_vs_data(main_path, dataset_type, best_estimator):
    print("Loading data...")
    print()
    raw_train_data = np.genfromtxt(main_path + dataset_type + '/raw/raw_train.csv', delimiter=',')
    raw_train_data = raw_train_data[1:, :]

    X_train = raw_train_data[:, 1:]
    y_train = raw_train_data[:, 0]

    raw_test_data = np.genfromtxt(main_path + dataset_type + '/raw/raw_test.csv', delimiter=',')
    raw_test_data = raw_test_data[1:, :]

    X_test = raw_test_data[:, 1:]
    y_test = raw_test_data[:, 0]

    cv_scores = []
    test_scores = []

    for percentage in np.arange(0.3, 1.1, 0.1):
        print("Using", str(percentage * 100) + "%", "of data...")

        rows = int(np.round(X_train.shape[0] * percentage))
        print("Corresponding to " + str(rows) + " rows of training data.")

        partial_X_train = X_train[:rows, :]
        print("partial_X_train shape: " + str(partial_X_train.shape))
        partial_y_train = y_train[:rows]
        print("partial_y_train shape: " + str(partial_y_train.shape))

        best_estimator.fit(partial_X_train, partial_y_train)

        cv_score = np.mean(cross_val_score(best_estimator, partial_X_train, partial_y_train, n_jobs=12,
                                           cv=StratifiedKFold(n_splits=5, random_state=789012), scoring='f1_weighted'))
        print("CV Score: " + str(cv_score))
        cv_scores.append(cv_score)

        y_pred = best_estimator.predict(X_test)

        test_score = f1_score(y_test, y_pred, average='weighted')
        print("Test Score: " + str(test_score))
        print()
        test_scores.append(test_score)

    plot_metrics_vs_data(cv_scores, test_scores)


if __name__ == '__main__':
    disease = "lung_cancer"
    chromosome = "chr12"

    main_path = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast-project/health-forecast/datasets/' + disease + '/' + chromosome + '/'
    dataset_type = "genomic"
    sampling_timing = "sampling_before_fs"
    sampling_type = "up_sample"
    fs_type = ("embedded", "rlr_l1")
    classifier_type = "knn"

    # best_estimator_dir = os.getcwd() + '/fs/' + fs_type[0] + '/' + fs_type[1] + '/classifiers/' + classifier_type + '/' + \
    #                      sampling_timing + '/' + sampling_type + '/' + dataset_type + '/' + classifier_type + '_results.pkl'
    #
    # results = load_object(best_estimator_dir)
    #
    # best_estimator = results['best_estimator']
    #
    # performance_vs_data(main_path, dataset_type, best_estimator)

    best_estimator_dir = os.getcwd() + '/fs/' + fs_type[0] + '/' + fs_type[1] + '/classifiers/' + classifier_type + '/' + \
                         sampling_timing + '/' + sampling_type + '/' + dataset_type + '/feature_coefficients.pkl'

    results = load_object(best_estimator_dir)