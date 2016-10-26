from utils_functions import load_object
import numpy as np
import csv
import os


def print_results(filename):
    results = load_object(filename)

    print('Positive precision: %s' % (results['pos_precision']))
    print()

    print('Positive recall: %s' % (results['pos_recall']))
    print()

    print('Positive F1: %s' % (results['pos_f1']))
    print()

    print('Negative precision: %s' % (results['neg_precision']))
    print()

    print('Negative recall: %s' % (results['neg_recall']))
    print()

    print('Negative F1: %s' % (results['neg_f1']))
    print()

    print('Weighted F1: %s' % (results['weighted_F1']))
    print()

    print('AUC: %s' % (results['neg_auc']))
    print()

    print('Accuracy: %s' % (results['accuracy']))
    print()

    print('Train score: %s' % (results['train_score']))
    print()

    print('CV score: %s' % (results['cv_score']))
    print()


if __name__ == '__main__':
    main_path = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/datasets/'

    dataset_type = 'genomic'

    filename = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/wrapper/rfe_lr/classifiers/linear_svm/' + dataset_type + '/linear_svm_results.pkl'

    results = load_object(filename)

    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/raw/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                             dtype=[('names', 'S120'), ('linear SVM coefficients', 'f4')])

    features_info['names'] = variable_names

    coefficients = np.zeros(len(variable_names))
    coefficients[results['best_estimator'].named_steps['rfe_lr'].get_support()] = np.absolute(
        results['best_estimator'].named_steps['linear_svm'].coef_)

    features_info['linear SVM coefficients'] = coefficients

    with open('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/wrapper/rfe_lr/classifiers/linear_svm/' + dataset_type + '/coefficients_features_info.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['names', 'linear SVM coefficients'])
        w.writerows(features_info)

    dataset_type = 'genomic_epidemiological'

    filename = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/wrapper/rfe_lr/classifiers/linear_svm/' + dataset_type + '/linear_svm_results.pkl'

    results = load_object(filename)

    print("Loading variable names...")
    print()
    with open(main_path + dataset_type + '/raw/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    variable_names = variable_names[1:]

    features_info = np.array(list(zip(np.repeat('', len(variable_names)), np.repeat(0, len(variable_names)))),
                             dtype=[('names', 'S120'), ('linear SVM coefficients', 'f4')])

    features_info['names'] = variable_names

    coefficients = np.zeros(len(variable_names))
    coefficients[results['best_estimator'].named_steps['rfe_lr'].get_support()] = np.absolute(
        results['best_estimator'].named_steps['linear_svm'].coef_)

    features_info['linear SVM coefficients'] = coefficients

    with open('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/wrapper/rfe_lr/classifiers/linear_svm/' + dataset_type + '/coefficients_features_info.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['names', 'linear SVM coefficients'])
        w.writerows(features_info)