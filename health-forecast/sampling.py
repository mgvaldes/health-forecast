from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
import csv
import os


# def down_sample(X, y, seed, return_target):
#     rus = RandomUnderSampler(random_state=seed)
#     X_res, y_res = rus.fit_sample(X, y)
#
#     if not return_target:
#         return X_res
#
#     return y_res
#
#
# def up_sample(X, y, seed, return_target):
#     ros = RandomOverSampler(random_state=seed)
#     X_res, y_res = ros.fit_sample(X, y)
#
#     if not return_target:
#         return X_res
#
#     return y_res
#
#
# def smote_sample(X, y, seed, return_target):
#     smote = SMOTE(random_state=seed)
#     X_res, y_res = smote.fit_sample(X, y)
#
#     if not return_target:
#         return X_res
#
#     return y_res


def down_sample(X, y, seed):
    rus = RandomUnderSampler(random_state=seed)
    X_res, y_res = rus.fit_sample(X, y)
    # y_res = np.reshape(y_res, (y_res.shape[0], 1))

    return X_res, y_res


def up_sample(X, y, seed):
    ros = RandomOverSampler(random_state=seed)
    X_res, y_res = ros.fit_sample(X, y)

    return X_res, y_res


def smote_sample(X, y, seed):
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_sample(X, y)

    return X_res, y_res


# def sample_single(variable_names, main_path, dataset_type, seeds, filename):
#     print("Loading " + dataset_type + " raw data...")
#     print()
#     raw_train_data = np.genfromtxt(main_path + dataset_type + '/raw/' + filename + '.csv', delimiter=',')
#     raw_train_data = raw_train_data[1:, :]
#
#     X_train = raw_train_data[:, 1:]
#     y_train = raw_train_data[:, 0]
#
#     print("Down sampling data...")
#     print()
#     # Perform down sample to all experiment data
#     sampled_X_train, sampled_y_train = down_sample(X_train, y_train, seeds[0])
#
#     sampled_train_dataset = np.zeros((sampled_X_train.shape[0], sampled_X_train.shape[1] + 1))
#     sampled_train_dataset[:, 0] = sampled_y_train
#     sampled_train_dataset[:, 1:] = sampled_X_train
#
#     sampled_train_dataset_name = main_path + dataset_type + '/down_sample/' + filename + '.csv'
#
#     with open(sampled_train_dataset_name, 'w') as f:
#         w = csv.writer(f)
#         w.writerow(variable_names)
#         w.writerows(sampled_train_dataset)
#
#     print("Up sampling data...")
#     print()
#     # Perform up sample to all experiment data
#     sampled_X_train, sampled_y_train = up_sample(X_train, y_train, seeds[1])
#
#     sampled_train_dataset = np.zeros((sampled_X_train.shape[0], sampled_X_train.shape[1] + 1))
#     sampled_train_dataset[:, 0] = sampled_y_train
#     sampled_train_dataset[:, 1:] = sampled_X_train
#
#     sampled_train_dataset_name = main_path + dataset_type + '/up_sample/' + filename + '.csv'
#
#     with open(sampled_train_dataset_name, 'w') as f:
#         w = csv.writer(f)
#         w.writerow(variable_names)
#         w.writerows(sampled_train_dataset)
#
#     print("SMOTE sampling data...")
#     print()
#     # Perform SMOTE sample to all experiment data
#     sampled_X_train, sampled_y_train = smote_sample(X_train, y_train, seeds[2])
#
#     sampled_train_dataset = np.zeros((sampled_X_train.shape[0], sampled_X_train.shape[1] + 1))
#     sampled_train_dataset[:, 0] = sampled_y_train
#     sampled_train_dataset[:, 1:] = sampled_X_train
#
#     sampled_train_dataset_name = main_path + dataset_type + '/smote_sample/' + filename + '.csv'
#
#     with open(sampled_train_dataset_name, 'w') as f:
#         w = csv.writer(f)
#         w.writerow(variable_names)
#         w.writerows(sampled_train_dataset)
#
#
# def sample_experiments(variable_names, main_path, dataset_type, seeds):
#     num_experiments = 10
#
#     for i in range(0, num_experiments):
#         sample_single(variable_names, main_path, dataset_type, seeds, 'experiment_' + str(i) + '_train')
#
#
# if __name__ == '__main__':
#     seeds = [123, 456, 789]
#     main_path = '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast-project/health-forecast/datasets/'
#
#     dataset_type = 'genomic'
#
#     print("Loading variable %s names..." % (dataset_type))
#     print()
#     with open(main_path + dataset_type + '/raw/raw_train.csv',
#               'r') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         for row in reader:
#             variable_names = np.array(list(row))
#             break
#
#     sample_single(variable_names, main_path, dataset_type, seeds, 'raw_train')
#
#     sample_experiments(variable_names, main_path, dataset_type, seeds)
#
#     dataset_type = 'genomic_epidemiological'
#
#     print("Loading variable %s names..." % (dataset_type))
#     print()
#     with open(main_path + dataset_type + '/raw/raw_train.csv',
#               'r') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         for row in reader:
#             variable_names = np.array(list(row))
#             break
#
#     sample_single(variable_names, main_path, dataset_type, seeds, 'raw_train')
#
#     sample_experiments(variable_names, main_path, dataset_type, seeds)