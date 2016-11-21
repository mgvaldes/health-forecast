import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import csv
import os
from utils_functions import iter_loadtxt
from sklearn.model_selection import StratifiedShuffleSplit

##############################################################################
# Generate complete dataset with all features, genomic and epidemiologic
# PC's in epidemiological data, not taken into account.
# genomic_data = np.genfromtxt(os.getcwd() + '/datasets/chr12_imputed_maf001.raw', delimiter=' ')
# print(genomic_data.shape)
#
# genomic_data = genomic_data[1:, 6:]
# print(genomic_data.shape)
#
# genomic_data_n_rows = genomic_data.shape[0]
# print('nrows: ' + str(genomic_data_n_rows))
#
# genomic_data_n_cols = genomic_data.shape[1]
# print('ncols: ' + str(genomic_data_n_cols))
#
# with open(os.getcwd() + '/datasets/chr12_imputed_maf001.raw', 'r') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ')
#     for row in reader:
#         genomic_data_variable_names = np.array(list(row))
#         break
#
# genomic_data_variable_names = genomic_data_variable_names[6:]
#
# epidem_data = np.genfromtxt(os.getcwd() + '/datasets/IMPPC_lung_progres.txt', delimiter=' ')
# print(epidem_data.shape)
#
# epidem_data = epidem_data[1:, 10:]
# print(epidem_data.shape)
#
# epidem_data_n_rows = epidem_data.shape[0]
# print('nrows: ' + str(epidem_data_n_rows))
#
# epidem_data_n_cols = epidem_data.shape[1]
# print('ncols: ' + str(epidem_data_n_cols))
#
# with open('IMPPC_lung_progres.txt', 'r') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ')
#     for row in reader:
#         epidem_data_variable_names = np.array(list(row))
#         break
#
# epidem_data_variable_names = epidem_data_variable_names[10:]
#
# ganomic_dataset = np.zeros((genomic_data_n_rows, (genomic_data_n_cols + 1)))
# ganomic_dataset[:, 0] = epidem_data[:, -1] - 1
# ganomic_dataset[:, 1:] = genomic_data
#
# genomic_variable_names = np.append(epidem_data_variable_names[-1], genomic_data_variable_names)
#
# with open(os.getcwd() + '/datasets/genomic/genomic_dataset_with_pheno.csv', 'w') as f:
#     w = csv.writer(f)
#     w.writerow(genomic_variable_names)
#     w.writerows(ganomic_dataset)
#
# complete_dataset = np.zeros((genomic_data_n_rows, (genomic_data_n_cols + epidem_data_n_cols)))
# complete_dataset[:, 0] = epidem_data[:, -1] - 1
# complete_dataset[:, 1:epidem_data_n_cols] = epidem_data[:, :-1]
# complete_dataset[:, epidem_data_n_cols:] = genomic_data
#
# variable_names = np.append(np.append(epidem_data_variable_names[-1], epidem_data_variable_names[:-1]), genomic_data_variable_names)
#
# with open(os.getcwd() + '/datasets/genomic_epidemiological/complete_dataset_with_pheno.csv', 'w') as f:
#     w = csv.writer(f)
#     w.writerow(variable_names)
#     w.writerows(complete_dataset)

##############################################################################
# # Read complete dataset and remove variables with low variance
# complete_data = np.genfromtxt(os.getcwd() + '/datasets/genomic/genomic_dataset_with_pheno.csv', delimiter=',')
# print(complete_data.shape)
#
# complete_data = complete_data[1:, :]
# print(complete_data.shape)
#
# complete_data_n_rows = complete_data.shape[0]
# print('nrows: ' + str(complete_data_n_rows))
#
# complete_data_n_cols = complete_data.shape[1]
# print('ncols: ' + str(complete_data_n_cols))
#
# with open(os.getcwd() + '/datasets/genomic/genomic_dataset_with_pheno.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row in reader:
#         variable_names = np.array(list(row))
#         break
#
# variable_names = variable_names[1:]
#
# X = np.zeros((complete_data_n_rows, (complete_data_n_cols - 1)))
# X = complete_data[:, 1:]
#
# y = complete_data[:, 0]
#
# variance_selector = VarianceThreshold(threshold=0.15)
#
# variance_selector_result = variance_selector.fit(X, y)
#
# new_X = variance_selector_result.transform(X)
#
# new_variable_names = variable_names[variance_selector_result.get_support()]
#
# new_complete_dataset = np.zeros((complete_data_n_rows, (new_X.shape[1] + 1)))
# new_complete_dataset[:, 0] = y
# new_complete_dataset[:, 1:] = new_X
#
# new_variable_names = np.append(["progres"], new_variable_names)
#
# with open(os.getcwd() + '/datasets/genomic/variance_threshold/reduced_genomic_dataset_with_pheno_variance_threshold_0_15.csv', 'w') as f:
#     w = csv.writer(f)
#     w.writerow(new_variable_names)
#     w.writerows(new_complete_dataset)

##############################################################################
# Read reduced dataset from low variance process and create training/test sets
# reduced_data = np.genfromtxt(os.getcwd() + '/datasets/genomic/genomic_dataset_with_pheno.csv', delimiter=',')

dataset_type = "CD2W_vs_CD2F"

print("Loading data...")
reduced_data = iter_loadtxt(os.getcwd() + '/datasets/diabetes/genomic_epidemiological/' + dataset_type + '/' + dataset_type + '.csv')
print(reduced_data.shape)

# reduced_data  = reduced_data[1:, :]
# print(reduced_data.shape)

reduced_data_n_rows = reduced_data.shape[0]
print('Rows: ' + str(reduced_data_n_rows))

reduced_data_n_cols = reduced_data.shape[1]
print('Columns: ' + str(reduced_data_n_cols))

with open(os.getcwd() + '/datasets/diabetes/genomic_epidemiological/' + dataset_type + '/' + dataset_type + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        variable_names = np.array(list(row))
        break

# print("Dividing X and y...")
# X = np.zeros((reduced_data_n_rows, (reduced_data_n_cols - 1)))
# X = reduced_data[:, 1:]
# y = reduced_data[:, 0]

print("Spliting into train and test...")
# X_train, y_train, X_test, y_test, indexes_train, indexes_test = train_test_split(X, y, range(len(y)), test_size=0.2, random_state=0, stratify=y)
# _, _, _, _, indexes_train, indexes_test = train_test_split(reduced_data[:, 1:], reduced_data[:, 0], range(reduced_data_n_rows), test_size=0.2, random_state=0, stratify=reduced_data[:, 0])
train_indexes, test_indexes = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=6547891).split(reduced_data[:, 1:], reduced_data[:, 0]))[0]

# print("Creating training dataset...")
# train_dataset = np.zeros((len(indexes_train), reduced_data_n_cols))
# # train_dataset[:, 0] = y_train
# # train_dataset[:, 1:] = X_train
# train_dataset[:, 0] = reduced_data[indexes_train, 0]
# train_dataset[:, 1:] = reduced_data[indexes_train, 1:]
#
# print("Creating test dataset...")
# test_dataset = np.zeros((len(indexes_test), reduced_data_n_cols))
# # test_dataset[:, 0] = y_test
# # test_dataset[:, 1:] = X_test
# test_dataset[:, 0] = reduced_data[indexes_test, 0]
# test_dataset[:, 1:] = reduced_data[indexes_test, 1:]

print("Saving train dataset...")
with open(os.getcwd() + '/datasets/diabetes/genomic_epidemiological/' + dataset_type + '/raw_train.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(variable_names)
    w.writerows(reduced_data[train_indexes, :])

print("Saving test dataset...")
with open(os.getcwd() + '/datasets/diabetes/genomic_epidemiological/' + dataset_type + '/raw_test.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(variable_names)
    w.writerows(reduced_data[test_indexes, :])


# num_experiments = 10
# seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#
# experiments = dict()
#
# for i in range(0, num_experiments):
#     seed = seeds[i]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
#
#     train_dataset = np.zeros((X_train.shape[0], reduced_data_n_cols))
#     train_dataset[:, 0] = y_train
#     train_dataset[:, 1:] = X_train
#
#     test_dataset = np.zeros((X_test.shape[0], reduced_data_n_cols))
#     test_dataset[:, 0] = y_test
#     test_dataset[:, 1:] = X_test
#
#     train_dataset_name = os.getcwd() + '/datasets/genomic/raw/experiment_' + str(i) + '_train.csv'
#
#     with open(train_dataset_name, 'w') as f:
#         w = csv.writer(f)
#         w.writerow(variable_names)
#         w.writerows(train_dataset)
#
#     test_dataset_name = os.getcwd() + '/datasets/genomic/raw/experiment_' + str(i) + '_test.csv'
#
#     with open(test_dataset_name, 'w') as f:
#         w = csv.writer(f)
#         w.writerow(variable_names)
#         w.writerows(test_dataset)




