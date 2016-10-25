from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
import csv
import os


def down_sample(X, y, seed):
    rus = RandomUnderSampler(random_state=seed)
    X_res, y_res = rus.fit_sample(X, y)

    return X_res, y_res


def up_sample(X, y, seed):
    ros = RandomOverSampler(random_state=seed)
    X_res, y_res = ros.fit_sample(X, y)

    return X_res, y_res


def smote_sample(X, y, seed):
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_sample(X, y)

    return X_res, y_res


if __name__ == '__main__':
    num_experiments = 10

    seeds = [123, 456, 789]

    print("Loading variable names...")
    print()
    with open('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/raw/raw_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            variable_names = np.array(list(row))
            break

    for i in range(0, num_experiments):
        print("Loading experiment " + str(i) + " data...")
        print()
        raw_train_data = np.genfromtxt(
            '/home/mgvaldes/devel/MIRI/master-thesis/health-forecast/fs/raw/experiment_' + str(i) + '_train.csv',
            delimiter=',')
        raw_train_data = raw_train_data[1:, :]

        X_train = raw_train_data[:, 1:]
        y_train = raw_train_data[:, 0]

        print("Down sampling data...")
        print()
        # Perform down sample to all experiment data
        sampled_X_train, sampled_y_train = down_sample(X_train, y_train, seeds[0])

        sampled_train_dataset = np.zeros((sampled_X_train.shape[0], sampled_X_train.shape[1] + 1))
        sampled_train_dataset[:, 0] = sampled_y_train
        sampled_train_dataset[:, 1:] = sampled_X_train

        sampled_train_dataset_name = os.getcwd() + '/fs/down_sample/experiment_' + str(i) + '_down_sample_train.csv'

        with open(sampled_train_dataset_name, 'w') as f:
            w = csv.writer(f)
            w.writerow(variable_names)
            w.writerows(sampled_train_dataset)

        print("Up sampling data...")
        print()
        # Perform up sample to all experiment data
        sampled_X_train, sampled_y_train = up_sample(X_train, y_train, seeds[1])

        sampled_train_dataset = np.zeros((sampled_X_train.shape[0], sampled_X_train.shape[1] + 1))
        sampled_train_dataset[:, 0] = sampled_y_train
        sampled_train_dataset[:, 1:] = sampled_X_train

        sampled_train_dataset_name = os.getcwd() + '/fs/up_sample/experiment_' + str(i) + '_up_sample_train.csv'

        with open(sampled_train_dataset_name, 'w') as f:
            w = csv.writer(f)
            w.writerow(variable_names)
            w.writerows(sampled_train_dataset)

        print("SMOTE sampling data...")
        print()
        # Perform SMOTE sample to all experiment data
        sampled_X_train, sampled_y_train = smote_sample(X_train, y_train, seeds[2])

        sampled_train_dataset = np.zeros((sampled_X_train.shape[0], sampled_X_train.shape[1] + 1))
        sampled_train_dataset[:, 0] = sampled_y_train
        sampled_train_dataset[:, 1:] = sampled_X_train

        sampled_train_dataset_name = os.getcwd() + '/fs/smote_sample/experiment_' + str(i) + '_smote_sample_train.csv'

        with open(sampled_train_dataset_name, 'w') as f:
            w = csv.writer(f)
            w.writerow(variable_names)
            w.writerows(sampled_train_dataset)