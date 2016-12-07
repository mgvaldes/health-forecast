import csv
import os
import numpy as np
from utils_functions import load_object


def read_results(filename):
    results = load_object(filename)

    results_tup = ()

    results_tup = results_tup + (results['cv_score'],)
    results_tup = results_tup + (results['train_score'],)
    results_tup = results_tup + (results['f1'], )
    results_tup = results_tup + (results['neg_auc'],)
    results_tup = results_tup + (results['precision'],)
    results_tup = results_tup + (results['recall'],)
    results_tup = results_tup + (results['accuracy'],)
    results_tup = results_tup + (results['neg_precision'],)
    results_tup = results_tup + (results['neg_recall'],)
    results_tup = results_tup + (results['neg_f1'],)
    results_tup = results_tup + (results['pos_precision'],)
    results_tup = results_tup + (results['pos_recall'],)
    results_tup = results_tup + (results['pos_f1'],)

    return results_tup


if __name__ == '__main__':
    # print(read_results('/home/mgvaldes/devel/MIRI/master-thesis/health-forecast-project/health-forecast/fs/embedded/'
    #              'rlr_l1/classifiers/knn/sampling_after_fs_old/up_sample/genomic/knn_results.pkl'))

    sampling_timings = ["sampling_after_fs"]
    sampling_types = ["raw", "down_sample", "up_sample", "smote_sample"]
    dataset_types = ["genomic_epidemiological"]
    fs_types = [("filter", "anova"), ("wrapper", "rfe_lr"), ("embedded", "rlr_l1")]
    classifier_types = ["linear_svm", "rf", "knn"]
    disease = "lung_cancer"
    chromosome = "chr12"

    for dataset_type in dataset_types:
        resume_results = np.zeros(0, dtype=('a50, a50, a50, a50, float64, float64, float64, float64, float64, float64, float64, '
                                            'float64, float64, float64, float64, float64, float64'))

        for fs_type in fs_types:
            fs_dir = os.getcwd() + '/fs/' + disease + '/' + chromosome + '/' + fs_type[0] + '/' + fs_type[1] + '/classifiers/'

            for classifier_type in classifier_types:
                classifier_dir = fs_dir + classifier_type + '/'

                for sampling_timing in sampling_timings:
                    sampling_timing_dir = classifier_dir + sampling_timing + '/'

                    for sampling_type in sampling_types:
                        sampling_dir = sampling_timing_dir + sampling_type + '/'

                        dataset_dir = sampling_dir + dataset_type + '/' + classifier_type + '_results.pkl'

                        resume_results = np.append(resume_results,
                                                   np.array([(sampling_type, fs_type[0] + ': ' + fs_type[1],
                                                              classifier_type, dataset_type) + read_results(
                                                       dataset_dir)],
                                                            dtype=resume_results.dtype))

        with open(os.getcwd() + '/fs/' + disease + '/' + chromosome + '/' + dataset_type + '_resumed_results.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['sampling', 'fs', 'classifier', 'data', 'cv f1', 'train f1', 'test f1', 'test auc', 'test precision', 'test recall', 'test accuracy', 'test precision (1)', 'test recall (1)', 'test f1 (1)', 'test precision (0)', 'test recall (0)', 'test f1 (0)'])
            w.writerows(resume_results)

    # classifier_dir = os.getcwd() + '/' + fs_step_name + '/classifiers/' + classifier_step_name
    #
    # if not os.path.exists(classifier_dir):
    #     os.makedirs(classifier_dir)
    #
    # for sampling in sampling_types:
    #     sampling_dir = classifier_dir + '/' + sampling
    #
    #     if not os.path.exists(sampling_dir):
    #         os.makedirs(sampling_dir)
    #
    #     for dataset_type in dataset_types:
    #         dataset_dir = sampling_dir + '/' + dataset_type
    #
    #         if not os.path.exists(dataset_dir):
    #             os.makedirs(dataset_dir)

