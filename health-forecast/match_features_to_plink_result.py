import numpy as np
from numpy.lib.recfunctions import append_fields
import os
import csv

fs_step_name = "rlr_l1"
classifier_step_name = "rf"
sampling_timing = "sampling_after_fs"
sampling = "down_sample"
dataset_type = "genomic_epidemiological"
disease = "lung_cancer"
chromosome = "chr12"

result_files_path = os.getcwd() + '/fs/' + disease + '/' + chromosome + '/embedded/' + fs_step_name + '/classifiers/' + classifier_step_name + '/' + sampling_timing + '/' + sampling + '/' + dataset_type

general_features_info = np.genfromtxt(result_files_path + '/general_features_info.csv', delimiter=';',
                                      dtype=[('names', 'S120'), ('stability', '>i4'), ("importances_mean", 'float64'),
                                        ("abs_importances_mean", 'float64'), ("scaled_importances", 'float64')], skip_header=1)
# general_features_info = general_features_info[1:, :]
# general_features_info["names"] = np.core.defchararray.replace(general_features_info["names"], "b'", "")
# general_features_info["names"] = np.core.defchararray.replace(general_features_info["names"], "'", "")

plink_features_info = np.genfromtxt(os.getcwd() + '/' + dataset_type + '_plink_threshold_0_01_features.csv', delimiter=',',
                                    dtype=[('names', 'S120'), ("P", 'float64')], skip_header=1)
# plink_features_info = plink_features_info[1:, :]

# p1 = np.array([('A', 0), ('B', 1), ('C', 2), ('D', 3)], dtype=[("id", "S1"), ("p-value", "i4")])
# p2 = np.array([('F', 94623), ('D', 93456), ('E', 7846), ('A', 1086)], dtype=[("id", "S1"), ("p-value", "i4")])
#
# partial = [(list(p2["id"]).index(x), p2["p-value"][list(p2["id"]).index(x)]) if x in p2["id"] else (-1, -1) for x in p1["id"]]
# pos, p_value = zip(*partial)

partial = [(list(plink_features_info["names"]).index(x), plink_features_info["P"][list(plink_features_info["names"]).index(x)]) if x in plink_features_info["names"] else (-1, -1) for x in general_features_info["names"]]
pos, p_value = zip(*partial)

result = append_fields(general_features_info, 'plink_position', np.array(pos, dtype="int"))
result = append_fields(result, 'plink_p_value', np.array(p_value, dtype="float64"))

with open(result_files_path + '/best_model_general_features_info.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(list(result.dtype.names))
    w.writerows(result)