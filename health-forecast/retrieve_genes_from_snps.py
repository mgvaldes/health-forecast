import subprocess
import os
import csv
import numpy as np

# Define command and arguments
command = 'Rscript'
path_to_script = os.getcwd() + '/retrieve_genes_from_snps.R'

sampling_timing = "sampling_after_fs"
sampling_type = "down_sample"
dataset_type = "genomic_epidemiological"
fs_type = ("embedded", "rlr_l1")
classifier_type = "rf"
disease = "lung_cancer"
chromosome = "chr12"

relevant_snps = np.genfromtxt(os.getcwd() + '/fs/' + disease + '/' + chromosome + '/' + fs_type[0] + '/' +
                              fs_type[1] + '/classifiers/' + classifier_type + '/' + sampling_timing + '/' +
                              sampling_type + '/' + dataset_type +
                              '/genomic_epidemiological_best_model_stability_greater_than_35_snps.csv', delimiter=',', dtype='S120')
relevant_snps = relevant_snps[1:]
relevant_snps = relevant_snps[:, 0]

gene_results = np.zeros(0, dtype=('a120, a120'))

for snp in relevant_snps:
    snp = str(snp).replace("b", "").replace("'", "").replace('"', "")
    print('SNP:', snp)

    if snp.startswith('rs'):
        gene_results = np.append(gene_results, np.array([(snp, [])], dtype=gene_results.dtype))

        continue

    args = [snp]

    # Build subprocess command
    cmd = [command, path_to_script] + args

    # check_output will run the command and store to result
    genes_related = subprocess.check_output(cmd, universal_newlines=True)
    genes_related = genes_related.split('\n')
    genes_related = genes_related[-2][4:].replace('"', '')

    print(genes_related)
    print()

    gene_results = np.append(gene_results, np.array([(snp, genes_related)], dtype=gene_results.dtype))

with open(os.getcwd() + '/fs/' + disease + '/' + chromosome + '/' + fs_type[0] + '/' +
                              fs_type[1] + '/classifiers/' + classifier_type + '/' + sampling_timing + '/' +
                              sampling_type + '/' + dataset_type + '/genomic_epidemiological_top_snps_genes.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['snps', 'genes'])
    w.writerows(gene_results)