import subprocess
import os
import csv
import numpy as np

# Define command and arguments
command = 'Rscript'
path_to_script = os.getcwd() + '/retrieve_genes_from_snps.R'

relevant_snps = np.genfromtxt(os.getcwd() + '/best_model_stability_10_features.csv', delimiter=',', dtype='S120')
relevant_snps = relevant_snps[1:]

gene_results = np.zeros(0, dtype=('a120, a120'))

for snp in relevant_snps:
    snp = str(snp).replace("b", "").replace("'", "")
    print('SNP: ', snp)

    args = [snp]

    # Build subprocess command
    cmd = [command, path_to_script] + args

    # check_output will run the command and store to result
    genes_related = subprocess.check_output(cmd, universal_newlines=True)
    genes_related = genes_related.split('\n')
    genes_related = genes_related[-2][4:].replace('"', '')

    print(genes_related)

    gene_results = np.append(gene_results, np.array([(snp, genes_related)], dtype=gene_results.dtype))

with open(os.getcwd() + '/top_snps_genes.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['snps', 'genes'])
    w.writerows(gene_results)