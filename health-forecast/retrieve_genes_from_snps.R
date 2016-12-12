setwd("~/devel/MIRI/master-thesis/health-forecast-project/health-forecast")

# setRepositories()
# Select option 2
# 
library(seq2pathway)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)

var.name <- args[1]

significant.snps.related.genes <- c()

snp.name <- strsplit(var.name, "_")[[1]][1]

snp.position <- as.numeric(strsplit(var.name, ":")[[1]][2])

snp.data <- data.frame(rs = snp.name, chrom = "chr12", start = snp.position, end = snp.position)

snp.related.genes <- runseq2gene(inputfile = snp.data, genome = "hg38", adjacent = FALSE, SNP = TRUE, search_radius = 1000, PromoterStop = FALSE, NearestTwoDirection = TRUE)

significant.snps.related.genes <- c(significant.snps.related.genes, as.character(snp.related.genes$seq2gene_CodingGeneOnlyResult["gene_name"][, 1]))
significant.snps.related.genes <- unique(significant.snps.related.genes)

print(paste(significant.snps.related.genes, collapse = " "))

# 
# genes <- data.frame(genes = significant.snps.related.genes)
# 
# write.table(genes, file = "genes.csv", row.names = FALSE, col.names = FALSE, sep = ",", quote = FALSE)
# 
# 
# print(significant.snps.related.genes)
# 
# setRepositories()
# Select option 1
# Select option 17