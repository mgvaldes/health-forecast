library(data.table)

raw.file <- fread("chr12_imputed_maf001.raw", data.table = FALSE, header = TRUE)
raw.file <- raw.file[, 7:ncol(raw.file)]

pheno.data <- fread("IMPPC_lung_progres.txt", header = TRUE, data.table = FALSE)
pheno.data <- pheno.data[, 11:17]
pheno.data <- sapply(pheno.data, as.numeric)

library(mice)

imp <- mice(pheno.data, m = 1)
pheno.data <- complete(imp)

################ complete raw data
raw.data.with.pheno.data <- cbind(pheno.data[, 1:6], raw.file)
raw.data.with.pheno.data <- cbind(pheno.data[, 7], raw.data.with.pheno.data)
colnames(raw.data.with.pheno.data)[1] <- "progres"

write.table(raw.data.with.pheno.data, file = "complete_dataset_with_pheno.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)