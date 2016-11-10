library(data.table)

setwd("~/Documentos/master-thesis/ehealth/cancer-pulmon-chr12/")
setwd("~/Dropbox/cancer-pulmon-chr12")

cancer.data.ped <- fread("chr12_imputed.recode.geno.txt", header = FALSE, data.table = FALSE)
cancer.data.ped <- cancer.data.ped[-1 ,]

num.snps = nrow(cancer.data.ped)

pheno.data <- fread("IMPPC_lung_progres.txt", header = FALSE, data.table = FALSE)
colnames(pheno.data) <- pheno.data[1,]
pheno.data <- pheno.data[-1,]

data.map <- data.frame(chr = rep(12, num.snps), snp = rep(0, num.snps), gendist = rep(0, num.snps), bppos = seq(1, num.snps))
data.map["snp"] <- cancer.data.ped[1]

cancer.data.ped <- cancer.data.ped[, -1]
cancer.data.ped <- as.data.frame(t(cancer.data.ped))

num.ind <- nrow(cancer.data.ped)

data2.ped <- data.frame(fid = pheno.data["FID"], id = pheno.data["IID"], sex = pheno.data["sex"], phenotype = pheno.data["progres"])
data2.ped["progres"] <- sapply(c(data2.ped["progres"]), as.numeric) + 1

cancer.data.ped <- cbind(data2.ped, cancer.data.ped)

write.table(cancer.data.ped, file = "chr12_imputed.ped", row.names = FALSE, col.names = FALSE, sep = " ", quote = FALSE)
write.table(data.map, file = "chr12_imputed.map", row.names = FALSE, col.names = FALSE, sep = " ", quote = FALSE)

#/opt/plink-beta-3.38/plink --file chr12_imputed --no-parents --make-bed --out chr12_imputed
#/opt/plink-beta-3.38/plink --bfile chr12_imputed --snps-only --model --out data --covar IMPPC_lung_progres.txt --covar-name PC1,PC2,PC3,PC4,PC5,PC6,PC7,ECOG,fumador,sex

system("/opt/plink-beta-3.38/plink --bfile chr12_imputed_maf001 --allow-no-sex --pheno IMPPC_lung_progres2.txt --pheno-name progres --logistic --covar IMPPC_lung_progres2.txt --covar-name PC1,PC2,PC3,PC4,PC5,PC6,PC7,ECOG,fumador,sex --out genomic_epidemiological")
system("/opt/plink-beta-3.38/plink --bfile chr12_imputed_maf001 --allow-no-sex --pheno IMPPC_lung_progres2.txt --pheno-name progres --logistic --out genomic")

###################################### manhattan plot of SNP's after --model --logistic in plink
library(qqman)
library(muStat)

#model.result <- fread("data.model", data.table = TRUE, header = TRUE)

# logistic.result <- fread("genomic.assoc.logistic", data.table = TRUE, header = TRUE)
logistic.result <- fread("genomic_epidemiological.assoc.logistic", data.table = TRUE, header = TRUE)

#logistic.adjusted.result <- fread("data2.assoc.logistic.adjusted", data.table = TRUE, header = TRUE)

#P <- model.result$CHISQ[which(model.result$TEST == "TREND")]
#P <- model.result$P[which(model.result$TEST == "TREND")]
P <- logistic.result$P[which(logistic.result$TEST == "ADD")]
#P <- model.result$CHISQ
#P <- logistic.adjusted.result$GC
#P.na.indexes <- which.na(P)
#P <- P[-P.na.indexes]

#SNP <- model.result$SNP[which(model.result$TEST == "TREND")]
SNP <- logistic.result$SNP[which(logistic.result$TEST == "ADD")]
#SNP <- model.result$SNP
#SNP <- SNP[-P.na.indexes]

#CHR <- logistic.result$CHR[which(logistic.result$TEST == "ADD")]
#CHR <- CHR[-P.na.indexes]

#logistic.result2 <- cbind(logistic.result$SNP[which(logistic.result$TEST == "ADD")], logistic.result$BP[which(logistic.result$TEST == "ADD")])
#colnames(logistic.result2) <- c("SNP", "BP")

BP <- logistic.result$BP[which(logistic.result$TEST == "ADD")]
#BP <- as.numeric(logistic.result[which(logistic.result[, "SNP"] %in% SNP), "BP"])
#BP <- BP[-P.na.indexes]

CHR <- rep(12, length(SNP))
#BP <- 1:length(SNP)

result <- data.frame(SNP = SNP, CHR = CHR, BP = BP, P = P)

threshold1 <- -1 * log10(0.001)
threshold2 <- -1 * log10(0.003)
threshold3 <- -1 * log10(0.01)

#min.BP <- min(BP)
#max.BP <- max(BP)

# png("genomic_manhattan.png", width=2000, height=1500)
png("genomic_epidemiological_manhattan.png", width=2000, height=1500)
#manhattan(subset(result, CHR == 12), main = "Manhattan Plot", suggestiveline = F, genomewideline = F, xlim = c(0,10), ylim = c(0,7))
manhattan(result, main = "Manhattan Plot", suggestiveline = F, genomewideline = F, xlim = c(0,150), ylim = c(0,8))
#abline(h=5.3,col="red")
abline(h = threshold1, col = "red")
abline(h = threshold2, col = "blue")
abline(h = threshold3, col = "green")
dev.off()

#qq(P, main = "Q-Q plot of GWAS p-values")

############################# create file for locuszoom
significative.SNPs.indexes <- which(BP >= 50000000 & BP <= 52000000)

new.data.logistic <- data.frame(CHR = CHR[significative.SNPs.indexes], SNP = SNP[significative.SNPs.indexes], BP = BP[significative.SNPs.indexes], A1 = logistic.result$A1[which(logistic.result$TEST == "ADD")][significative.SNPs.indexes], TEST = logistic.result$TEST[which(logistic.result$TEST == "ADD")][significative.SNPs.indexes], NMISS = logistic.result$NMISS[which(logistic.result$TEST == "ADD")][significative.SNPs.indexes], OR = logistic.result$OR[which(logistic.result$TEST == "ADD")][significative.SNPs.indexes], STAT = logistic.result$STAT[which(logistic.result$TEST == "ADD")][significative.SNPs.indexes], P = P[significative.SNPs.indexes])

write.table(new.data.logistic, file = "new.data.assoc.logistic", row.names = FALSE, col.names = TRUE, sep = " ", quote = FALSE)

############################# create reduced dataset using threshold chosen using previous manhattan plot.
raw.file <- fread("chr12_imputed_maf001.raw", data.table = FALSE, header = TRUE)

minus.log10.P <- -1 * sapply(P, log10)

A1 <- logistic.result$A1[which(logistic.result$TEST == "ADD")]

################################# dataset with threshold1
#SNPs.extracted.indexes <- which(minus.log10.P > 5.3)
SNPs.extracted.indexes.1 <- which(minus.log10.P > threshold1)
SNPs.extracted.names.1 <- SNP[SNPs.extracted.indexes.1]

A1.1 <- A1[SNPs.extracted.indexes.1]

sufix.1 <- with(data.frame(PREFIX = rep("_", length(A1.1)), A1 = A1.1), paste0(PREFIX, A1))
SNPs.extracted.names.raw.1 <- with(data.frame(NAMES = SNPs.extracted.names.1, SUFIX = sufix.1), paste0(NAMES, SUFIX))

SNPs.extracted.raw.1 <- raw.file[, SNPs.extracted.names.raw.1]

################################# dataset with threshold2
SNPs.extracted.indexes.2 <- which(minus.log10.P > threshold2)
SNPs.extracted.names.2 <- SNP[SNPs.extracted.indexes.2]

A1.2 <- A1[SNPs.extracted.indexes.2]

sufix.2 <- with(data.frame(PREFIX = rep("_", length(A1.2)), A1 = A1.2), paste0(PREFIX, A1))
SNPs.extracted.names.raw.2 <- with(data.frame(NAMES = SNPs.extracted.names.2, SUFIX = sufix.2), paste0(NAMES, SUFIX))

SNPs.extracted.raw.2 <- raw.file[, SNPs.extracted.names.raw.2]

################################# dataset with threshold3
SNPs.extracted.indexes.3 <- which(minus.log10.P > threshold3)
SNPs.extracted.names.3 <- SNP[SNPs.extracted.indexes.3]

A1.3 <- A1[SNPs.extracted.indexes.3]

sufix.3 <- with(data.frame(PREFIX = rep("_", length(A1.3)), A1 = A1.3), paste0(PREFIX, A1))
SNPs.extracted.names.raw.3 <- with(data.frame(NAMES = SNPs.extracted.names.3, SUFIX = sufix.3), paste0(NAMES, SUFIX))

SNPs.extracted.raw.3 <- raw.file[, SNPs.extracted.names.raw.3]

P.extracted.3 <- P[SNPs.extracted.indexes.3]

# write.table(data.frame(SNPs=SNPs.extracted.names.raw.3, P=P.extracted.3), file = "genomic_plink_threshold_0_01_features.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)
write.table(data.frame(SNPs=SNPs.extracted.names.raw.3, P=P.extracted.3), file = "genomic_epidemiological_plink_threshold_0_01_features.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

###################################

pheno.data <- fread("IMPPC_lung_progres.txt", header = FALSE, data.table = FALSE)
colnames(pheno.data) <- pheno.data[1,]
pheno.data <- pheno.data[-1,]

pheno.data <- pheno.data[, 11:17]
pheno.data <- sapply(pheno.data, as.numeric)

#SNPs.extracted.raw <- cbind(SNPs.extracted.raw, )

#ped.file <- fread("chr12_imputed.ped", data.table = FALSE, header = FALSE)

#SNPs.extracted <- ped.file[, (SNPs.extracted.indexes + 4)]

#SNPs.extracted <- raw.file[, SNPs.extracted.names.raw]

############################# Run 'transform_to_binary.py' script.

#reduced.binary.data <- fread("reducedBinaryDataset.csv", header = TRUE, data.table = FALSE)
#ped.file <- fread("chr12_imputed.ped", data.table = FALSE, header = FALSE)

#reduced.binary.data <- cbind(ped.file[, 4], reduced.binary.data)
#colnames(reduced.binary.data)[1] <- 'progres'

#write.table(reduced.binary.data, file = "reducedBinaryDatasetWithPheno.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

#pheno.data <- fread("IMPPC_lung_progres.txt", header = FALSE, data.table = FALSE)
#colnames(pheno.data) <- pheno.data[1,]
#pheno.data <- pheno.data[-1,]

#pheno.data <- pheno.data[, 11:16]
#pheno.data <- sapply(pheno.data, as.numeric)

########################### 
#fam.data <- fread("chr12_imputed.fam", header = FALSE, data.table = FALSE)
#new.pheno.data <- pheno.data
#new.pheno.data[, 1] <- fam.data[, 1]
#new.pheno.data[, 2] <- fam.data[, 2]
#new.pheno.data[, "progres"] <- as.numeric(new.pheno.data[, "progres"])
#new.pheno.data[, "progres"] <- new.pheno.data[, "progres"] + 1
#write.table(new.pheno.data, file = "IMPPC_lung_progres2.txt", row.names = FALSE, col.names = TRUE, sep = " ", quote = FALSE)

raw.file <- raw.file[, 7:ncol(raw.file)]
  
#chr12_imputed_maf001.raw

library(mice)

imp <- mice(pheno.data, m = 1)
pheno.data <- complete(imp)

################ complete raw data
raw.data.with.pheno.data <- cbind(pheno.data[, 1:6], raw.file)
raw.data.with.pheno.data <- cbind(pheno.data[, 7], raw.data.with.pheno.data)
colnames(raw.data.with.pheno.data)[1] <- "progres"

write.table(raw.data.with.pheno.data, file = "complete_dataset_with_pheno.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

num.rows <- nrow(raw.data.with.pheno.data)

x.cancer.data <- raw.data.with.pheno.data[, -1]

num.cols <- ncol(x.cancer.data)

y.cancer.data <- raw.data.with.pheno.data[, 1]

set.seed(123456)
(train <- sample(1:num.rows, round(0.80*num.rows)))
(test <- (1:num.rows)[-train])

train.cancer.data <- cbind(y.cancer.data[train], x.cancer.data[train,])
colnames(train.cancer.data)[1] <- "phenotype"

table(train.cancer.data$phenotype)

write.table(train.cancer.data, file = "complete_dataset_with_pheno_TRAIN.csv", row.names = FALSE, col.names = TRUE, sep=",", quote = FALSE)

test.cancer.data <- cbind(y.cancer.data[test], x.cancer.data[test,])
colnames(test.cancer.data)[1] <- "phenotype"

table(test.cancer.data$phenotype)

write.table(test.cancer.data, file = "complete_dataset_with_pheno_TEST.csv", row.names = FALSE, col.names = TRUE, sep=",", quote = FALSE)

###############################################################

SNPs.extracted.raw.1.with.pheno.data <- cbind(pheno.data[, 1:6], SNPs.extracted.raw.1)
SNPs.extracted.raw.1.with.pheno.data <- cbind(pheno.data[, 7], SNPs.extracted.raw.1.with.pheno.data)
colnames(SNPs.extracted.raw.1.with.pheno.data)[1] <- "progres"

write.table(SNPs.extracted.raw.1.with.pheno.data, file = "reduced_dataset_threshold001_with_pheno.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

SNPs.extracted.raw.2.with.pheno.data <- cbind(pheno.data[, 1:6], SNPs.extracted.raw.2)
SNPs.extracted.raw.2.with.pheno.data <- cbind(pheno.data[, 7], SNPs.extracted.raw.2.with.pheno.data)
colnames(SNPs.extracted.raw.2.with.pheno.data)[1] <- "progres"

write.table(SNPs.extracted.raw.2.with.pheno.data, file = "reduced_dataset_threshold003_with_pheno.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

SNPs.extracted.raw.3.with.pheno.data <- cbind(pheno.data[, 1:6], SNPs.extracted.raw.3)
SNPs.extracted.raw.3.with.pheno.data <- cbind(pheno.data[, 7], SNPs.extracted.raw.3.with.pheno.data)
colnames(SNPs.extracted.raw.3.with.pheno.data)[1] <- "progres"

write.table(SNPs.extracted.raw.3.with.pheno.data, file = "reduced_dataset_threshold01_with_pheno.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

############################# Merge binary data and extra data.

reduced.binary.data <- fread("reducedBinaryDatasetWithPheno.csv", header = TRUE, data.table = FALSE)
extra.data <- fread("extraData.csv", header = TRUE, data.table = FALSE)

final.reduced.data <- cbind(reduced.binary.data, extra.data)

write.table(final.reduced.data, file = "finalReducedBinaryDatasetWithPheno.csv", row.names = FALSE, col.names = TRUE, sep = ",", quote = FALSE)

