# Example call: Rscript "C:\Users\bmcge\Documents\University\HonoursDefaulterPrediction\src\external_signifigance_testing\signifigance_testing.R" "C:\Users\bmcge\Documents\University\HonoursDefaulterPrediction\results\test.csv" "BACC"

require(PMCMR)
args<-commandArgs(TRUE)
if (length(args) < 2) {
    stop("Path to CSV file and data description expected as argument")
}

y = read.csv(args[1], header=TRUE, sep=",")
description <- args[2]

args <- commandArgs(trailingOnly = F)
scriptPath <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))
currentTime <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
options(digits.secs=6)
outPath <- paste(scriptPath, "\\..\\..\\results\\", "signifigance_tests_", description, "_", currentTime, "_.txt", sep = "")
outPath <- normalizePath(outPath)

y <- as.matrix(y)

friedman <- friedman.test(y)
posthocTest <- posthoc.friedman.nemenyi.test(y=y)
capture.output(print(friedman), print(posthocTest), file = outPath)