import sys, numpy as np
from operator import itemgetter
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print('Needs 2 arguments - <qid\tAP> <qid\tpredicted_qpp>')
    exit(0)

arg_ap = sys.argv[1]
arg_qpp = sys.argv[2]

def read_file(file):
    fp = open(file)
    scores = []
    for line in fp.readlines():
        scores.append(float(line.split('\t')[1]))
    return scores

ap = read_file(arg_ap)
# print('map : ', ap)
qpp = read_file(arg_qpp)
# print('qpp : ', qpp)
corrp, _ = stats.pearsonr(ap, qpp)
print('Kendalls correlation: %.4f', round(corrp, 4))

# spearman's rank co-relation
# Calculate the rank of x's
xranks = pd.Series(ap).rank()
# print("Rankings of X:", xranks)
# Caclulate the ranking of the y's
yranks = pd.Series(qpp).rank()
# print("Rankings of Y:", yranks)

# Calculate Pearson's correlation coefficient on the ranked versions of the data
# print("Spearman's Rank correlation:", scipy.stats.pearsonr(xranks, yranks)[0])
corrs, _ = stats.spearmanr(ap, qpp)
print('Spearmans correlation: %.4f', round(corrs, 4))

# Calculating Kendall Rank correlation
corrk, _ = stats.kendalltau(ap, qpp)
print('Pearsons correlation: %.4f', round(corrk, 4))