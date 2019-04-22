import os
import time
import json
import numpy
import pandas as pd
import csv
import matplotlib.pyplot as plt
import math
from scipy import stats
import pprint
from prettytable import PrettyTable

# filenames = ["test10k.csv", "test20k.csv"]
# filenames = ["autoInsur.csv"]
# filenames = ["test10k.csv"]

def simpleLinear(filename):

    with open(filename, newline='') as csvFile:
        start = time.time()
        df = pd.read_csv(csvFile)
        elapsed = time.time() - start
        print("Read Time ->", round(elapsed, 4), "seconds.")
        start = time.time()
        x = df['X']
        y = df['Y']
        avg_x = x.mean()
        avg_y = y.mean()
        diff_x = []
        diff_y = []
        for index, row in df.iterrows():
            diff_x.append(row['X'] - avg_x)
            diff_y.append(row['Y'] - avg_y)
        b1_numerator = sum([i * j for i, j in zip(diff_x, diff_y)])
        b1_denominator = sum([i ** 2 for i in diff_x])
        b1 = b1_numerator/b1_denominator
        b0 = avg_y - (b1 * avg_x)
        yhat = []
        residuals = []
        for index, row in df.iterrows():
            yhat.append(b0 + b1*row['X'])
            residuals.append(row['Y'] - (b0 + b1*row['X']))
        tss = sum([i ** 2 for i in diff_y])
        rss = sum([i ** 2 for i in residuals])
        rse = math.sqrt(rss/(df.shape[0] - 2))
        stdErrB0 = math.sqrt((rse ** 2) * (1/df.shape[0] + (avg_x ** 2)/b1_denominator))
        stdErrB1 = math.sqrt((rse ** 2)/b1_denominator)
        b0CI = [round(b0 - 2*stdErrB0, 4), round(b0 + 2*stdErrB0, 4)]
        b1CI = [round(b1 - 2*stdErrB1, 4), round(b1 + 2*stdErrB1, 4)]
        b0Tstat = b0/stdErrB0
        b1Tstat = b1/stdErrB1
        b0pval = stats.t.sf(abs(b0Tstat), df.shape[0] - 2)*2
        b1pval = stats.t.sf(abs(b1Tstat), df.shape[0] - 2)*2
        Rsquared = 1 - rss/tss
        adjRsquared = 1 - ((1 - Rsquared) * (df.shape[0] - 1))/(df.shape[0] - 2)
        fStat = (tss - rss)/(rss/(df.shape[0] - 2))
        # print("yhat ->", yhat)
        # print("residuals ->", residuals)
        # f, ax = plt.subplots(figsize = (6, 4), dpi= 300)
        # ax.scatter(x, y)
        # ax.plot(x, yhat, color = 'red')
        elapsed = time.time() - start
        results = {
            "intercept" : {
                "b0" : round(b0, 4),
                "std_error" : round(stdErrB0, 4),
                "confidence_interval" : b0CI,
                "t_value" : round(b0Tstat, 4),
                "p_value" : round(b0pval, 6)
                },
            "slope" : {
                "b1" : round(b1, 4),
                "std_error" : round(stdErrB1, 4),
                "confidence_interval" : b1CI,
                "t_value" : round(b1Tstat, 4),
                "p_value" : round(b1pval, 6)
                },
            "equation" : "y = " + str(round(b0, 4)) + " + " + str(round(b1, 4)) + " * x",
            "rss" : round(rss, 4),
            "tss" : round(tss, 4),
            "rse" : round(rse, 4),
            "R_squared" : round(Rsquared, 4),
            "adjRsquared" : round(adjRsquared, 4),
            "f_statistic": round(fStat, 2),
            "degrees_of_freedom" : df.shape[0] - 2,
            "predicted_y" : yhat,
            "residuals" : residuals
            }
        print("Computation Time ->", round(elapsed, 4), "seconds.")
        df['yhat'] = pd.Series(yhat).values
        df['residuals'] = pd.Series(residuals).values
        t = PrettyTable(['X', 'Y', 'Yhat', 'residuals'])
        for index, row in df.iterrows():
            # t.add_row([row['X'], row['Y'], row['yhat'], row['residuals']])
            t.add_row(row)
        print(t)
        return results

def main():
    path = os.getcwd()
    filename = path + '\\' + "autoInsur.csv"
    print("+----------+")
    print("Model Output")
    print("+----------+")
    results = simpleLinear(filename)
    with open('simpleLinearModel.json', 'w') as outfile:
        json.dump(results, outfile)
    print("+-----------------+")
    print("Model File Written.")
    print("+-----------------+")
    #pprint.pprint(results, width = 1)

if __name__ == "__main__":
    main()
