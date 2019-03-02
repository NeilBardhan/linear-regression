import csv
import pandas as pd
import math
from itertools import combinations

k = 10 #split data into k folds
filename = "autoInsur.csv"

with open(filename, newline = '') as csvFile:
    df = pd.read_csv(csvFile)
    foldSize = int(math.ceil(df.shape[0]/k))
    print("K Fold splitting")
    wrapper = []
    temp = []
    cnt = 0
    for index, row in df.iterrows():
        temp.append([row['X'], row['Y']])
        if len(temp) == foldSize:
            wrapper.append(temp)
            temp = []
    if(temp):
        wrapper.append(temp)
    for fold in wrapper:
        print("Fold ->", fold)
    print("Data split into", len(wrapper), "folds.")

    print("Iterations")
    combos = list(combinations(wrapper, len(wrapper)-1))
    for elem in combos:
        print("Iteration ->", elem)
        break
