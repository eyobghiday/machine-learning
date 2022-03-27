# -*- coding: utf-8 -*-
"""
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
"""
import numpy
import pandas
import sys

import itertools

sys.path.append('C:\\IIT\\Machine Learning\\Job')
import Utility

# Set some options for printing all the columns
numpy.set_printoptions(precision = 7, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.set_option('precision', 10)

pandas.options.display.float_format = '{:,.7e}'.format

cars = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\cars.csv')

predictor = ['Horsepower', 'Weight', 'Length']
n_predictor = len(predictor)

target = 'MSRP'

trainData = cars[predictor + [target]].dropna().reset_index(drop = True)

X = trainData[predictor]
X.insert(0, 'Intercept', 1.0)

y = trainData[target]

n_sample = X.shape[0]
step_summary = pandas.DataFrame()

# Sum of deviation from the target mean
SS_y = n_sample * numpy.var(y, ddof = 0)

all_combination = itertools.product(range(2), repeat = n_predictor)
i_comb = 0
for comb in all_combination:
    i_comb = i_comb + 1
    var_in_model = ['Intercept']
    for j in range(n_predictor):
        if (comb[j] == 1):
            var_in_model.append(predictor[j])
    resultList = Utility.LinearRegressionModel(X[var_in_model], y)
    modelDF = len(resultList[4])
    SSE = resultList[1]
    AIC = n_sample * numpy.log(SSE/n_sample) + 2.0 * modelDF
    R_Square = 1.0 - (SSE / SS_y)
    if (R_Square < 0.0):
        R_Square = 0.0
    step_summary = step_summary.append([[i_comb, var_in_model, modelDF, SSE, R_Square, AIC]], ignore_index = True)

step_summary.columns = ['Index', 'Term Entered', 'Number of Parameters', 'Residual Sum of Squares', 'R-Square', 'AIC']

print('All Possible Summary')
print(step_summary)
