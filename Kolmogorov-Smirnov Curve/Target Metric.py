# Name: Week 8 Nominal Target Metric.py
# Creation Date: February 14, 2022
# Author: Ming-Long Lam
# Organization: Illinois Institute of Technology

import numpy
import pandas

scoreData = pandas.DataFrame({'Y': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
                              'P_A': [0.47, 0.13, 0.33, 0.47, 0.37, 0.47, 0.5, 0.47, 0.33, 0, 0.47, 0.47, 0.33, 0.47, 0.47, 0, 0.47, 0, 0.47, 0.47],
                              'P_B': [0.13, 0.4, 0.34, 0.13, 0.13, 0.13, 0.5, 0.13, 0.34, 0.33, 0.13, 0.13, 0.34, 0.13, 0.13, 0.33, 0.13, 0.33, 0.13, 0.13],
                              'P_C': [0.4, 0.47, 0.33, 0.4, 0.5, 0.4, 0, 0.4, 0.33, 0.67, 0.4, 0.4, 0.33, 0.4, 0.4, 0.67, 0.4, 0.67, 0.4, 0.4]})

# Inputs:
# y: a Pandas Series that contains the actual target categories
# predProb: a Pandas DataFrame that contains the predicted probabilities
#           column names must have prefix P_ followed by target category value
#           column order must match the ascending order of the target categories
# Output:
# outMetric: a Pandas Series that contains the metric values
#            MCE = Misclassification Rate
#            ASE = Average Squared Error
#            RASE = Root Average Squared Error
#            AUC = Area Under Curve

def NominalMetric (y, predProb):
   
   n = predProb.shape[0]          # Number of observations
   K = predProb.shape[1]          # Number of target categories

   # Retrieve target categories with the prefix P_
   y_cat = predProb.columns

   # Predicted target category
   j_max = predProb.values.argmax(axis = 1)
   predYCat = y_cat[j_max]

   # Misclassification rate
   yWithP_ = 'P_' + y
   qMisClass = numpy.where(predYCat == yWithP_, 0, 1)

   # Root Average Squared Error
   delta = pandas.DataFrame(numpy.zeros((n,K)), columns = y_cat)
   for col in y_cat:
      delta[col] = numpy.where(yWithP_ == col, 1.0, 0.0)
   ase = numpy.mean(numpy.mean((delta - predProb.reset_index()) ** 2))
   
   # Area Under Curve
   nComb = 0
   auc = 0.0
   for row in y_cat:
      eProb = predProb[row][yWithP_ == row]
      for col in y_cat:
         if (row != col):
            neProb = predProb[row][yWithP_ == col]

            # Calculate the number of concordant, discordant, and tied pairs
            nConcordant = 0
            nDiscordant = 0
            nTied = 0
            for eP in eProb:
               nConcordant = nConcordant + numpy.sum(numpy.where(neProb < eP, 1, 0))
               nDiscordant = nDiscordant + numpy.sum(numpy.where(neProb > eP, 1, 0))
               nTied = nTied + numpy.sum(numpy.where(neProb == eP, 1, 0))
            nPairs = nConcordant + nDiscordant + nTied
            if (nPairs > 0):
               nComb = nComb + 1
               auc = auc + 0.5 + 0.5 * (nConcordant - nDiscordant) / nPairs
   if (nComb > 0):
      auc = auc / nComb
   else:
      auc = numpy.nan

   outMetric = pandas.Series({'MCE': numpy.mean(qMisClass),
                              'ASE': ase,
                              'RASE': numpy.sqrt(ase),
                              'AUC': auc})
   return (outMetric)

# Calculate the metrics for the entire data
outMetric = NominalMetric(y = scoreData['Y'], predProb = scoreData[['P_A', 'P_B', 'P_C']])

# Calculate the metrics only for Y = 'A'
subsetData = scoreData[scoreData['Y'] == 'A']
outMetric_A = NominalMetric(y = subsetData['Y'], predProb = subsetData[['P_A', 'P_B', 'P_C']])

# Calculate the metrics only for Y = 'B'
subsetData = scoreData[scoreData['Y'] == 'B']
outMetric_B = NominalMetric(y = subsetData['Y'], predProb = subsetData[['P_A', 'P_B', 'P_C']])

# Calculate the metrics only for Y = 'C'
subsetData = scoreData[scoreData['Y'] == 'C']
outMetric_C = NominalMetric(y = subsetData['Y'], predProb = subsetData[['P_A', 'P_B', 'P_C']])
