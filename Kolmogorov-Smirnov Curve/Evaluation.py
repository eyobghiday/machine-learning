# -*- coding: utf-8 -*-
"""
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.set_option('precision', 10)

pandas.options.display.float_format = '{:,.10}'.format

sys.path.append('C:\\IIT\\Machine Learning\\Job')
import Utility

Y = numpy.array(['Event',
                 'Non-Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Non-Event'])

predProbEvent = numpy.array([0.9,0.5,0.3,0.7,0.3,0.8,0.4,0.2,1,0.5,0.3])

# Calculate the binary model metrics
outSeries = Utility.binary_model_metric (Y, 'Event', 'Non-Event', predProbEvent, eventProbThreshold = 0.5)

print('                  Accuracy: {:.13f}' .format(1.0-outSeries['AUC']))
print('    Misclassification Rate: {:.13f}' .format(outSeries['AUC']))
print('          Area Under Curve: {:.13f}' .format(outSeries['AUC']))
print('Root Average Squared Error: {:.13f}' .format(outSeries['RASE']))

# Generate the coordinates for the ROC curve
outCurve = Utility.curve_coordinates (Y, 'Event', 'Non-Event', predProbEvent)

Threshold = outCurve['Threshold']
Sensitivity = outCurve['Sensitivity']
OneMinusSpecificity = outCurve['OneMinusSpecificity']
Precision = outCurve['Precision']
Recall = outCurve['Recall']
F1Score = outCurve['F1Score']

# Draw the ROC curve
plt.figure(dpi = 200)
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.xticks(numpy.arange(0.0,1.1,0.1))
plt.yticks(numpy.arange(0.0,1.1,0.1))
plt.show()

# Draw the Kolmogorov Smirnov curve
plt.figure(dpi = 200)
plt.plot(Threshold, Sensitivity, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(Threshold, OneMinusSpecificity, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()

# Draw the Precision-Recall curve
plt.figure(dpi = 200)
plt.plot(Recall, Precision, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [6/11, 6/11], color = 'red', linestyle = ':', label = 'No Skill')
plt.grid(True)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Draw the F1 Score curve
plt.figure(dpi = 200)
plt.plot(Threshold, F1Score, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.show()