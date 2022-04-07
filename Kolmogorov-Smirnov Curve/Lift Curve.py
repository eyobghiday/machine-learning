# Load the necessary libraries
import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas

from sklearn.model_selection import train_test_split
from sklearn import tree

# Define a function to compute the coordinates of the Lift chart
def compute_lift_coordinates (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Debug = 'N'):    # Show debugging information (Y/N)

    # Find out the number of observations
    nObs = len(DepVar)

    # Get the quantiles
    quantileCutOff = numpy.percentile(EventPredProb, numpy.arange(0, 100, 10))
    nQuantile = len(quantileCutOff)

    quantileIndex = numpy.zeros(nObs)
    for i in range(nObs):
        iQ = nQuantile
        EPP = EventPredProb[i]
        for j in range(1, nQuantile):
            if (EPP > quantileCutOff[-j]):
                iQ -= 1
        quantileIndex[i] = iQ

    # Construct the Lift chart table
    countTable = pandas.crosstab(quantileIndex, DepVar)
    decileN = countTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = countTable[EventValue]
    totalNResponse = gainN.sum(0)
    gainPct = 100 * (gainN /totalNResponse)
    responsePct = 100 * (gainN / decileN)
    overallResponsePct = 100 * (totalNResponse / nObs)
    lift = responsePct / overallResponsePct

    LiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                    axis = 1, ignore_index = True)
    LiftCoordinates = LiftCoordinates.rename({0:'Decile N',
                                              1:'Decile %',
                                              2:'Gain N',
                                              3:'Gain %',
                                              4:'Response %',
                                              5:'Lift'}, axis = 'columns')

    # Construct the Accumulative Lift chart table
    accCountTable = countTable.cumsum(axis = 0)
    decileN = accCountTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = accCountTable[EventValue]
    gainPct = 100 * (gainN / totalNResponse)
    responsePct = 100 * (gainN / decileN)
    lift = responsePct / overallResponsePct

    accLiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                       axis = 1, ignore_index = True)
    accLiftCoordinates = accLiftCoordinates.rename({0:'Acc. Decile N',
                                                    1:'Acc. Decile %',
                                                    2:'Acc. Gain N',
                                                    3:'Acc. Gain %',
                                                    4:'Acc. Response %',
                                                    5:'Acc. Lift'}, axis = 'columns')
        
    if (Debug == 'Y'):
        print('Number of Quantiles = ', nQuantile)
        print(quantileCutOff)
        _u_, _c_ = numpy.unique(quantileIndex, return_counts = True)
        print('Quantile Index: \n', _u_)
        print('N Observations per Quantile Index: \n', _c_)
        print('Count Table: \n', countTable)
        print('Accumulated Count Table: \n', accCountTable)

    return(LiftCoordinates, accLiftCoordinates)

# Read the HMEQ data
hmeq = pandas.read_csv('hmeq.csv',
                       delimiter=',', usecols = ['BAD', 'DEBTINC', 'DELINQ', 'DEROG'])

# Drop all NaN values in BAD column
hmeq = hmeq.dropna(subset=['BAD'])

# Replace all NaN values in columns 'DEBTINC', 'DELINQ', and 'DEROG' with 300, -1, and -1 respectively
values = {'DEBTINC': 300, 'DELINQ': -1, 'DEROG': -1}
hmeq = hmeq.fillna(value = values)

# Partition the data
hmeq_train, hmeq_test = train_test_split(hmeq, test_size = 0.3, random_state = 60616, stratify = hmeq['BAD'])

# Build a CART model using the training partition
y_train = hmeq_train[['BAD']].astype('category')

X_name = ['DEBTINC', 'DELINQ', 'DEROG']
X_train = hmeq_train[X_name]

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=60616)
hmeq_tree = classTree.fit(X_train, y_train)
tree_accuracy = classTree.score(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(tree_accuracy))

dot_data = tree.export_graphviz(hmeq_tree,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = X_name,
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)

graph

# Score the training partition
y_train_predProb = classTree.predict_proba(X_train)

# Get the Lift chart coordinates
lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = y_train['BAD'],
        EventValue = 1,
        EventPredProb = y_train_predProb[:,1],
        Debug = 'Y')

# Draw the Lift chart
plt.plot(lift_coordinates.index, lift_coordinates['Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("Training Partition")
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Lift")
plt.show()

# Draw the Accumulative Lift chart
plt.plot(acc_lift_coordinates.index, acc_lift_coordinates['Acc. Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("Training Partition")
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.show()

# Score the test partition
y_test = hmeq_test[['BAD']].astype('category')

X_test = hmeq_test[X_name]

y_test_predProb = classTree.predict_proba(X_test)

# Get the Lift chart coordinates
lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = y_test['BAD'],
        EventValue = 1,
        EventPredProb = y_test_predProb[:,1],
        Debug = 'Y')

# Draw the Lift chart
plt.plot(lift_coordinates.index, lift_coordinates['Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("Testing Partition")
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Lift")
plt.show()

# Draw the Accumulative Lift chart
plt.plot(acc_lift_coordinates.index, acc_lift_coordinates['Acc. Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("Testing Partition")
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.show()
