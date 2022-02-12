import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.neighbors import KNeighborsRegressor

toy_example = pandas.read_csv("C:\\MScAnalytics\\Data Mining Principles\\Data\\Week 2 Toy Example.csv", header = 0)

# Specify the data
X = toy_example[['x1', 'x2']]
y = toy_example['y']

print(X.describe())
print(y.describe())

# Build nearest neighbors
kNNSpec = KNeighborsRegressor(n_neighbors = 2, metric = 'euclidean')
nbrs = kNNSpec.fit(X, y)
distances, indices = nbrs.kneighbors(X)

# Calculate prediction, errors, and sum of squared error
pred_y = nbrs.predict(X)
error_y = y - pred_y
sse_y = numpy.sum(error_y ** 2)

# Build nearest neighbors
result = pandas.DataFrame()
for k in range(10):
   kNNSpec = KNeighborsRegressor(n_neighbors = (k+1), metric = 'euclidean')
   nbrs = kNNSpec.fit(X, y)
   pred_y = nbrs.predict(X)
   error_y = y - pred_y
   sse_y = numpy.sum(error_y ** 2)
   result = result.append([[(k+1), sse_y]], ignore_index = True)
 
result = result.rename(columns = {0: 'Number of Neighbors', 1: 'Sum of Squared Error'})

plt.scatter(result['Number of Neighbors'], result['Sum of Squared Error'])
plt.xlabel('Number of Neighbors')
plt.ylabel('Sum of Squared Error')
plt.xticks(numpy.arange(1,11,1))
plt.grid(axis = 'both')
plt.show()