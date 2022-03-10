import numpy
import pandas

import sklearn.metrics as metrics

# A function to identify cluster membership using the Euclidean distance
def euclideanCluster (df, nCluster, nIteration = 500, tolerance = 1e-7):

   # Initialize
   X = df.to_numpy()
   centroid = X[range(nCluster),:]
   member_prev = numpy.zeros(df.shape[0])

   for iter in range(nIteration):
      distance = metrics.pairwise.euclidean_distances(X, centroid)
      member = numpy.argmin(distance, axis = 1)
      wc_distance = numpy.min(distance, axis = 1)

      print('==================')
      print('Iteration = ', iter)
      print('Centroid: \n', centroid)
      print('Distance: \n', distance)
      print('Member: \n', member)

      for cluster in range(nCluster):
         inCluster = (member == cluster)
         if (numpy.sum(inCluster) > 0):
            centroid[cluster,:] = numpy.mean(X[inCluster,], axis = 0)

      member_diff = numpy.sum(numpy.abs(member - member_prev))
      if (member_diff > 0):
          member_prev = member
      else:
          break

   return (member, centroid, wc_distance)

df = pandas.DataFrame({'x': [0.1, 0.3, 0.4, 0.8, 0.9]})

member, centroid, wc_distance = euclideanCluster(df, 2)