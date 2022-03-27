import numpy

import sklearn.naive_bayes as naive_bayes

# Columns are Carolina, French, Korean, New York, Philly, Texas,
#             Barbecue, Macaron, Souffle, Toast, Streak
X = numpy.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
)

y = numpy.array([1,0,0,1,0,0,1,1,1])

classifier = naive_bayes.MultinomialNB(alpha = 1).fit(X, y)
print('Alpha Value = ', classifier.alpha)

print('Class Count:\n', classifier.class_count_)
print('Log Class Probability:\n', classifier.class_log_prior_ )
print('Feature Count (before adding alpha):\n', classifier.feature_count_)
print('Log Feature Probability:\n', classifier.feature_log_prob_)

predProb = classifier.predict_proba(X)
print('Predicted Conditional Probability (Training):', predProb)

X_test = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])

print('Predicted Conditional Probability (Testing):\n', classifier.predict_proba(X_test))
