'''
Date: September 14, 2017

Author: Apurba Sengupta

Description: Utilizing Decision Tree, Random Forest and k-nearest neighbors classifiers to classify fashion images from the Fashion-MNIST dataset and compare the mean acccuracies obtained from the three classifiers (for more details, see https://www.kaggle.com/zalando-research/fashionmnist)
'''

print __doc__


# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree, ensemble, neighbors, metrics


# Read data from .csv files to Pandas dataframe object for training and test images
images_train = pd.read_csv("fashion-mnist_train.csv")
images_test = pd.read_csv("fashion-mnist_test.csv")

# Convert Pandas dataframe objects to NumPy arrays for training and test data
data_train = np.array(images_train)
data_test = np.array(images_test)

X_train = data_train[:,1:]
y_train = data_train[:,0]

X_test = data_test[:,1:]
y_test = data_test[:,0]



print "\n\n***** Decision Tree classifier *****\n\n"

# Generate classifier for Decision Tree classifer
clf1 = tree.DecisionTreeClassifier(criterion = 'gini')

# Fit training data for the Decision Tree classifier
clf1.fit(X_train, y_train)

# Make predictions for the Decision Tree classifier
y_pred_1 = clf1.predict(X_test)

# Print out accuracy results for Decision Tree classifier
print "Accuracy for Decision Tree classifier: " + str(metrics.accuracy_score(y_test, y_pred_1) * 100) + " %"



print "\n\n***** Random Forest classifier *****\n\n"

# Generate multiple classifiers to find best accuracy model
n_estimators = np.arange(1, 50)
train_accu_rf = np.empty(len(n_estimators))
test_accu_rf = np.empty(len(n_estimators))
pred_accu_rf = np.empty(len(n_estimators))

for i, e in enumerate(n_estimators):

	# Generate classifier for Random Forest classifer
	clf2 = ensemble.RandomForestClassifier(n_estimators = e, criterion = 'gini')

	# Fit training data for the Random Forest classifier
	clf2.fit(X_train, y_train)

	# Find training set accuracy
	train_accu_rf[i] = clf2.score(X_train, y_train)

	# Find test set accuracy
	test_accu_rf[i] = clf2.score(X_test, y_test)

	# Make predictions for the Random Forest classifier
	y_pred_2 = clf2.predict(X_test)

	# Find prediction accuracy
	pred_accu_rf[i] = metrics.accuracy_score(y_test, y_pred_2)


plt.title('Random Forest: Varying Number of Trees')
plt.plot(n_estimators, test_accu_rf, label = 'Testing Accuracy')
plt.plot(n_estimators, train_accu_rf, label = 'Training Accuracy')
plt.plot(n_estimators, pred_accu_rf, label = 'Prediction Accuracy')
plt.legend()
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()

# Print out accuracy results for Random Forest classifier
print "Accuracy for Random Forest classifier: " + str(max(pred_accu_rf) * 100) + " %"



print "\n\n***** k-Nearest Neighbors (k-NN) classifier *****\n\n"

# Generate multiple classifiers to find best accuracy model 
n_neighbors = np.arange(1, 50)
train_accu_knn = np.empty(len(n_neighbors))
test_accu_knn = np.empty(len(n_neighbors))
pred_accu_knn = np.empty(len(n_neighbors))

for i, k in enumerate(n_neighbors):
	
	# Generate classifier for k-Nearest Neighbors (k-NN) classifier
	clf3 = neighbors.KNeighborsClassifier(n_neighbors = k, weights = 'distance', algorithm = 'auto')

	# Fit training data for the k-NN classifier
	clf3.fit(X_train, y_train)

	# Find training set accuracy
	train_accu_knn[i] = clf3.score(X_train, y_train)

	# Find test set accuracy
	test_accu_knn[i] = clf3.score(X_test, y_test)

	# Make predictions for the k-NN classifier
	y_pred_3 = clf3.predict(X_test)

	# Find prediction accuracy
	pred_accu_knn[i] - metrics.accuracy_score(y_test, y_pred_3)

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(n_neighbors, test_accu_knn, label = 'Testing Accuracy')
plt.plot(n_neighbors, train_accu_knn, label = 'Training Accuracy')
plt.plot(n_neighbors, pred_accu_knn, label = 'Prediction Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Print out accuracy results for k-NN classifier
print "Accuracy for k-NN classifier: " + str(max(pred_accu_knn) * 100) + " %"








