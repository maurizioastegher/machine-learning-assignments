#! /usr/local/bin/python
#
# Machine Learning - Assignment 2
# Astegher Maurizio 175195

import numpy
from sklearn import datasets, cross_validation, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def rbf_svm(X_train, y_train, X_test, C):
	clf = SVC(C=C, kernel="rbf", class_weight="balanced", random_state=None)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

def random_forest(X_train, y_train, X_test, N_EST):
	clf = RandomForestClassifier(n_estimators = N_EST, criterion="gini", random_state=None)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

# Create a classification dataset (1500 samples, 15 features)
dataset = datasets.make_classification(n_samples=1500, n_features=15, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2)

data = dataset[0]
target = dataset[1]
N = len(data)

# Split the dataset using 10-fold cross-validation
kf = cross_validation.KFold(N, n_folds=10, shuffle=True, random_state=None)

acc = {} # acc[0] -> Gaussian, acc[1] -> SVM, acc[2] -> Random Forest
f1 = {}
auc = {}
for i in range(0, 3):
	acc[i] = []
	f1[i] = []
	auc[i] = []

for train_index, test_index in kf:
	X_train, X_test = data[train_index], data[test_index]
	y_train, y_test = target[train_index], target[test_index]
	NN = len(X_train)

	# Gaussian Naive Bayes 
	clf = GaussianNB()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	acc[0].append(metrics.accuracy_score(y_test, pred))
	f1[0].append(metrics.f1_score(y_test, pred))
	auc[0].append(metrics.roc_auc_score(y_test, pred))

	# SVM
	bestC = None
	CValues = [1e-2, 1e-1, 1e0, 1e1, 1e2]
	innerscore = []
	# Inner 5-fold cross-validation for parameter selection (C)
	for C in CValues:
		ikf = cross_validation.KFold(NN, n_folds=5, shuffle=True, random_state=5678)
		innerf1 = []
		for t_index, v_index in ikf:
			X_t, X_v = X_train[t_index], X_train[v_index]
			y_t, y_v = y_train[t_index], y_train[v_index]

			ipred = rbf_svm(X_t, y_t, X_v, C)
			# Save the F1-score of the inner cross-validation
			innerf1.append(metrics.f1_score(y_v, ipred))
		# Compute the average
		innerscore.append(sum(innerf1) / len(innerf1))
	
	# Pick the C that gives best F1-score
	bestC = CValues[numpy.argmax(innerscore)]
	# Predict the labels for the test set using the best C parameter
	pred = rbf_svm(X_train, y_train, X_test, bestC)

	acc[1].append(metrics.accuracy_score(y_test, pred))
	f1[1].append(metrics.f1_score(y_test, pred))
	auc[1].append(metrics.roc_auc_score(y_test, pred))

	# Random Forest classifier
	bestN_EST = None
	N_EST_Values = [10, 100, 1000]
	innerscore = []
	# Inner 5-fold cross-validation for parameter selection (N_EST)
	for N_EST in N_EST_Values:
		ikf = cross_validation.KFold(NN, n_folds=5, shuffle=True, random_state=5678)
		innerf1 = []
		for t_index, v_index in ikf:
			X_t, X_v = X_train[t_index], X_train[v_index]
			y_t, y_v = y_train[t_index], y_train[v_index]

			ipred = random_forest(X_t, y_t, X_v, N_EST)
			# Save the F1-score of the inner cross-validation
			innerf1.append(metrics.f1_score(y_v, ipred))
		# Compute the average
		innerscore.append(sum(innerf1) / len(innerf1))
	
	# Pick the N_EST that gives best F1-score
	bestN_EST = N_EST_Values[numpy.argmax(innerscore)]
	# Predict the labels for the test set using the best N_EST parameter
	pred = random_forest(X_train, y_train, X_test, bestN_EST)

	acc[2].append(metrics.accuracy_score(y_test, pred))
	f1[2].append(metrics.f1_score(y_test, pred))
	auc[2].append(metrics.roc_auc_score(y_test, pred))

print "\n- Gaussian Naive Bayes -"
print "Accuracy:", (float(sum(acc[0])) / len(acc[0]))
print "F1-score:", (float(sum(f1[0])) / len(f1[0]))
print "AUC ROC:", (float(sum(auc[0])) / len(auc[0]))

print "\n- SVM -"
print "Accuracy:", (float(sum(acc[1])) / len(acc[1]))
print "F1-score:", (float(sum(f1[1])) / len(f1[1]))
print "AUC ROC:", (float(sum(auc[1])) / len(auc[1]))
print "C value:", bestC

print "\n- Random Forest Classifier -"
print "Accuracy:", (float(sum(acc[2])) / len(acc[2]))
print "F1-score:", (float(sum(f1[2])) / len(f1[2]))
print "AUC ROC:", (float(sum(auc[2])) / len(auc[2]))
print "N_EST value", bestN_EST
print ""