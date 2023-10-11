# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

wine = datasets.load_wine()

wine

# input features

print(wine.feature_names)

# output features

print(wine.target_names)

# glimpse of the data

# input features


wine.data

# output variable (the Class label)

wine.target

# assigning input and output variables

# let's assign the 4 input variables to X and the output variable (class label) to Y


X = wine.data
Y = wine.target

# examine the data dimension

X.shape

Y.shape

# build classification model using random forest

clf = RandomForestClassifier()

clf.fit(X,Y)

# feature importance

print(clf.feature_importances_)

# make prediction

X[58]

print(clf.predict([X[58]]))

print(clf.predict_proba([X[58]]))

# makes the output display class name and not the number

clf.fit(wine.data, wine.target_names[wine.target])

# data split 80/20 ratio

X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size= 0.2)

X_train.shape, Y_train.shape

X_test.shape, Y_test.shape

# rebuild the random forest model

clf.fit(X_train, Y_train)

# performs prdiction on single sample from the data set

print(clf.predict([[1.372e+01, 1.430e+00, 2.500e+00, 1.670e+01, 1.080e+02, 3.400e+00,
       3.670e+00, 1.900e-01, 2.040e+00, 6.800e+00, 8.900e-01, 2.870e+00,
       1.285e+03]]))

print(clf.predict_proba([[1.372e+01, 1.430e+00, 2.500e+00, 1.670e+01, 1.080e+02, 3.400e+00,
       3.670e+00, 1.900e-01, 2.040e+00, 6.800e+00, 8.900e-01, 2.870e+00,
       1.285e+03]]))

# performs prediction on the test set

# predicted class labels


print(clf.predict(X_test))

# actual class labels

print(Y_test)

# model performance

print(clf.score(X_test, Y_test))