# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Exploring and plotting data

# <headingcell level=2>

# Load data and find out what it is

# <codecell>

from sklearn.datasets import load_iris
iris = load_iris()
print iris.DESCR

# <headingcell level=2>

# About this data set

# <codecell>

#class labels are called target names
print "Label names",  "\n", iris.target_names
#feature map to our columns
print "Column names" ,  "\n",  iris.feature_names


num_recs = len(iris.data)
print "The first few rows in our data set" ,  "\n",   iris.data[0:2]
print "Number of rows in the data set", num_recs
print "The first two columns of the first few rows in our data set" ,  "\n",   iris.data[0:2, 0:2]
print type(iris.data)

# <headingcell level=2>

# Operations on rows and columns

# <codecell>


avg_first_col = np.mean(iris.data[:,0])
smaller_than_mean = len(iris.data[iris.data[:,0]<avg_first_col])
greater_than_mean = len(iris.data[iris.data[:,0]>=avg_first_col])

print "{0} records have a '{1}' smaller than average and {2} records have a {1} greater than average".format(smaller_than_mean,iris.feature_names[0],greater_than_mean)

# <headingcell level=2>

# Plot data

# <codecell>

#No need to import matplotlib because this iPython notebook loads matplotlib, numpy and scipy by default
#http://matplotlib.org/api/pyplot_api.html?highlight=scatter#matplotlib.pyplot.scatter
#Matplotlib is the the ggplot2 of the Python world. Does some useful things like accepting latex formatting and saves to SVG (You can share the output of your analysis as a chart users can view in their browser)
plt.scatter(iris.data[:,0],iris.data[:,1])

# <codecell>

#We would like a scatterplot matrix like what we get from R's plot() . Using matplotlib we'd need 
#to write code to achieve this - http://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
#Is there a differrent way of achieving this?
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
df = DataFrame(iris.data, columns=iris.feature_names)
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
#http://pandas.pydata.org/pandas-docs/stable/visualization.html
#It can be easily achieved with the Pandas library - http://pandas.pydata.org
#Things it does well: extends Python data structures to emulate R's data frames, allows join, merge, group by operations, dealing with missing data, backfilling time series data, and many other things

# <headingcell level=2>

# Building a test and training set

# <codecell>

#There are better ways to do this but this is an excuse to write some Python
#>>> indices = np.random.permutation(len(iris_X))
#>>> iris_X_train = iris_X[indices[:-10]]
import random
data = iris.data
train_pct = 0.70
train_sample = []
#num recs was created in the 'About this data set' cell, note scope
num_train = int(num_recs * train_pct)

#sample
while len(train_sample) < num_train:
    num = randint(0, num_recs)
    if num not in train_sample: train_sample.append(num)
        
print "Is it {0} =  {1}?".format(num_train, len(train_sample))
print train_sample

# <codecell>

#Using a list to index an n-dimensional array (numpy array)
train = data[train_sample,:]
#label data is .target
train_labels = iris.target[train_sample,:]
#list comprehension
test_sample = [x for x in range(0, num_recs) if x not in train_sample]
test = data[test_sample,:]
test_labels = iris.target[test_sample,:]

print num_recs, len(train), len(train_labels), len(test), len(test_labels) 

# <headingcell level=2>

# Naive Bayesian Classification

# <codecell>

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
bn_model = gnb.fit(train, train_labels)

label_pred = bn_model.predict(test)
print "Number of mislabeled points : %d" % (test_labels != label_pred).sum()

# <codecell>

print test_labels
print label_pred
(test_labels != label_pred)

# <headingcell level=2>

# k-Nearest Neighbors

# <codecell>

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train, train_labels)
knn_pred = knn.predict(test)
print "Number of mislabeled points : %d" % (test_labels != knn_pred).sum()

#warning message

# <codecell>

#Different number of k
knn2 = KNeighborsClassifier(n_neighbors=150)
knn2.fit(train, train_labels)
knn_pred2 = knn2.predict(test)
print "Number of mislabeled points : %d" % (test_labels != knn_pred2).sum()

# <headingcell level=2>

# Using cross-validation

# <codecell>

#http://scikit-learn.org/dev/modules/cross_validation.html
from sklearn import cross_validation
knn = KNeighborsClassifier()
scores = cross_validation.cross_val_score(knn, iris.data, iris.target, cv=5)
print scores
print mean(scores)

#could also use to get test and training set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
print len(X_train), len(X_test), len(y_train), len(y_test)

# <headingcell level=2>

# K Means Clustering

# <headingcell level=2>

# Read from file

# <codecell>

import pandas
#http://www.kaggle.com/c/facial-keypoints-detection/data
data_set = pandas.read_csv("/Users/davidasfaha/Downloads/training.csv")

#read and transform data
#do k-means
#save data file for future use

# <codecell>

#Read compressed csv file from web
#data_set = pandas.read_csv("http://www.kaggle.com/c/facial-keypoints-detection/download/training.zip",compression="
#Couldn't actually get it to work but in theory pandas allows you to do this. Useful because data in your company might be made available to you as a zip on a webpage

# <codecell>

#It is an object of type data frame. It borrows ideas from R but the syntax is all Python
type(data_set)

# <codecell>

#Name of columns
data_set.columns
#row indices
data_set.index
#get some rows - returns a data frame
data_set[0:2]

# <codecell>

eye_x = pandas.concat([data_set['left_eye_center_x'], data_set['right_eye_center_x']]).dropna()
eye_y = pandas.concat([data_set['left_eye_center_y'], data_set['right_eye_center_y']]).dropna()

# <codecell>

plt.scatter(eye_x,eye_y)

# <codecell>


#http://scikit-learn.org/dev/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans
#default clusters: 8
kmeans = KMeans(n_clusters=2)

# <codecell>

eye_df = DataFrame( { "x": eye_x, "y": eye_y})
clusters = kmeans.fit_predict(eye_df)

# <codecell>


plt.scatter(eye_x,eye_y, c=clusters, cmap=mpl.cm.winter)

# <codecell>


