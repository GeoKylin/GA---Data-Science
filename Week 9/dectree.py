#!/usr/local/bin/python

# do these first:
# 1) install dependency pyparsing (from source!)
# 2) pip install pydot

# these are part of the python standard library (psl)
from collections import Counter
import StringIO

# this is a third-party library (requires installation)
import pydot

# sklearn is also a third-party library (requires installation)
from sklearn.datasets import load_iris
from sklearn import tree

def main():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    # create visualization using graphviz (first: brew install graphviz)
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('iris_dectree.pdf')
    print '\nimage created!'

    # print predictions 
    print '\npredictions:'
    print clf.predict(iris.data)

    # print Counter object with predictions (note: no training error!)
    print '\npredictions Counter:'
    print Counter(clf.predict(iris.data))

if __name__ == '__main__':
    main()

# ref: http://scikit-learn.org/dev/modules/tree.html#classification

# NOTE: to get other file: 
#           install numpy, matplotlib
#           curl -O http://scikit-learn.org/dev/_downloads/plot_iris1.py
