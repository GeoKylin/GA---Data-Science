import pandas as pd
data = pd.read_csv("/home/mahbub/Desktop/sourcecode/GA - Data Science/Week 11/train.csv")
from sklearn.cross_validation import train_test_split
df = data[['Survived','Pclass', 'Age']]
df = df.dropna(subset=['Survived','Pclass', 'Age'], how='any')

X_train, X_test, y_train, y_test = cross_validation.train_test_split(df, df['Survived'], test_size=0.3, random_state=0)
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train) 
testData = pd.read_csv("/home/mahbub/Desktop/sourcecode/GA - Data Science/Week 11/test.csv")