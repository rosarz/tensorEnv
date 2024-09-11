import sklearn
from sklearn import svm
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("heart_disease_health_indicators.csv", sep=',')

x = data["HeartDiseaseorAttack"]
y = data.drop(columns=["HeartDiseaseorAttack"])

x = x[:20000]
y = y[:20000]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(y, x, test_size=0.2)

#print(x_train, y_train)
classes = ["HeartDiseaseorAttack"]

#clf = svm.SVC(kernel="linear", C=2)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)