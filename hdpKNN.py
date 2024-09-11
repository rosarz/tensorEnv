import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.impute import SimpleImputer

data = pd.read_csv("heart_disease_uci.csv", sep=',')
#print(data.head())

#converts strings to ints
le = preprocessing.LabelEncoder()
sex = le.fit_transform(list(data["sex"]))
cp = le.fit_transform(list(data["cp"]))
fbs = le.fit_transform(list(data["fbs"]))
restecg = le.fit_transform(list(data["restecg"]))
exang = le.fit_transform(list(data["exang"]))
slope = le.fit_transform(list(data["slope"]))
thal = le.fit_transform(list(data["thal"]))
#ints
age = list(data["age"])
trestbps = list(data["trestbps"])
chol = list(data["chol"])
thalch = list(data["thalch"])
oldpeak = list(data["oldpeak"])
ca = list(data["ca"])

# Create an imputer object that replaces NaN values with the mean value of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

predict = "num"

x = list(zip(age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal))
y = list(data[predict])

best = 0
for i in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)
    if(acc > best):
        best = acc

print("Best accuracy: ", best)

names = ["no heart disease", "1st stage", "2nd stage", "3rd stage", "stages of heart disease"]

predicted = model.predict(x_test)
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 5, True)
    print("N:", n)