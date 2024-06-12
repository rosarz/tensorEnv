import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("heart_disease_uci.csv", sep=',')
data['sex'] = data['sex'].map({'Female': 0, 'Male': 1})
data['dataset'] = data['dataset'].map({'Cleveland': 0, 'Hungarian': 1, 'Switzerland': 2, 'Long Beach VA': 3})
data['cp'] = data['cp'].map({'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3})
data['fbs'] = data['fbs'].map({'FALSE': 0, 'TRUE': 1})
data['restecg'] = data['restecg'].map({'normal': 0, 'ST-T abnormality': 1, 'lv hypertrophy': 2})
data['slope'] = data['slope'].map({'upsloping': 0, 'flat': 1, 'downsloping': 2})
data['thal'] = data['thal'].map({'normal': 0, 'fixed defect': 1, 'reversable defect': 2})

#all data without id
data = data.drop(columns=["id"])
data = data.drop(columns=["dataset"])
#data = data[["id", "sex", "age", "trestbps", "thalch", "chol", "num"]]

# Create an imputer object that replaces NaN values with the mean value of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

predict = "num"

x = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

"""
best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    

    # Fit the imputer on the training data
    x_train = imputer.fit_transform(x_train)

    # Apply the imputer to the test data
    x_test = imputer.transform(x_test)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        print(best)
        with open("heart_disease.pickle", "wb") as f:
            pickle.dump(linear, f)
print("Best acc: ")
print(best) #last best 0.6089522431342025
"""

pickle_in = open("heart_disease.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("{:.2f} {} {}".format(predictions[x], list(map('{:.2f}'.format, x_test[x])), y_test[x]))

p1 = "age"
style.use("ggplot")
plt.scatter(data[p1], data["num"])
plt.xlabel(p1)
plt.ylabel("Heart Disease")
plt.show()