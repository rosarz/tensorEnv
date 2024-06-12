import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

data = pd.read_csv("heart_disease_uci.csv", sep=',')

data = data[["age", "sex", "dataset", "cp", "fbs", "restecg", "slope", "thal", "trestbps", "thalch", "chol", "oldpeak", "num"]]
data['sex'] = data['sex'].map({'Female': 0, 'Male': 1})
data['dataset'] = data['dataset'].map({'Cleveland': 0, 'Hungarian': 1, 'Switzerland': 2, 'Long Beach VA': 3})
data['cp'] = data['cp'].map({'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3})
data['fbs'] = data['fbs'].map({'FALSE': 0, 'TRUE': 1})
data['restecg'] = data['restecg'].map({'normal': 0, 'ST-T abnormality': 1, 'lv hypertrophy': 2})
data['slope'] = data['slope'].map({'upsloping': 0, 'flat': 1, 'downsloping': 2})
data['thal'] = data['thal'].map({'normal': 0, 'fixed defect': 1, 'reversable defect': 2})

print(data.head())

predict = "num"


x = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
