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
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #, fill_value=0

predict = "num"

x = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)


best = 0
for _ in range(100):
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


pickle_in = open("heart_disease.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print("{:.2f} {} {}".format(predictions[x], list(map('{:.2f}'.format, x_test[x])), y_test[x]))

names = ["no heart disease", "1st stage", "2nd stage", "3rd stage", "stages of heart disease"]

predicted = linear.predict(x_test)

for x in range(len(predicted)):
    print("Predicted: ", names[int(predicted[x])], "Data: ", x_test[x], "Actual: ", names[int(y_test[x])])

# p1 = "age"
# style.use("ggplot")
# plt.scatter(data[p1], data["num"])
# plt.xlabel(p1)
# plt.ylabel("Heart Disease")
# plt.show()

# # Grupowanie danych według 'num' i obliczanie średniej wieku
# grouped = data.groupby('num')['age'].mean()
#
# # Tworzenie wykresu
# plt.figure(figsize=(10, 6))
# grouped.plot(kind='bar')
# plt.xlabel('Heart Disease Stage')
# plt.ylabel('Average Age')
# plt.title('Average Age per Stage')
# plt.show()

nums = ["no heart disease", "1st stage", "2nd stage", "3rd stage", "stages of heart disease"]
# Tworzenie słownika do zastąpienia wartości
replace_dict = {i: num for i, num in enumerate(nums)}

# Zastępowanie wartości numerycznych etykietami tekstowymi
data['num'] = data['num'].replace(replace_dict)

# Tworzenie nowej kolumny 'age_group' reprezentującej przedziały wiekowe
bins = [20, 30, 40, 50, 60, 70, 80]
labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

# Grupowanie danych według 'age_group' i 'num', a następnie obliczanie liczby przypadków
grouped = data.groupby(['age_group', 'num']).size().unstack(fill_value=0)

# Tworzenie wykresu
grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Age Group')
plt.ylabel('Number of Cases')
plt.title('Number of Cases per Age Group and Stage')
plt.show()

# Grupowanie danych według 'num' i obliczanie średniego ciśnienia krwi
groupedECG = data.groupby('num')['restecg'].mean()

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
groupedECG.plot(kind='bar')
plt.xlabel('Heart Disease Stage')
plt.ylabel('Resting electrocardiographic results')
plt.title('Resting electrocardiographic results per Stage')
plt.show()