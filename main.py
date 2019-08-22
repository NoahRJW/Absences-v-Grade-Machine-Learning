import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from matplotlib.colors import from_levels_and_colors

'''
Created by Noah Wilson
Project for Adv Game Design Mountain View High School
Grade: 12
8/21/19
How to use, you need to get student data in a perferably CSV format, which you can export with Excel or Google Docs, 
make sure you cover the data points the algorithm uses but you can also easily change these this code is free to use 
for any, I would appreciate credit to parts of the code you usebut it is not required.
'''
studentdata = "student-mat.csv"
def graph():
    cmap, norm = from_levels_and_colors([0.0, 0.5, 1.5], ['red', 'blue'])
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    xd = [data[predict]]
    yd = [data["semester1"]]
    zd = [data["absences"]]
    c = data["sex"]
    ax.scatter(xd, yd, zd, c=c, marker='o', cmap=cmap, norm=norm)
    ax.set_xlabel('Final Grade')
    ax.set_ylabel("Semester 1 Grade")
    ax.set_zlabel('Absences')
    pyplot.show()


def training():
    best_score = 0
    while best_score < 0.95:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        print("\nAccuracy: ", accuracy, "%")
        if accuracy > best_score:
            best_score = accuracy
            with open("bestresults.pickle", "wb") as f:
                pickle.dump(model, f)
    print(best_score)


# In sex / gender 0 = female : 1 = male

data = pd.read_csv(studentdata, sep=";")

print(data.head())

data = data[["semester1", "semester2", "failures", "absences", "sex"]]
predict = "semester2"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

# Only train if you are running first or you want to get new data
training()

pickle_in = open("bestresults.pickle", "rb")
model = pickle.load(pickle_in)

print("Co: \n", model.coef_)
print("Inte: \n", +model.intercept_)

predictions = model.predict(x_test).round()

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


graph()
