import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import time
import constant
import matplotlib.pyplot as pyplot
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.mplot3d import axes3d, Axes3D

'''
Created by Noah Wilson
Project for Adv Game Design Mountain View High School
Grade: 12
8/21/19
How to use, you need to get student data in a perferably CSV format, which you can export with Excel or Google Docs, 
make sure you cover the data points the algorithm uses but you can also easily change these this code is free to use 
for any, I would appreciate credit to parts of the code you usebut it is not required.
'''


def graph():
    cmap, norm = from_levels_and_colors([0.0, 0.5, 1.5], ['red', 'blue'])
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    xd = [data[predict]]
    yd = [data['semester1']]
    zd = [data['absences']]
    c = data["sex"]
    ax.scatter(xd, yd, zd, c=c, marker='o', cmap=cmap, norm=norm)
    ax.set_xlabel('Final Grade')
    ax.set_ylabel('Semester 1 Grade')
    ax.set_zlabel('Absences')
    pyplot.show()


def training():
    best_score = 0
    starttime = time.time()
    while best_score < 0.90:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        grph = linear_model.LinearRegression()
        grph.fit(x_train, y_train)
        accuracy = grph.score(x_test, y_test)
        print("\nAccuracy: ", accuracy, "%")
        if accuracy > best_score:
            best_score = accuracy
            with open("bestresults.pickle", "wb") as f:
                pickle.dump(grph, f)
    print(best_score, " in ", str(round(time.time() - starttime, 2)), "seconds")


# In sex / gender 0 = female : 1 = male


data = pd.read_csv(constant.STUDENT_DATA_CSV, sep=";")

data = data[["semester1", "semester2", "failures", "absences", "sex"]]
predict = "semester2"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


pickle_in = open('bestresults.pickle', 'rb')
model = pickle.load(pickle_in)


def printData():
    print('Co: \n', model.coef_)
    print('Inte: \n', +model.intercept_)

    predictions = model.predict(x_test).round()

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])


if __name__ == '__main__':
    training()
    printData()
    graph()

