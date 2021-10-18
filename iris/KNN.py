from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dateset = load_iris()
X = iris_dateset["data"]
Y = iris_dateset["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
print("test set score:{:.2f}".format(knn.score(X_test, Y_test)))

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("prediction:{}".format(prediction))
print("predicted target name:{}".format(iris_dateset["target_names"][prediction]))
