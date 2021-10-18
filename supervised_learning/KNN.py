import mglearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:", X.shape)

mglearn.plots.plot_knn_classification(n_neighbors=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions:", clf.predict(X_test))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, alpha=.4)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("{} neighbor(s)".format(3))
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.show()
