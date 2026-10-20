import numpy as np
import sklearn
from pyod.models import knn
import pyod.utils.data
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = pyod.utils.data.generate_data(n_train=400, n_test=100, n_features=2,
                                                                 contamination=0.1)
# 1)
# outliner_indexes = np.where(Y_train == 1.0)
# inliner_indexes = np.where(Y_train == 0.0)
# outliners = X_train[outliner_indexes]
# inliners = X_train[inliner_indexes]
#
# plt.scatter(inliners[:, 0], inliners[:, 1])
# plt.scatter(outliners[:, 0], outliners[:, 1])
# plt.show()

# 2)
# k_neigh = knn.KNN(contamination=0.1)
# knn.KNN.fit(k_neigh, X_train)
# pred_train = knn.KNN.predict(k_neigh, X_train)
# pred_test = knn.KNN.predict(k_neigh, X_test)
# CM = sklearn.metrics.confusion_matrix(Y_train, pred_train)
# print(CM)
#
# TN, FP, FN, TP = CM[0][0], CM[0][1], CM[1][0], CM[1][1]
# TPR = TP / (TP + FN)
# TNR = TN / (TN + FP)
# BA = (TPR + TNR) / 2
# print(BA)
# scores_test = k_neigh.decision_function(X_test)
# fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=Y_test, y_score=scores_test)
# plt.plot(fpr, tpr)
# plt.show()


# 3)
# X_train, Y_train = pyod.utils.data.generate_data(n_train=1000, n_features=1, contamination=0.1, train_only=True)
# x = X_train.ravel()  # sa il faci din 1000,1 in 1000,
# z = (x - np.mean(x)) / np.std(x)
# abs_z = np.abs(z)
# threshold = np.quantile(abs_z, 1 - 0.1)
#
# pred = (abs_z > threshold).astype(int)  # 1 = anomalie, 0 = normal
# CM = sklearn.metrics.confusion_matrix(Y_train, pred)
# TN, FP, FN, TP = CM[0][0], CM[0][1], CM[1][0], CM[1][1]
# BA = (TP / (TP + FN) + TN / (TN + FP)) / 2
# print("Prag Z:", threshold)
# print("Balanced accuracy:", BA)

# 4)
n = 1000
contamination = 0.1
n_out = int(n * contamination)
n_in = n - n_out
d = 2
mu = np.array([5.0, -3.0])
Sigma = np.array([[4.0, 1.5],
                  [1.5, 1.0]])  # ca sa fac cholesky trb sa fie simetrica si poz def

L = np.linalg.cholesky(Sigma)
x = np.random.randn(n_in, d)
Y_in = x @ L.T + mu

anomalies = np.random.uniform(-15, 20, size=(n_out, d))

Y = np.vstack([Y_in, anomalies])
labels = np.hstack([np.zeros(n_in), np.ones(n_out)])

L_inv = np.linalg.inv(L)
z = (Y - mu) @ L_inv.T   # Pun si explicatia la formula pentru ca initial o intuiam dar nu stiam exact daca merge
                            # din Mahalanobis vine ca d^2 = (y - mu)^T * Sigma^-1 * (y - mu)
                            # inlocuiesc sigma^-1 cu L^-T * L^-1
                            # si vine (y - mu)^T * L^-T * L^-1 * (y - mu), care e basically z * z^T
scores = np.sqrt(z[:, 0]**2 + z[:, 1]**2)

threshold = np.quantile(scores, 1 - contamination)
pred = (scores > threshold).astype(int)

CM = sklearn.metrics.confusion_matrix(labels, pred)
TN, FP, FN, TP = CM[0][0], CM[0][1], CM[1][0], CM[1][1]
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BA = (TPR + TNR) / 2

print(CM)
print("Threshold:", threshold)
print("Balanced accuracy:", BA)
