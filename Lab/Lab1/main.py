import numpy as np
import sklearn
from pyod.models import knn
import pyod.utils.data
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = pyod.utils.data.generate_data(n_train=400, n_test=100, n_features=2,
                                                                 contamination=0.1)
# outliner_indexes = np.where(Y_train == 1.0)
# inliner_indexes = np.where(Y_train == 0.0)
# outliners = X_train[outliner_indexes]
# inliners = X_train[inliner_indexes]
#
# plt.scatter(inliners[:, 0], inliners[:, 1])
# plt.scatter(outliners[:, 0], outliners[:, 1])
# plt.show()

k_neigh = knn.KNN(contamination=0.1)
knn.KNN.fit(k_neigh, X_train)
pred_train = knn.KNN.predict(k_neigh, X_train)
pred_test = knn.KNN.predict(k_neigh, X_test)
CM = sklearn.metrics.confusion_matrix(Y_train, pred_train)

print(CM)
TP, FN, FP, TN = CM[0][0], CM[0][1], CM[1][0], CM[1][1]
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BA = (TPR + TNR) / 2
print(BA)