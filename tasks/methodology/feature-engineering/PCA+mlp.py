from __future__ import print_function

import mlflow
import mlflow.sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.ndimage import convolve
from sklearn import  datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

if __name__ == "__main__":
  def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

  digits = datasets.load_digits()
  X = np.asarray(digits.data, 'float32')
  X, Y = nudge_dataset(X, digits.target)
  X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)
  X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
    

  mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 16), random_state=1)
  pca = PCA(n_components=36)
  pca_features_mlp_classifier = Pipeline(steps=[('pca', pca), ('mlp', mlp)])
  pca_features_mlp_classifier.fit(X_train, Y_train)
  Y_pred = pca_features_mlp_classifier.predict(X_test)
  report = metrics.classification_report(Y_test, Y_pred)
  print("MLP using PCA features:\n%s\n" % (report))
  score=metrics.accuracy_score(Y_test, Y_pred)
  mlflow.log_metric("score", score)
  mlflow.sklearn.log_model(pca_features_mlp_classifier, "model")
  print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

