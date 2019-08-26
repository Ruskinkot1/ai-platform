#  Representation Learning

Scikit-learn implementation of a Restricted Boltzmann Machine (RBM), Principal Component Analysis (PCA) for feature engineering in MNIST digits classification by logistic regression, sklearn multiayer perceptron (MLP) and support vector machine classifier.

# Requirements

 - Python 3.5 
 - MLflow 1.0

# Dataset
MNIST handwritten digits dataset was used for implementation of feature engineering approaches. To make classification task more complicated, the dataset was nudged by moving pictures by different axes. This way we also multiplied the amount of data.

## Unsupervised learning algorithms for feature selection
### Restricted Boltzmann machine (RBM)
A restricted Boltzmann machine (RBM) is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs.
![ÐšÐ°Ñ€Ñ‚Ð¸Ð½ÐºÐ¸ Ð¿Ð¾ Ð·Ð°Ð¿Ñ€Ð¾ÑÑƒ Restricted Boltzmann machine](https://skymind.ai/images/wiki/reconstruction_RBM.png)
Hidden and visible nodes are described by an energy function. This value also defines probability distribution according to Boltzmann equation. 
![ÐšÐ°Ñ€Ñ‚Ð¸Ð½ÐºÐ¸ Ð¿Ð¾ Ð·Ð°Ð¿Ñ€Ð¾ÑÑƒ rbm energy function](data:image/png;base64,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)

Restricted Boltzmann machines are trained to maximize the product of probabilities assigned to some training set V (a matrix, each row of which is treated as a visible vector v). Through the training process, the RBM learns how to allocate its hidden nodes to specific features. However, the RBM doesn't know what the nodes are in terms of naming, it just knows that one or some hidden nodes (features) have correlations to one or some of the visible nodes.

## Component extraction with RBM

100 components were extracted by using RBM as features.


![enter image description here](https://sun9-50.userapi.com/c855736/v855736397/d13e0/U0mE5zE9Kfw.jpg)

 RBM-Classifier model was implemented on Sklearn Logistic regression, MLP and SVC.

**linear SVM using RBM features:**
             precision    recall  f1-score   support

          0       1.00      0.98      0.99       174
          1       0.88      0.91      0.89       184
          2       0.96      0.96      0.96       166
          3       0.91      0.81      0.86       194
          4       0.96      0.93      0.95       186
          5       0.88      0.91      0.89       181
          6       0.98      0.98      0.98       207
          7       0.91      0.94      0.93       154
          8       0.75      0.88      0.81       182
          9       0.85      0.76      0.80       169

avg / total       0.91      0.91      0.91      1797


**linear SVM without RBM:**
             precision    recall  f1-score   support

          0       0.84      0.94      0.89       174
          1       0.66      0.70      0.68       184
          2       0.76      0.88      0.82       166
          3       0.80      0.75      0.77       194
          4       0.88      0.80      0.84       186
          5       0.75      0.83      0.79       181
          6       0.94      0.90      0.92       207
          7       0.83      0.90      0.86       154
          8       0.77      0.58      0.66       182
          9       0.73      0.70      0.72       169

avg / total                  0.80             0.80      	    0.80             1797

## Reduction of components using PCA
PCA decomposition of images was applied to reduce the number of features. The result improved implementation of an MLP (layer size = 20, 8) algorithm. 

MLP using PCA features:
             precision    recall  		f1-score   support

          0       0.98      0.99      0.99       174
          1       0.95      0.96      0.95       184
          2       0.98      0.99      0.98       166
          3       0.96      0.98      0.97       194
          4       0.97      0.98      0.98       186
          5       0.97      0.96      0.96       181
          6       1.00      0.98      0.99       207
          7       0.97      0.99      0.98       154
          8       0.97      0.93      0.95       182
          9       0.96      0.96      0.96       169

avg / total       0.97      0.97      0.97      1797

MLP:
             precision    recall  f1-score   support

          0       0.95      0.97      0.96       174
          1       0.80      0.78      0.79       184
          2       0.85      0.93      0.89       166
          3       0.86      0.84      0.85       194
          4       0.95      0.90      0.92       186
          5       0.65      0.61      0.63       181
          6       0.96      0.93      0.95       207
          7       0.87      0.90      0.88       154
          8       0.70      0.71      0.70       182
          9       0.71      0.74      0.73       169

avg / total       0.83      0.83      0.83      1797

## Final model 

The sklearn PCA-MLP pipeline was chosen as optimal model and tuned by hyperparameters. I used MLflow package to build final PCA-MLP model.

