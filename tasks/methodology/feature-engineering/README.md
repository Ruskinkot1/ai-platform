---


---

<h1 id="feature-engineering">Feature Engineering</h1>
<p>Scikit-learn implementation of a Restricted Boltzmann Machine (RBM), Principal Component Analysis (PCA) for feature engineering in MNIST digits classification by logistic regression, sklearn multiayer perceptron (MLP) and support vector machine classifier.</p>
<h1 id="requirements">Requirements</h1>
<ul>
<li>Python 3.5</li>
<li>MLflow 1.0</li>
</ul>
<h1 id="dataset">Dataset</h1>
<p>MNIST handwritten digits dataset was used for implementation of feature engineering approaches. To make classification task more complicated, the dataset was nudged by moving pictures by different axes. This way we also multiplied the amount of data.</p>
<h2 id="unsupervised-learning-algorithms-for-feature-selection">Unsupervised learning algorithms for feature selection</h2>
<h3 id="restricted-boltzmann-machine-rbm">Restricted Boltzmann machine (RBM)</h3>
<p>A restricted Boltzmann machine (RBM) is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs.<br>
<img src="https://skymind.ai/images/wiki/reconstruction_RBM.png" alt="ÐšÐ°Ñ€Ñ‚Ð¸Ð½ÐºÐ¸ Ð¿Ð¾ Ð·Ð°Ð¿Ñ€Ð¾ÑÑƒ Restricted Boltzmann machine"><br>
Hidden and visible nodes are described by an energy function. This value also defines probability distribution according to Boltzmann equation.</p>
<p>Restricted Boltzmann machines are trained to maximize the product of probabilities assigned to some training set V (a matrix, each row of which is treated as a visible vector v). Through the training process, the RBM learns how to allocate its hidden nodes to specific features. However, the RBM doesn’t know what the nodes are in terms of naming, it just knows that one or some hidden nodes (features) have correlations to one or some of the visible nodes.</p>
<h2 id="component-extraction-with-rbm">Component extraction with RBM</h2>
<p>100 components were extracted by using RBM as features.</p>
<p><img src="https://sun9-50.userapi.com/c855736/v855736397/d13e0/U0mE5zE9Kfw.jpg" alt="enter image description here"></p>
<p>RBM-Classifier model was implemented on Sklearn Logistic regression, MLP and SVC.</p>
<p><strong>linear SVM using RBM features:</strong><br>
precision    recall  f1-score   support</p>
<pre><code>      0       1.00      0.98      0.99       174
      1       0.88      0.91      0.89       184
      2       0.96      0.96      0.96       166
      3       0.91      0.81      0.86       194
      4       0.96      0.93      0.95       186
      5       0.88      0.91      0.89       181
      6       0.98      0.98      0.98       207
      7       0.91      0.94      0.93       154
      8       0.75      0.88      0.81       182
      9       0.85      0.76      0.80       169
</code></pre>
<p>avg / total       0.91      0.91      0.91      1797</p>
<p><strong>linear SVM without RBM:</strong><br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.84      0.94      0.89       174
      1       0.66      0.70      0.68       184
      2       0.76      0.88      0.82       166
      3       0.80      0.75      0.77       194
      4       0.88      0.80      0.84       186
      5       0.75      0.83      0.79       181
      6       0.94      0.90      0.92       207
      7       0.83      0.90      0.86       154
      8       0.77      0.58      0.66       182
      9       0.73      0.70      0.72       169
</code></pre>
<p>avg / total                  0.80             0.80      	    0.80             1797</p>
<h2 id="reduction-of-components-using-pca">Reduction of components using PCA</h2>
<p>PCA decomposition of images was applied to reduce the number of features. The result improved implementation of an MLP (layer size = 20, 8) algorithm.</p>
<p>MLP using PCA features:<br>
precision    recall  		f1-score   support</p>
<pre><code>      0       0.98      0.99      0.99       174
      1       0.95      0.96      0.95       184
      2       0.98      0.99      0.98       166
      3       0.96      0.98      0.97       194
      4       0.97      0.98      0.98       186
      5       0.97      0.96      0.96       181
      6       1.00      0.98      0.99       207
      7       0.97      0.99      0.98       154
      8       0.97      0.93      0.95       182
      9       0.96      0.96      0.96       169
</code></pre>
<p>avg / total       0.97      0.97      0.97      1797</p>
<p>MLP:<br>
precision    recall  f1-score   support</p>
<pre><code>      0       0.95      0.97      0.96       174
      1       0.80      0.78      0.79       184
      2       0.85      0.93      0.89       166
      3       0.86      0.84      0.85       194
      4       0.95      0.90      0.92       186
      5       0.65      0.61      0.63       181
      6       0.96      0.93      0.95       207
      7       0.87      0.90      0.88       154
      8       0.70      0.71      0.70       182
      9       0.71      0.74      0.73       169
</code></pre>
<p>avg / total       0.83      0.83      0.83      1797</p>
<h2 id="final-model">Final model</h2>
<p>The sklearn PCA-MLP pipeline was chosen as optimal model and tuned by hyperparameters. I used MLflow package to build final PCA-MLP model.</p>

