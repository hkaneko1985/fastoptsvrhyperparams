# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
Fast optimization of SVR hyperparameters with Gaussian kernel
"""

import time

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from sklearn import model_selection, svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV

# Settings
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # Candidates of C
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # Candidates of epsilon
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # Candidates of gamma
fold_number = 5  # "fold_number"-fold cross-validation
number_of_training_samples = 1000
number_of_test_samples = 1000

# Generate samples for demonstration
X, y = datasets.make_regression(n_samples=number_of_training_samples + number_of_test_samples, n_features=100,
                                n_informative=100, noise=100, random_state=0)

# Divide samples into training samples and test samples
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=number_of_test_samples, random_state=0)

# Standarize X and y
autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
autoscaled_ytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)
autoscaled_Xtest = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)

# Measure time in hyperparameter optimization
start_time = time.time()

# Optimize gamma by maximizing variance in Gram matrix
numpy_autoscaled_Xtrain = np.array(autoscaled_Xtrain)
variance_of_gram_matrix = list()
for svr_gamma in svr_gammas:
    gram_matrix = np.exp(
        -svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(axis=2))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]

# Optimize epsilon with cross-validation
svr_model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number)
svr_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
optimal_svr_epsilon = svr_model_in_cv.best_params_['epsilon']

# Optimize C with cross-validation
svr_model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number)
svr_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
optimal_svr_c = svr_model_in_cv.best_params_['C']

# Optimize gamma with cross-validation (optional)
svr_model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number)
svr_model_in_cv.fit(autoscaled_Xtrain, autoscaled_ytrain)
optimal_svr_gamma = svr_model_in_cv.best_params_['gamma']

# Check time in hyperparameter optimization
elapsed_time = time.time() - start_time
print("Elapsed time in hyperparameter optimization: {0} [sec]".format(elapsed_time))

# Check optimized hyperparameters
print("C: {0}, Epsion: {1}, Gamma: {2}".format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))

# Construct SVR model
regression_model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)
regression_model.fit(autoscaled_Xtrain, autoscaled_ytrain)

# Calculate y of training dataset
calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_Xtrain))
calculated_ytrain = calculated_ytrain * ytrain.std(ddof=1) + ytrain.mean()
# r2, RMSE, MAE
print("r2: {0}".format(float(1 - sum((ytrain - calculated_ytrain) ** 2) / sum((ytrain - ytrain.mean()) ** 2))))
print("RMSE: {0}".format(float((sum((ytrain - calculated_ytrain) ** 2) / len(ytrain)) ** 0.5)))
print("MAE: {0}".format(float(sum(abs(ytrain - calculated_ytrain)) / len(ytrain))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, calculated_ytrain)
YMax = np.max(np.array([np.array(ytrain), calculated_ytrain]))
YMin = np.min(np.array([np.array(ytrain), calculated_ytrain]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Calculated Y")
plt.show()

# Estimate y in cross-validation
estimated_y_in_cv = np.ndarray.flatten(
    model_selection.cross_val_predict(regression_model, autoscaled_Xtrain, autoscaled_ytrain, cv=fold_number))
estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()
# r2cv, RMSEcv, MAEcv
print("r2cv: {0}".format(float(1 - sum((ytrain - estimated_y_in_cv) ** 2) / sum((ytrain - ytrain.mean()) ** 2))))
print("RMSEcv: {0}".format(float((sum((ytrain - estimated_y_in_cv) ** 2) / len(ytrain)) ** 0.5)))
print("MAEcv: {0}".format(float(sum(abs(ytrain - estimated_y_in_cv)) / len(ytrain))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytrain, estimated_y_in_cv)
YMax = np.max(np.array([np.array(ytrain), estimated_y_in_cv]))
YMin = np.min(np.array([np.array(ytrain), estimated_y_in_cv]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y in CV")
plt.show()

# Estimate y of test dataset
predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_Xtest))
predicted_ytest = predicted_ytest * ytrain.std(ddof=1) + ytrain.mean()
# r2p, RMSEp, MAEp
print("r2p: {0}".format(float(1 - sum((ytest - predicted_ytest) ** 2) / sum((ytest - ytest.mean()) ** 2))))
print("RMSEp: {0}".format(float((sum((ytest - predicted_ytest) ** 2) / len(ytest)) ** 0.5)))
print("MAEp: {0}".format(float(sum(abs(ytest - predicted_ytest)) / len(ytest))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytest, predicted_ytest)
YMax = np.max(np.array([np.array(ytest), predicted_ytest]))
YMin = np.min(np.array([np.array(ytest), predicted_ytest]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y")
plt.show()
