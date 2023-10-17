# -*- coding: utf-8 -*-
"""linear_regression.py

# Linear Regression Boston Data

In the following, a data analysis is performed to predict the median price of houses in the Boston suburbs in the 1980s. The data is stored in `features` and `target`. `features` is a Numpy array with 506 columns containing the different cases and 13 different features stored in the rows. The Numpy array `target` contains 506 target values for the prediction of the "median value of owner-occupied houses in 1000 $". (MEDV).
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from matplotlib import cm
import matplotlib.pylab as plt

#to avoid problems with matplot
from importlib import reload
plt=reload(plt)

import math
import numpy as np
import random

# Import data preprocessing and R2 calculation packages
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score

# Import the datasets
from sklearn import datasets

# Load dataset that we will use
boston = datasets.load_boston()

# Create features variable matrix
features = boston.data #Zeile -case und Spalten die jeweiligen features

# Collect the target variable (MEDV)
target = boston.target

"""## Design Matrix class

First, the `Design_Matrix` class is created, in which the data is preprocessed and a series of ones corresponding to an additional feature is added, to later obtain the correct matrix product theta in the normal equation. The `base_line` method simply adds the extra row, whereas the `updated_matrix` further process the feature matrix with user-defined transformations. With the `sklearn.preprocessing.FunctionTransformer` a square and logarithm transformer is applied, which returns the non-negative square root and the natural logarithm of one plus the input array, respectively. Then interaction terms between two variables are applied, starting with the first and the second feature and ascending to the last one. Both methods again return a numpy array, each of which is then used for the subclass `Theta` to perform the linear regression using the normal equation. Finally, the `plot_feature` method prints a desired feature against the target value, which is set by default, but can also be specified as an optional argument when the method is called. For this, the abbreviation of the feature from the following list should be entered:

0: CRIM - per capita crime rate by town \
1: ZN - proportion of residential land zoned for lots over 25,000 sq.ft. \
2: INDUS - proportion of non-retail business acres per town. \
3: CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)\
4: NOX - nitric oxides concentration (parts per 10 million)\
5: RM - average number of rooms per dwelling\
6: AGE - proportion of owner-occupied units built prior to 1940\
7: DIS - weighted distances to five Boston employment centres\
8: RAD - index of accessibility to radial highways\
9: TAX - full-value property-tax rate per $10,000\
10: PTRATIO - pupil-teacher ratio by town\
11: B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\
12: LSTAT - % lower status of the population
"""

class Design_Matrix:
    def __init__(self, features, target):
        """ Initialize a new Design Matrix instance."""
        self.features = features
        self.target = target

    def baseline_model(self):
        """ Adds a one vector row to the beginning of the feature matrix. """

        one_array = np.ones((self.features.shape[0], 1))
        baseline_model = np.hstack([one_array, self.features])

        return baseline_model

    def updated_matrix(self):
        """Applies square and log transformer, interaction terms between
        two variables and adds a one vector row to the beginning of the
        feature matrix."""

        #Square and log transformer
        transformer = FunctionTransformer(np.sqrt, np.log1p, validate = True)
        X_transformed = transformer.transform(self.features)

        #Interaction terms
        for i in range(12):
          interaction_append_function = lambda x: np.append(x, (x[:, i] * x[:, i+1])[:, None], 1)
          interaction_transformer = FunctionTransformer(func=interaction_append_function)
          X_transformed = interaction_transformer.fit_transform(X_transformed)

        #ones-vector
        one_array = np.ones((self.features.shape[0], 1))
        design_matrix = np.hstack([one_array, X_transformed])

        return design_matrix

    def feature_names(self, feature_name):
        """Converts the feature name into a number to use as an index."""

        if feature_name == 'CRIM':
          return 0
        elif feature_name == 'ZN':
          return 1
        elif feature_name == 'INDUS':
          return 2
        elif feature_name == 'CHAS':
          return 3
        elif feature_name == 'NOX':
          return 4
        elif feature_name == 'RM':
          return 5
        elif feature_name == 'AGE':
          return 6
        elif feature_name == 'DIS':
          return 7
        elif feature_name == 'RAD':
          return 8
        elif feature_name == 'TAX':
          return 9
        elif feature_name == 'PTRATIO':
          return 10
        elif feature_name == 'B':
          return 11
        elif feature_name == 'LSTAT':
          return 12
        else:
          print('Please enter a valid feature name')

    def plot_feature(self, feature_name = None, color='b', shape='.'):
        """ Draw the specified or default feature in a plot with the target
        values, in the provided shape and color. """

        feature_name = feature_name if feature_name is not None else 'LSTAT'

        feature_number = self.feature_names(feature_name)
        selected_feature = np.transpose(self.features)[feature_number]
        plt.plot(selected_feature, self.target, color+shape)
        plt.xlabel(feature_name)
        plt.ylabel('MEDV')
        plt.title("Boston data - " + feature_name + ' / ' + "MEDV")

#Creates a Design_matrix instance for testing
boston_features = Design_Matrix(features, target)

#Test if/how the data in the feature matrix changed
print(boston_features.baseline_model()[1])
print(boston_features.updated_matrix()[1])

#Test the plot and the feature names method
boston_features.plot_feature()
plt.show()

"""## Theta class

The class `Theta` extends the class `Design_Matrix` by solving the normal equation with the preprocessed data from the `Design_Matrix`. The normal equation is given below, where X is the m x (n+1) feature matrix with n features and m cases, y is a vector with m stored target values and theta is the resulting n+1 vector:

$\Theta = (X^T * X)^{-1} * X^T * y$

To solve this, the method `get_theta_vector` takes a feature matrix as an argument (baseline or best), `self.target` as y, and returns the vector with the resulting theta values. The method `predicted_values` then calculates the linear regression equation with the theta values from the method `get_theta_vector` and a given feature matrix. The linear regression equation has the form:

$y = \Theta_0 * \Theta_1 x_1 + ... + \Theta_n x_n$

The `coefficient_determination` method uses the built in function `sklearn.metrics.r2_score`, which is a coefficient of determination $R^2$ regression score function. $R^2$ is a measure of how well observed outcomes are reproduced by the model, based on the proportion of the total variation in outcomes explained by the model. The best possible score is 1.0. The method `coefficient_determination` returns a tuple, storing the $R^2$ values of the baseline and the best model, respectively.  

Finally, the `plot_true_pred` method plots a specified or default feature against the predicted as well as the true target values in different colours. The methods `plot_coefficient_determination_baseline` and `plot_coefficient_determination_best` plot the predicted target values against the actual ones. The line illustrates how far they deviate from each other. In both cases, for the $R^2$ value as well as in the plot, it can be clearly seen that the value could be improved by preprocessing the data with the `updated_matrix` method.

"""

class Theta(Design_Matrix):
    def __init__(self, features, target):
        """ Initialize a new Theta instance. """
        Design_Matrix.__init__(self, features, target)

    def get_theta_vector(self, matrix):
        """ Solves the normal equation with a given feature matrix and
        returns a vector with the stored theta values """

        X_product_inverse = np.linalg.inv(np.dot(np.transpose(matrix), matrix))
        X_product_inverse_transpose = np.dot(X_product_inverse, np.transpose(matrix))
        theta_vector = np.dot(X_product_inverse_transpose, self.target)

        return theta_vector

    def predicted_values(self, matrix):
        """ Calculates the linear regression equation with the calculated
        theta values and a given feature matrix  """

        predicted_values = np.dot(matrix, self.get_theta_vector(matrix))

        return predicted_values

    def coefficient_determination(self):
        """ Calculates the coefficient of determination of the baseline and
        the best model. It returns both values stored in a tuple."""

        coefficient_determination_baseline = r2_score(self.target, self.predicted_values(self.baseline_model()))
        coefficient_determination_best = r2_score(self.target, self.predicted_values(self.updated_matrix()))

        return coefficient_determination_baseline, coefficient_determination_best

    def plot_true_pred(self, feature_name = None):
        """ Plots a specified or the default feature against its predicted as
        well as true target values."""

        feature_name = feature_name if feature_name is not None else 'NOX'

        fig, ax = plt.subplots()
        ax.plot(np.transpose(self.features)[self.feature_names(feature_name)], self.predicted_values(self.updated_matrix()), 'b'+'.', label='Predicted Values')
        ax.plot(np.transpose(self.features)[self.feature_names(feature_name)], self.target, 'g'+'.', label='True Values')

        legend = ax.legend(loc='upper right', shadow=True)
        plt.xlabel(feature_name)
        plt.ylabel('Target Value / Predicted Value')
        plt.title(feature_name +": true and predicted value")

    def plot_coefficient_determination_baseline(self):
        """ Plots the predicted values based on the baseline_model
        matrix against the true target values."""

        plt.plot(self.predicted_values(self.baseline_model()), self.target, 'b'+'.')
        plt.plot([0,50], [0,50])
        plt.xlabel('Baseline Predicted Values')
        plt.ylabel('Target Values')

    def plot_coefficient_determination_best(self):
        """ Plots the predicted values based on the updated_matrix, with
        the preprocessed data, against the true target values."""

        plt.plot(self.predicted_values(self.updated_matrix()), self.target, 'b'+'.')
        plt.plot([0,50], [0,50])
        plt.xlabel('Best Predicted Values')
        plt.ylabel('Target Values')

#Creates a Theta instance for testing
boston_theta = Theta(features, target)

#The difference between the first and second value shows how much
#the model could be improved by preprocessing the data.
print(boston_theta.coefficient_determination())

#Test the plot feature against predicted/true target values
boston_theta.plot_true_pred('RM')
plt.show()

#Test the plot for the illustration of the coefficient of
#dertermination for the baseline model
boston_theta.plot_coefficient_determination_baseline()
plt.title("R2 illustration - Baseline")
plt.show()

#Test the plot for the illustration of the coefficient of
#dertermination for the best model
boston_theta.plot_coefficient_determination_best()
plt.title("R2 illustration - Best Model")
plt.show()