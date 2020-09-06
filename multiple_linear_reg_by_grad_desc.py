import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_from_txt(data_dir):
    table = pd.read_csv(data_dir, header=None).to_numpy()
    X = table[:, :-1]
    X = np.concatenate((np.ones(len(X)).reshape(-1, 1), X), axis = 1)
    y = table[:, -1].reshape(-1,1)
    return X, y


def featureNormalize(X):
    """
    Normalizes the features in X
    returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    We should save mean and sigma for evaluation of new input data.
    """
    X_features = X[:, 1:]
    mu = np.mean(X_features, axis = 0)
    sigma = np.std(X_features, axis = 0)
    X_normalized = (X_features - mu) / sigma
    X_norm = np.concatenate((np.ones(len(X)).reshape(-1, 1),X_normalized), axis = 1)
    return X_norm, mu, sigma


def gradientDescent(X, y, alpha, n_iter):
    """
    Performs gradient descent to learn theta
    updates theta by taking num_iters gradient steps with learning rate alpha
    """
    n_features = X.shape[1]
    n_samples = len(y) # number of training examples
    J_history = np.zeros(n_iter)
    theta = np.zeros(n_features).reshape(-1,1)
    
    for iter in range(n_iter):
        Xt = np.transpose(X)
        delta = np.matmul(Xt, np.matmul(X, theta) - y)
        theta -= alpha / n_samples * delta

        J_now = computeCost(X, y, theta)
        J_history[iter] = J_now #Save the cost J in every iteration   

        if iter % 100 == 0:
            print(iter, 'th iteration, J = ', J_now)

    return theta, J_history


def computeCost(X, y, theta):
    """
    Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta
    as the parameter for linear regression to fit the data points in X and y
    """
    theta_vec = np.array(theta).reshape(-1, 1)
    m = len(y)  # number of training examples
    error_vector = np.matmul(X, theta_vec) - y
    J = 1 / (2*m) * np.matmul( error_vector.T , error_vector).item()
    return J


def Normal_eq(X, y):
    inv_XtX = np.linalg.inv(np.matmul(X.T, X)) 
    theta_opt = np.matmul(np.matmul(inv_XtX, X.T), y)
    return theta_opt

X, y = read_from_txt('data2.csv')

X_scaled, mean, sigma = featureNormalize(X)
theta, J = gradientDescent(X_scaled, y, 1.2, 500)
print('theta:',theta)
plt.plot( J)
plt.show()

print('theta_normal', Normal_eq(X_scaled, y)) #the same result. In practice, no need for scaling when using normal equation.