import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_from_txt(data_dir):
    table = pd.read_csv(data_dir, header=None).to_numpy()
    X = table[:, :-1]
    X = np.concatenate((np.ones(len(X)).reshape(-1, 1), X), axis = 1)
    y = table[:, -1].reshape(-1,1)
    return X, y


def gradientDescent(X, y, alpha, n_iter):
    """
    Performs gradient descent to learn theta
    updates theta by taking num_iters gradient steps with learning rate alpha
    """

    m = len(y) # number of training examples
    J_history = np.zeros(n_iter)
    theta = np.zeros(2).reshape(-1,1)
    
    for iter in range(n_iter):
        Xt = np.transpose(X)
        delta = np.matmul(Xt, np.matmul(X, theta) - y)
        theta -= alpha / m * delta

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
    n_samples = len(y)  # number of training examples
    error_vector = np.matmul(X, theta_vec) - y
    J = 1 / (2 * n_samples) * np.matmul( error_vector.T , error_vector).item()
    return J


X, y = read_from_txt('data1.csv')

theta, J = gradientDescent(X, y, 0.01, 2500)
print('theta:',theta)
plt.plot( J)
plt.show()

