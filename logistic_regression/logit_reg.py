import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
def read_from_txt(data_dir):
    table = pd.read_csv(data_dir).to_numpy()
    X = table[:, :-1]
    X = np.concatenate((np.ones(len(X)).reshape(-1, 1), X), axis = 1)
    y = table[:, -1].reshape(-1,1)
    return X, y


def sigmoid(z):
    return 1 / (1  + np.exp(-z))


def cost(X, y, theta):
    m = len(y)
    theta_colvec = theta.reshape(-1,1)
    y_pred = sigmoid(np.matmul(X, theta_colvec))
    J = (-np.matmul(y.T, np.log(y_pred)) - np.matmul((1-y).T, np.log(1 - y_pred)) ) / m
    return J.item()


def grad(X, y, theta):
    m = len(y)
    theta_colvec = theta.reshape(-1,1)
    y_pred = sigmoid(np.matmul(X, theta_colvec))   
    grad_vec = np.matmul(X.T, y_pred - y) / m
    return grad_vec.reshape(-1)


def learn_by_grad_desc(X, y, alpha, n_iter):
    J_values = np.zeros(n_iter)
    n_features = X.shape[1]
    theta = np.zeros(n_features).reshape(-1,1)
    for iter in range(n_iter):
        theta -= alpha * grad(X, y, theta)
        J_value = cost(X, y, theta)
        J_values[iter] = J_value
        # if iter % 100000 == 0:
        #     print("{}th iteration, J = {}".format(iter, J_value))
    return J_values, theta
X, y = read_from_txt('data1.csv')

# data_for_scatter = pd.read_csv('data1.csv')
# fig, ax = plt.subplots()
# groups = data_for_scatter.groupby("y")
# for name, group in groups:
#     ax.plot(group["x1"], group["x2"], marker="o", linestyle="", label=name)
# plt.legend()
# plt.show()

res = minimize(lambda t : cost(X, y, t), np.array([0,0,0]), method='BFGS', jac = lambda t : grad(X, y, t),
               options={'disp': True})
print(res)
# J_values, theta = learn_by_grad_desc(X, y, 0.001, 100000)
# print(theta)


# plt.plot(J_values)
# plt.show()  

# J_test = cost(X, y, np.array([0, 0, 0]))
# grad_test = grad(X, y, np.array([0, 0, 0]))
# print(J_test)
# print(grad_test)