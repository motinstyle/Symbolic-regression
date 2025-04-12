import pandas as pd
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def func(params, x):
    return params[0]*np.sqrt(1 - params[1]*x)

def dfunc(params, x):
    return -params[0]*params[1]/(2*np.sqrt(1 - params[1]*x))
# X = X.reshape(-1, 1)

def loss(params, x, y):
    return np.sum((func(params, x) - y)**2)

def dloss(params, x, y):
    # Вычисляем градиент функции потерь для каждого параметра
    grad_a = 2 * np.sum((func(params, x) - y) * np.sqrt(1 - params[1]*x))
    grad_b = 2 * np.sum((func(params, x) - y) * dfunc(params, x))
    
    # Возвращаем массив градиентов по каждому параметру
    return np.array([grad_a, grad_b])

def main():

    data = pd.read_csv("speed_force.csv")

    data.columns = ["x", "output"]

    data = data.to_numpy()

    X = data[:, 0]
    y = data[:, 1]

    a = 43.00259255308226
    b = 0.00778695082200213

    params = np.array([a, b])
    
    res = optimize.minimize(loss, params, args=(X, y), method="Nelder-Mead")
    
    print(res)

    plt.scatter(X, y)
    plt.plot(X, func(params, X), color="green")
    plt.plot(X, func(res.x, X), color="red")
    plt.show()

if __name__ == "__main__":
    main()
    
    
    