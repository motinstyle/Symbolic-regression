import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root, fsolve

np.random.seed(42)

# Истинная модель
def model(x):
    return np.exp(-1*x**2)

def dmodel(x):
    return -2*x*np.exp(-1*x**2)

# Создаём выборку с шумом
def create_dataset():
    X_data = np.linspace(-5, 5, 100)
    X_data = X_data + np.random.normal(0, 2, 100)
    y_data = model(X_data)
    y_noise = np.random.normal(0, 2, 100)
    y_noise_data = y_data + y_noise
    return X_data, y_data, y_noise_data

# Целевая функция: расстояние^2 до кривой y = model(x)
def objective(x_pred, x0, y0, model_func):
    # x_pred — массив длины 1, так как minimize передаёт его как вектор
    x = x_pred[0]
    if x0 - x == 0:
        x0 = x0 + 0.0001
    return np.sqrt(np.sum((x - x0)**2) + (model_func(x) - y0)**2)

def objective_root(x_pred, x0, y0, model_func, dmodel_func):
    if x0 - x_pred == 0:
        x0 = x0 + 0.0001
    return np.abs((x0 - x_pred)/(y0 - model_func(x_pred)) - dmodel_func(x_pred))

def objective_root_2(x_pred, x0, y0, model_func, dmodel_func):
    if x0 - x_pred == 0:
        x0 = x0 + 0.0001
    return (y0 - model_func(x_pred))/(x0 - x_pred) + 1/dmodel_func(x_pred)

def gradient_objective_root(x_pred, x0, y0, model_func, dmodel_func):
    if x0 - x_pred == 0:
        x0 = x0 + 0.0001
    return -2*(x0 - x_pred) + 2*(y0 - model_func(x_pred))*dmodel_func(x_pred)

def perpendicular_condition(x, x0, y0):
    f_x = model(x)
    df_x = dmodel(x)
    if x0 - x == 0:
        x0 = x0 + 0.0001    
    slope = (f_x - y0) / (x - x0)
    return (slope * df_x + 1)**2

# Визуализация результатов
def visualization_plot(X_data, y_noise_data, X_pred, model_func):
    plt.figure(figsize=(12, 8))
    
    # Гладкая кривая модели
    x_curve = np.linspace(min(X_data), max(X_data), 1000)
    y_curve = model_func(x_curve)
    plt.plot(x_curve, y_curve, 'b-', linewidth=2, label='Model function')

    # Предсказанные значения
    y_pred = model_func(X_pred)
    
    # Шумные наблюдения
    plt.scatter(X_data, y_noise_data, color='green', s=40, alpha=0.7, label='Noisy data points')
    
    # Предсказания (точки на кривой)
    plt.scatter(X_pred, y_pred, color='red', s=40, alpha=0.7, label='Predicted points')
    
    # Соединяем отрезками оригинальные точки и их проекции на кривую
    for i in range(len(X_data)):
        plt.plot([X_data[i], X_pred[i]], [y_noise_data[i], y_pred[i]], 'r-', alpha=0.5)
    
    plt.axis('equal')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.title('Closest Points on the Curve to Noisy Observations', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Главный блок
if __name__ == "__main__":
    X_data, _, y_noise_data = create_dataset()

    # # Нормализация данных
    # data = np.array([X_data, y_noise_data])
    # data = data - np.mean(data)
    # data = data / np.std(data)
    # X_data, y_noise_data = data

    X_pred = np.zeros(len(X_data))
    
    for i in range(len(X_data)):
        res = minimize(
            objective,
            # perpendicular_condition,
            # objective_root,
            # objective_root_2,
            x0=[X_data[i]],  # Начальная точка — текущий x_data
            args=(X_data[i], y_noise_data[i], model),
            method='Nelder-Mead',
            options={'maxiter': 100}
        )

        # res = root(
        #     objective_root_2,
        #     x0=[X_data[i]+0.0001],  # Начальная точка — текущий x_data
        #     args=(X_data[i], y_noise_data[i], model, dmodel),
        # )

        # res = fsolve(objective_root, x0=X_data[i], args=(X_data[i], y_noise_data[i], model, dmodel))
        # res = fsolve(objective_root_2, x0=X_data[i], args=(X_data[i], y_noise_data[i], model, dmodel))

        # sol = root(perpendicular_condition, x0=X_data[i], args=(X_data[i], y_noise_data[i]))
        print(res)
        X_pred[i] = res.x[0] if res.fun < 10 else X_data[i]
        # X_pred[i] = res.x
        # # Метод градиентного спуска
        # X_pred[i] = X_data[i]
        # for j in range(100):
        #     X_pred[i] = X_pred[i] - 0.001*gradient_objective_root(X_pred[i], X_data[i], y_noise_data[i], model)

    
    # Визуализация
    visualization_plot(X_data, y_noise_data, X_pred, model)
    print("Optimization complete. Visualization displayed.")
