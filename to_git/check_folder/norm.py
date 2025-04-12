import torch
import numpy as np
import matplotlib.pyplot as plt

# Исходная функция с экспонентой (плохая для оптимизации в raw X)
def target_func(x):
    return torch.exp(x[:, 0]) + 0.5 * x[:, 1] ** 2

# Данные:
X_data = torch.tensor(np.random.uniform(-20, 20, (500, 2)), dtype=torch.float32)
y_target = target_func(X_data)

# Параметры оптимизации
steps = 1000
lr = 0.1

# Метод 1: Normalization (mean/std)
X_mean = X_data.mean(dim=0)
X_std = X_data.std(dim=0)

Z_norm = ((X_data - X_mean) / X_std).detach().clone().requires_grad_(True)
opt_norm = torch.optim.Adam([Z_norm], lr=lr)

norm_losses = []
for _ in range(steps):
    opt_norm.zero_grad()
    X_optim = Z_norm * X_std + X_mean
    loss = ((target_func(X_optim) - y_target) ** 2).mean()
    loss.backward()
    norm_losses.append(loss.item())
    opt_norm.step()

# Метод 2: tanh scaling
X_min = X_data.min(dim=0).values
X_max = X_data.max(dim=0).values

Z_tanh = torch.zeros_like(X_data, requires_grad=True)
opt_tanh = torch.optim.Adam([Z_tanh], lr=lr)

def z_to_x(Z):
    return 0.5 * (X_min + X_max) + 0.5 * (X_max - X_min) * torch.tanh(Z)

tanh_losses = []
for _ in range(steps):
    opt_tanh.zero_grad()
    X_optim = z_to_x(Z_tanh)
    loss = ((target_func(X_optim) - y_target) ** 2).mean()
    loss.backward()
    tanh_losses.append(loss.item())
    opt_tanh.step()

# Визуализация
plt.plot(norm_losses, label='mean/std normalization')
plt.plot(tanh_losses, label='tanh scaling (min/max)')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()