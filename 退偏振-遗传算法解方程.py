# import numpy as np
# import matplotlib.pyplot as plt
# from sko.GA import GA
#
# # 定义要解的非线性方程
# def equations(vars):
#     L, D = vars
#     eq1 = D_t - (Kb * T / (3 * np.pi * eta * L) * (np.log(L/D) + 0.565 * D/L + 0.312 - 0.1 * D**2/L**2))
#     eq2 = D_r - (3 * Kb * T / (np.pi * eta *( L**3)) * (np.log(L/D) + 0.917 * D/L - 0.662 - 0.05 * D**2/L**2))
#     return (eq1/D_t)**2+(eq2/D_r)**2
#
#
# Kb = 1.38E-23
# T = 273.15 + 25.9
# eta = 0.873E-3
# L = 80E-9
# D = 20E-9
# P=L/D
# noise_factor = 0.01
#
# D_t = Kb * T / (3 * np.pi * eta * L) * (np.log(P) + 0.312 + 0.565 / P - 0.1 / P / P)* (1 + noise_factor*np.random.randn())
# D_r = 3 * Kb * T / ( np.pi * eta * (L ** 3)) * (np.log(P) - 0.662 + 0.917 / P - 0.05 / P / P)* (1 + noise_factor*np.random.randn())
#
#
# # 创建遗传算法对象
# ga = GA(func=equations, n_dim=2, size_pop=1000 , max_iter=50, lb=[10E-9,1E-9] , ub=[500E-9,200E-9] , precision=1e-10)
#
# # 运行遗传算法
# best_x, best_y = ga.run()
# print(equations([L,D]))
# # 打印结果
# print('Best x:', best_x)
# print('Value of the equation at the best x:', best_y)
#
# # 绘制收敛过程图
# Y_history = ga.all_history_Y
# plt.plot(Y_history, '.', color='red')
# plt.title('Convergence Process')
# plt.xlabel('Iteration')
# plt.ylabel('Equation Value')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA


# 定义要解的非线性方程
def equations(vars):
    L, D = vars
    eq1 = D_t - (Kb * T / (3 * np.pi * eta * L) * (np.log(L / D) + 0.565 * D / L + 0.312 - 0.1 * D ** 2 / L ** 2))
    eq2 = D_r - (3 * Kb * T / (np.pi * eta * (L ** 3)) * (
                np.log(L / D) + 0.917 * D / L - 0.662 - 0.05 * D ** 2 / L ** 2))
    return (eq1 / D_t) ** 2 + (eq2 / D_r) ** 2


# 定义常数和初始值
Kb = 1.38E-23
T = 273.15 + 25.9
eta = 0.873E-3
noise_factor = 0.1

# 生成一系列随机的L和D值
num_samples = 50
L_values = np.random.uniform(low=10E-9, high=500E-9, size=num_samples)
D_values = np.random.uniform(low=1E-9, high=200E-9, size=num_samples)

results = []

# 使用遗传算法计算并保存结果
for i in range(num_samples):
    L = L_values[i]
    D = D_values[i]

    D_t = Kb * T / (3 * np.pi * eta * L) * (np.log(L / D) + 0.312 + 0.565 / (L / D) - 0.1 / (L / D) ** 2) * (
                1 + noise_factor * np.random.randn())
    D_r = 3 * Kb * T / (np.pi * eta * (L ** 3)) * (np.log(L / D) - 0.662 + 0.917 / (L / D) - 0.05 / (L / D) ** 2) * (
                1 + noise_factor * np.random.randn())

    ga = GA(func=equations, n_dim=2, size_pop=1000, max_iter=50, lb=[10E-9, 1E-9], ub=  [500E-9, 200E-9], precision=1e-10)
    best_x, best_y = ga.run()

    results.append((L, D, best_x[0], best_x[1], best_y))  # 使用元组而不是列表

# 转换为结构化数组
results = np.array(results, dtype=[('L', float), ('D', float), ('best_L', float), ('best_D', float), ('error', float)])

# 提取结果数据
L_values, D_values, best_L_values, best_D_values, errors = results['L'], results['D'], results['best_L'], results[
    'best_D'], results['error']

# 绘制散点图
plt.scatter(L_values, D_values, c='blue', label='Random Samples')
plt.scatter(best_L_values, best_D_values, c='red', marker='x', label='Optimized Results')
plt.title('Random Samples and Optimized Results')
plt.xlabel('L Values')
plt.ylabel('D Values')
plt.legend()
plt.show()
# 计算平均相对误差
relative_errors_L = np.abs(best_L_values - L_values) / L_values
relative_errors_D = np.abs(best_D_values - D_values) / D_values

average_relative_error_L = np.mean(relative_errors_L)
average_relative_error_D = np.mean(relative_errors_D)

# 打印结果
print(f'Average Relative Error for L: {average_relative_error_L:.4f}')
print(f'Average Relative Error for D: {average_relative_error_D:.4f}')

