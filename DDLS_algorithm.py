import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import csv
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


class RVFLNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化随机权重和偏差
        self.input_weights = np.random.rand(input_size, hidden_size)
        self.bias = np.random.rand(hidden_size)

        # 初始化输出层权重
        self.output_weights = None

    def train(self, X, y):
        # 计算隐层输出
        hidden_output = np.dot(X, self.input_weights) + self.bias
        hidden_output = np.tanh(hidden_output)
        hidden_output = np.hstack((X, hidden_output))
        a = np.ones(X.shape[0])
        # 添加偏置项
        hidden_output = np.column_stack((hidden_output, np.ones(X.shape[0])))#.shape[0]读取矩阵行数

        # 使用最小二乘法学习输出层权重
        self.output_weights = np.dot(np.linalg.pinv(hidden_output), y)

    def predict(self, X):
        # 计算隐层输出
        hidden_output = np.dot(X, self.input_weights) + self.bias
        a=np.ones(X.shape[0])
        # 添加偏置项
        hidden_output = np.column_stack((hidden_output, np.ones(X.shape[0])))

        # 预测输出
        y_pred = np.dot(hidden_output, self.output_weights)

        return y_pred


# 示例用法
if __name__ == "__main__":
    #导入参数----------------------------------------------------------------------

    # 生成随机训练数据------------------------------------------------------
    num_training_set=11
    noise_level=0.001
    L = np.random.uniform(50E-9, 300E-9, size=(1, num_training_set))
    P = np.random.uniform(2, 30, size=(1, num_training_set))
    Dt = Kb * T / (3 * np.pi * eta * L) * (np.log(P) - 0.312 + 0.565 / P - 0.1 / P / P)
    Dr = 3 * Kb * T / (3 * np.pi * eta * (L ** 3)) * (np.log(P) - 0.662 + 0.917 / P - 0.05 / P / P)
    qqdt=q*q*Dt
    Dr6=6*Dr
    gamma2=(qqdt+Dr6).reshape(-1, 1)

    print((gamma2))
    # gamma2 = np.linspace(1, 101, num_training_set).reshape(-1, 1)  # -q^2*Dt-6Dr
    noise=noise_level*np.random.normal(size=(num_training_set,input_size))
    g_vh=np.exp(-np.dot(gamma2,tao))+noise
    # g_vh = np.exp(-np.dot(gamma2, tao))
    X_train=g_vh
    y_train=gamma2
    # 训练模型-----------------------------------------------------------
    model.train(X_train, y_train)

    # 生成测试数据-----------------------------------------------------------
    l_test = 80E-9
    p_test = 4
    dt_test = Kb * T / (3 * np.pi * eta * l_test) * (np.log(p_test) + 0.312 + 0.565 / p_test - 0.1 / p_test / p_test)
    dr_test = 3 * Kb * T / (3 * np.pi * eta * (l_test ** 3)) * (
                np.log(p_test) - 0.662 + 0.917 / p_test - 0.05 / p_test / p_test)
    gamma_test = q * q * dt_test + 6 * dr_test
    #for gamma_test in range(100,501,50):
    #gamma_test=500
    X_test = np.exp(-(gamma_test) * tao)+noise_level*np.random.normal(size=(1,input_size))
    y_test = [gamma_test]
    # 进行预测---------------------------------------------------------------------
    y_pred = model.predict(X_test)
    #popt , pcov = curve_fit(func, tao , X_test)
    # 计算均方误差------------------------------------------------------------
    #mse = mean_squared_error(y_test, y_pred)
    mse=(y_test-y_pred)/y_test
    print(f"测试参数：{gamma_test} 预测结果：{y_pred}相对误差: {mse}")
    #print(f"测试参数：{gamma_test} 拟合结果：{y_fit}相对误差: {(y_test-y_fit)/y_test}")
    #plt.scatter(X_train, y_train, color='b', label='origion')
    #plt.scatter(X_test, y_pred, color='g', label='predict')
    plt.xscale('log')
    plt.scatter(tao,X_test,  color='g', label='origin')
    plt.scatter(tao, np.exp(-y_pred*tao), color='r', label='Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()