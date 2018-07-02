import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0, 15, 50)
x = np.concatenate([[0], x])
# x = np.array(x, dtype=float)  ##发送序列 第一个0不属于发送序列，为算法的初始化条件
N = len(x) - 1      ##发送序列长度，
M = 5               ##滤波器抽头个数
dal = 1
lam = 0.05
print(x)

y = x*2
y[0] =0
x0 = np.zeros(M, dtype=float)
x1 = np.concatenate([x0, x])
X = np.mat(np.zeros((len(x), M)))   ##输入信号矩阵
A = np.mat(np.zeros((len(x), M)))   ##权值矩阵
C = dal*np.eye(M)       ##逆矩阵
u = np.zeros(len(x))
e = np.zeros(len(x))    ##误差矩阵
print(y)

X[0] = x1[0:M]
for i in range(len(x)):
    X[i] = x1[i+M:i:-1]
print(X)

for i in range(1, len(x)):
    e[i] = y[i] - A[i-1]*X[i].T
    u[i] = X[i]*C*X[i].T
    g = C*X[i].T/(lam + u[i])
    # print(g)
    A[i] = A[i-1] + g.T*e[i]
    C = (C - g*X[i]*C)/lam

print(A)
print(e)

e2 = np.zeros(len(x))
for i in range(len(x)):
    e2[i] = (y[i] - A[N]*X[i].T)
print(e2)

plt.plot(10*np.log10(abs(e[1:])))
plt.show()