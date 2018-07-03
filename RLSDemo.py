import numpy as np
import matplotlib.pyplot as plt
import random

# x = np.random.randint(0, 15, 50)

BPSK = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
x = []  ##训练信号
for i in range(600):
    x.append(BPSK[random.randint(0, 3)])
x = np.concatenate([[0], x])
# x = np.array(x, dtype=float)  ##发送序列 第一个0不属于发送序列，为算法的初始化条件
N = len(x) - 1      ##发送序列长度，
M = 5               ##滤波器抽头个数
dal = 1
lam = 0.05
print(x)

y = x*2
y[0] =0
x0 = np.zeros(M, dtype=complex)
x1 = np.concatenate([x0, x])
xMatric = np.mat(np.zeros((len(x), M)), dtype=complex)   ##输入信号矩阵
A = np.mat(np.zeros((len(x), M)), dtype=complex)   ##权值矩阵
C = dal*np.eye(M, dtype=complex)       ##逆矩阵
u = np.zeros(len(x), dtype=complex)
e = np.zeros(len(x), dtype=complex)    ##误差矩阵
print(y)

xMatric[0] = x1[0:M]
for i in range(len(x)):
    xMatric[i] = x1[i+M:i:-1]
print(xMatric)

for i in range(1, len(x)):
    e[i] = y[i] - A[i-1]*xMatric[i].T
    u[i] = xMatric[i]*C*xMatric[i].T
    g = C*xMatric[i].T/(lam + u[i])
    # print(g)
    A[i] = A[i-1] + g.T*e[i]
    C = (C - g*xMatric[i]*C)/lam

print(A)
print(e)

e2 = np.zeros(len(x), dtype=complex)
for i in range(len(x)):
    e2[i] = (y[i] - A[N]*xMatric[i].T)
print(type(e2))
print(type(xMatric))

plt.plot(10*np.log10(abs(e[1:])))

eSum = np.zeros(len(x), dtype=complex)
for i in range(len(x)):
    eSum[i] = (y - A[i] * xMatric.T).sum()

fig, ax = plt.subplots(1)
ax.plot(10 * np.log10(pow(abs(eSum[1:]), 2) / pow(abs(y).sum(), 2)))
ax.set(title="Mse curve", xlabel="Iterations", ylabel="MSE(dB)")
plt.show()
