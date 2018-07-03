import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal
from filter import *
from RLS import *

BPSK = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
Nc = 100     ##子载波长度
fb = 500    ##基础频率
Nb = 4000    ##50000
iterations = 30

x1 = np.zeros((iterations, Nc), dtype=complex)     ##训练信号
x2 = np.zeros((iterations, Nc), dtype=complex)     ##需要消除时的发送信号
x3 = np.zeros((iterations, Nc), dtype=complex)        ##远端有用信号

# x = np.zeros(Nc)
# y = np.zeros(Nc)

for i in range(iterations):
    for j in range(Nc):
        x1[i][j] = BPSK[random.randint(0, 3)]
        x2[i][j] = BPSK[random.randint(0, 3)]
        x3[i][j] = BPSK[random.randint(0, 3)]

n = np.arange(1, Nb+1, 1)
t = n/Nb
fc = 1    ##子载波间隔
y1 = np.zeros((iterations, Nb))   ##训练序列
y2 = np.zeros((iterations, Nb))   ##发送序列
y3 = np.zeros((iterations, Nb))   ##远端有用信号
signal1 = np.zeros((iterations, Nb))
signal2 = np.zeros((iterations, Nb))
Y1 = np.zeros((iterations, 2 * Nc - 1), dtype=complex)
Y2 = np.zeros((iterations, 2 * Nc - 1), dtype=complex)
x1Conv = np.zeros((iterations, 2 * Nc - 1), dtype=complex)
x2Conv = np.zeros((iterations, 2 * Nc - 1), dtype=complex)

for col in range(iterations):
    # print(y1[col])
    # print(x1[col][0])
    for i in range(Nc):
        y1[col] = y1[col] + x1[col][i].real*np.cos(2*np.pi*(fb + i+1)*fc*t)*0.05 - x1[col][i].imag*np.sin(2*np.pi*(fb + i+1)*fc*t)*0.05
        y2[col] = y2[col] + x2[col][i].real*np.cos(2*np.pi*(fb + i+1)*fc*t)*0.05 - x2[col][i].imag*np.sin(2*np.pi*(fb + i+1)*fc*t)*0.05

    x = y1[col]
    y = y2[col]
    k = [10, 0.5, 0.1]

    noise1 = [0.01*random.random() for i in range(Nb)]

    signal1[col] = (k[0]*x - k[1]*pow(x, 2) - k[2]*pow(x, 3) + noise1)/10000

    noise2 = [0.01*random.random() for i in range(Nb)]
    signal2[col] = (k[0]*y - k[1]*pow(y, 2) - k[2]*pow(y, 3) + noise2)/10000

    real1 = np.zeros(2 * Nc - 1)
    imag1 = np.zeros(2 * Nc - 1)
    real2 = np.zeros(2 * Nc - 1)
    imag2 = np.zeros(2 * Nc - 1)

    for i in range(2 * (fb + 1), 2 * (fb + 1) + 2 * Nc - 1):
        real1[i - 2 * (fb + 1)] = sum(signal1[col] * np.cos(2 * np.pi * fc * i * t))
        imag1[i - 2 * (fb + 1)] = sum(signal1[col] * (-np.sin(2 * np.pi * fc * i * t)))
        real2[i - 2 * (fb + 1)] = sum(signal2[col] * np.cos(2 * np.pi * fc * i * t))
        imag2[i - 2 * (fb + 1)] = sum(signal2[col] * (-np.sin(2 * np.pi * fc * i * t)))

    Y1[col] = (real1 + 1j * imag1) / 2
    Y2[col] = (real2 + 1j * imag2) / 2

    x1Conv[col] = -signal.convolve(x1[col], x1[col])
    x2Conv[col] = -signal.convolve(x2[col], x2[col])

Signal1 = fft(signal1[0])/(Nb/2)
Signal2 = fft(signal2[0])/(Nb/2)

Signal1[0] = 0
Signal2[0] = 0


plt.figure(1)
plt.plot(10*np.log10(abs(fft(y1[0]))/(Nb/2)), linewidth=0.5)

plt.figure(2)
plt.plot(y1[0], signal1[0,:], linewidth=0.5, linestyle='-')
#
plt.figure(3)
plt.plot(10*np.log10(abs(Signal1)), linewidth=0.5)

Y1Angle = np.angle(Y1[0])
x1ConvAngle = np.angle(x1Conv[0])
angleDiff = np.angle(Y1[0], deg=True) - np.angle(x1Conv[0], deg=True)   ##查看二次谐波序列和卷积序列的俯角差是否固定
estH = Y1[0]/x1Conv[0]           ##信道估计，LS

fig, ax = plt.subplots(2)
ax[0].plot(abs(x1Conv[0]), linewidth=0.5)
ax[1].plot(abs(estH), linewidth=0.5)

fig1, ax = plt.subplots(3)
ax[0].plot(angleDiff, linewidth=0.5)
ax[1].plot(abs(estH), linewidth=0.5)

for i in range(len(estH)):
    if abs(estH[i].real) + abs(estH[i].imag) > 100:
        print("estH:", i,",", estH[i])
        estH[i] = (estH[i-1] + estH[i+1])/2     ##去除一些非常大的H，由于x1Conv中有一些非常小的值，这部分程序需要完善，
    if abs(x1Conv[0][i])<1:
        print("x1Conv:", i, ",", x1Conv[0][i])

ax[2].plot(abs(estH), linewidth=0.5)


##谐波消除
estHa = meanFilter(estH, 10)
residual = Y2[0] - x2Conv[0]*estH
residual1 = Y2[0] - x2Conv[0]*estHa

plt.figure(6)
plt.plot(10*np.log10(pow(abs(Y2[0]), 2)), linewidth = 0.5)
plt.plot(10*np.log10(pow(abs(residual), 2)), linewidth = 0.5)
plt.plot(10*np.log10(pow(abs(residual1), 2)), linewidth = 0.5, color='green')
plt.plot(10*np.log10(pow(fft(noise1[0:2*Nc-1])/10000, 2)), linewidth=0.5, color='black')

# Arls = np.zeros(2*Nc - 1, dtype=complex)
# xMatric = np.zeros(2*Nc - 1, dtype=complex)
residualRls1 = np.zeros(2*Nc-1, dtype=complex)
residualRls2 = np.zeros(2*Nc-1, dtype=complex)

eSum = np.zeros((2 * Nc - 1, iterations), dtype=complex)
for i in range(2*Nc - 1):
    Arls1, xMatric1 = Rls(x1Conv[:, i], Y1[:, i], 5)
    Yrls1 = Arls1[len(x1Conv[:, i])-1]*xMatric1.T
    Yrls1 = np.array(Yrls1)
    residualRls1[i] = (Y1[iterations-2][i] - Yrls1[0][iterations-1])

    for j in range(iterations):
        eSum[i][j] = (Y1[:, i] - Arls1[j] * xMatric1.T).sum()

    Arls2, xMatric2 = Rls(x2Conv[:, i], Y2[:, i], 5)
    Yrls2 = Arls1[len(x2Conv[:, i])-1]*xMatric2.T
    Yrls2 = np.array(Yrls2)
    residualRls2[i] = (Y2[iterations-1][i] - Yrls2[0][iterations-1])



print(type(residualRls1))
print(residualRls1[0])
print(Y1)

plt.figure(7)
plt.plot(10*np.log10(pow(abs(Y1[iterations-1]), 2)), linewidth = 0.5)
# plt.plot(10*np.log10(pow(abs(residualRls1), 2)), linewidth=0.5)
plt.plot(10*np.log10(pow(abs(residualRls2), 2)), linewidth=0.5)
plt.plot(10*np.log10(pow(fft(noise1[0:2*Nc-1])/10000, 2)), linewidth=0.5, color='black')

plt.figure(8)

eMean = meanFilter(eSum[3][1:], 0)
eMean1 = meanFilter(eSum[2][1:], 0)
plt.plot(10 * np.log10(pow(abs(eMean), 2)), color='black')
plt.plot(10 * np.log10(pow(abs(eMean1), 2)), color='blue')
plt.show()