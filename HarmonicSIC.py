import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal
from filter import *
from RLS import *

BPSK = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
x1 = []     ##训练信号
x2 = []     ##需要消除时的发送信号
x3 = []        ##远端有用信号
Nc = 600     ##子载波长度
fb = 3000    ##基础频率
Nb = 30000    ##50000
x = 0

for i in range(Nc):
    x1.append(BPSK[random.randint(0, 3)])
    x2.append(BPSK[random.randint(0, 3)])
    x3.append(BPSK[random.randint(0, 3)])
# print(x1)

n = np.arange(1, Nb+1, 1)
t = n/Nb
# print(len(t))
fc = 1    ##子载波间隔
y1 = np.zeros(Nb)   ##训练序列
y2 = np.zeros(Nb)   ##发送序列
y3 = np.zeros(Nb)   ##远端有用信号

for i in range(Nc):
    y1 = y1 + x1[i].real*np.cos(2*np.pi*(fb + i+1)*fc*t)*0.05 - x1[i].imag*np.sin(2*np.pi*(fb + i+1)*fc*t)*0.05
    y2 = y2 + x2[i].real*np.cos(2*np.pi*(fb + i+1)*fc*t)*0.05 - x2[i].imag*np.sin(2*np.pi*(fb + i+1)*fc*t)*0.05

x = y1
y = y2
k = [10, 1, 0.01]

noise1 = [0.1*random.random() for i in range(Nb)]

signal1 = (k[0]*x - k[1]*pow(x, 2) - k[2]*pow(x, 3) + noise1)/10000

noise2 = [0.1*random.random() for i in range(Nb)]
signal2 = (k[0]*y - k[1]*pow(y, 2) - k[2]*pow(y, 3) + noise2)/10000

Signal1 = fft(signal1)/(Nb/2)
Signal2 = fft(signal2)/(Nb/2)

Signal1[0] = 0
Signal2[0] = 0

real1 = np.zeros(2*Nc-1)
imag1 = np.zeros(2*Nc-1)
real2 = np.zeros(2*Nc-1)
imag2 = np.zeros(2*Nc-1)

for i in range(2*(fb+1), 2*(fb+1) + 2*Nc-1):
    real1[i-2*(fb+1)] = sum(signal1*np.cos(2*np.pi*fc*i*t))
    imag1[i-2*(fb+1)] = sum(signal1*(-np.sin(2*np.pi*fc*i*t)))
    real2[i-2*(fb+1)] = sum(signal2*np.cos(2*np.pi*fc*i*t))
    imag2[i-2*(fb+1)] = sum(signal2*(-np.sin(2*np.pi*fc*i*t)))

Y1 = (real1+1j*imag1)/2
Y2 = (real2+1j*imag2)/2

x1Conv = -signal.convolve(x1, x1)
x2Conv = -signal.convolve(x2, x2)

plt.figure(1)
plt.plot(10*np.log10(abs(fft(y1))/(Nb/2)), linewidth=0.5)

plt.figure(2)
plt.plot(x, signal1, linewidth=0.5, linestyle='-')
#
plt.figure(3)
plt.plot(10*np.log10(abs(Signal1)), linewidth=0.5)

Y1Angle = np.angle(Y1)
x1ConvAngle = np.angle(x1Conv)
angleDiff = np.angle(Y1, deg=True) - np.angle(x1Conv, deg=True)   ##查看二次谐波序列和卷积序列的俯角差是否固定
estH = Y1/x1Conv           ##信道估计，LS

fig, ax = plt.subplots(2)
ax[0].plot(abs(x1Conv), linewidth=0.5)
ax[1].plot(abs(estH), linewidth=0.5)

fig1, ax = plt.subplots(3)
ax[0].plot(angleDiff, linewidth=0.5)
ax[1].plot(abs(estH), linewidth=0.5)

for i in range(len(estH)):
    if abs(estH[i].real) + abs(estH[i].imag) > 100:
        print("estH:", i,",", estH[i])
        estH[i] = (estH[i-1] + estH[i+1])/2     ##去除一些非常大的H，由于x1Conv中有一些非常小的值，这部分程序需要完善，
    if abs(x1Conv[i])<1:
        print("x1Conv:", i, ",", x1Conv[i])

ax[2].plot(abs(estH), linewidth=0.5)


##谐波消除
estHa = meanFilter(estH, 10)
residual = Y2 - x2Conv*estH
residual1 = Y2 - x2Conv*estHa

plt.figure(6)
plt.plot(10*np.log10(pow(abs(Y2), 2)), linewidth = 0.5)
plt.plot(10*np.log10(pow(abs(residual), 2)), linewidth = 0.5)
plt.plot(10*np.log10(pow(abs(residual1), 2)), linewidth = 0.5, color='green')
plt.plot(10*np.log10(pow(fft(noise1[0:2*Nc-1])/10000, 2)), linewidth=0.5, color='black')

plt.show()