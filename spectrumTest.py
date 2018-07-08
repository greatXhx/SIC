import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from RLS import *

BPSK = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
x1 = []     ##训练信号
x2 = []     ##需要消除时的发送信号
x3 = []        ##远端有用信号
Nc = 200     ##子载波长度
fb = 600    ##基础频率
Nb = 2000    ##50000
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
    y1 = y1 + x1[i].real*np.cos(2*np.pi*(fb + (i+1)*fc)*t) - x1[i].imag*np.sin(2*np.pi*(fb + (i+1)*fc)*t)
    y2 = y2 + x2[i].real*np.cos(2*np.pi*(fb + (i+1)*fc)*t) - x2[i].imag*np.sin(2*np.pi*(fb + (i+1)*fc)*t)
    # y1 = y1 + x1[i].real*np.cos(2*np.pi*(fb + i+1)*fc*t)
    # y2 = y2 + x2[i].real*np.cos(2*np.pi*(fb + i+1)*fc*t)

x = y1
y = y2
k = [1, 0, 0.00]

noise1 = [0.1*random.random() for i in range(Nb)]

signal1 = (k[0]*x - k[1]*pow(x, 2) - k[2]*pow(x, 3))


Signal1 = fft(signal1)/(Nb/2)

Signal1[0] = 0

real1 = np.zeros(Nc)
imag1 = np.zeros(Nc)

for i in range(Nc):
    cos = np.cos(2*np.pi*fc*(fb + i+1)*t)
    sin = -np.sin(2*np.pi*fc*(fb + i+1)*t)
    real1[i] = sum(signal1*cos)
    imag1[i] = sum(signal1*sin)

Y1 = (real1+1j*imag1)/Nb


plt.figure(1)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(10*np.log10(abs(fft(y1))/(Nb/2)), linewidth=0.5)
plt.title("频谱（幅度谱）")
plt.text(0, -170, "时域信号FFT后除以N/2(N为FFT的点数)")

plt.figure(2)
plt.plot(signal1, linewidth=0.5, linestyle='-')

plt.figure(3)
plt.plot(10*np.log10(abs(Y1)), linewidth=0.5)
plt.ylim(-100,10)
plt.title("功率谱")
plt.text(0, -110, "时域信号解调后乘T（T为每两个点的时间间隔）")
plt.show()