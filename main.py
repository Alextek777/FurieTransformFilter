import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import cmath
import time

data = pd.read_csv("blink.csv");

timeStamp = data[' Timestamp (Formatted)']
start_time = timeStamp[0][12:]
end_time = timeStamp[timeStamp.size-1][12:]

s_h = int(start_time[0:2])
s_m = int(start_time[3:5])
s_s = int(start_time[6:8])
s_ms = int(start_time[9:])

e_h = int(end_time[0:2])
e_m = int(end_time[3:5])
e_s = int(end_time[6:8])
e_ms = int(end_time[9:])

elapsedSeconds = np.abs(s_h - e_h)*3600 + np.abs(s_m-e_m)*60 + np.abs(s_s-e_s) + np.abs(s_ms-e_ms)/1000

print("Elapsed seconds : ",elapsedSeconds)

eraseColumns = data.columns.values
eraseColumns = eraseColumns[9:24]
data = data.drop(columns=eraseColumns)

t_whole = 1.0
elapsedSeconds
es = elapsedSeconds
dt = es / len(data[' EXG Channel 0'])
t_long = np.arange(0, elapsedSeconds - dt/2, dt)

x_whole_filtred = np.array(0)
y_whole_filtred = np.array(0)

final_filtred = np.zeros(len(data[' EXG Channel 0']))


# -------------------------------------------plot and build data interval
start = int(t_whole / dt)
end = int(start + t_whole / dt)
fourie_data = data[' EXG Channel 0'][start:end]

t = np.arange(0, t_whole, dt)
f = fourie_data

if len(t) > len(f):
    t = t[0:len(t) - 1]
elif len(t) < len(f):
    f = f[0:len(f) - 1]

fig, axs = plt.subplots(1, 1)

plt.plot(t_long, data[' EXG Channel 0'][:], color="red", label="Whote Interval [1]")
plt.plot(t, f, color="blue", label="Noisy Interval [1]")
plt.xlim(t_long[0], t_long[-1])
plt.legend()

# -------------------------------------------fft transform and plot
n = len(t)
fhat = np.fft.fft(f, n)
PSD = fhat * np.conj(fhat) / n
PSD = np.log(PSD)
freq = (1 / (dt * n)) * np.arange(n)
L = np.arange(1, np.floor(n / 2), dtype='int')

fig, axs = plt.subplots(1, 1)

plt.plot(freq[L], np.abs(PSD[L]), color="green", label="Noisy fourie [2]")
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

# ------------------------------------------inverse fft
indexes = np.zeros(len(freq))
# First half
for q in range(0, len(L)):
    indexes[q] = 1 if L[q] > 7 and L[q] < 20 else 0.1
# Mirror
for q in range(len(L), len(freq)):
    indexes[q] = indexes[len(L) - q]

PSDclean = PSD * indexes
fhat = indexes * fhat
ffilt = np.fft.ifft(fhat)

fig, axs = plt.subplots(3, 1)

plt.sca(axs[0])
plt.plot(t, f, color="red", label="Noisy [3]")
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(t, ffilt, color="green", label="Clean [4]")
plt.xlim(t[0], t[-1])
plt.legend()
np.append(x_whole_filtred, t)
np.append(y_whole_filtred, ffilt)

plt.sca(axs[2])
plt.plot(freq[L], np.abs(PSD[L]), color="red", label="Noisy [5]")
plt.plot(freq[L], np.abs(PSDclean[L]), color="green", label="Clean [5]")
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.show()
