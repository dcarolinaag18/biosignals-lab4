# -*- coding: utf-8 -*-
"""
Created on Wed May  2 08:04:37 2018

@author: Carolina
"""

import numpy as np
import scipy.signal as signal;
import matplotlib.pyplot as plt

Fs = 25;
Fo = 0.01;
n = np.arange(0,1000);
t=np.arange(0,1000/Fs,1/Fs)
x = np.sin(2*np.pi*Fo*n);
alpha = 0.8;
D = 0;
w = 0 + 0.1*np.random.randn(len(n));
nd = n+D; 
xd = np.sin(2*np.pi*Fo*nd); 
y = alpha*xd + w;
plt.figure();
plt.plot(n, x, n, y);
plt.xlabel('Muestras');
plt.ylabel('Amplitud (V)');
plt.show()

#%% Diseño del filtro FIR pasabajas
Fn = Fs/2;
Fc1 = 1.25;
Wn1 = Fc1/Fn;
N=31;
b=signal.firwin(N, Wn1)
w, h = signal.freqz(b);
h = 20*np.log10(np.abs(h));
w = w/np.max(w);
plt.figure();
plt.plot(w, h,'b');
plt.title('Respuesta en frecuencia del filtro digital');
plt.xlabel('Frecuencia (rad/muestra)');
plt.ylabel('Amplitud (dB)');
plt.show()

# Aplicación del filtro con filtfil y lfilter

yf = signal.filtfilt(b, 1, y);
yf2 = signal.lfilter(b, 1, y);
plt.figure()
plt.subplot(311)
plt.plot(n, y)
plt.subplot(312)
plt.plot(n, yf);
plt.subplot(313)
plt.plot(n, yf2);
plt.show()


#%% Diseño del firltro IIR
N=31;
b, a =signal.iirfilter(N,Wn=Wn1,btype='lowpass')
w, h = signal.freqz(b,a);
h = 20*np.log10(np.abs(h));
w = w/np.max(w);
plt.figure();
plt.plot(w, h,'b');
plt.title('Respuesta en frecuencia del filtro digital');
plt.xlabel('Frecuencia (rad/muestra)');
plt.ylabel('Amplitud (dB)');
plt.show()

# Aplicación del filtro

yf= signal.filtfilt(b, a, y);
yf2 = signal.lfilter(b, a, y);
plt.figure()
plt.subplot(311)
plt.plot(n, y)
plt.subplot(312)
plt.plot(n, yf);
plt.subplot(313)
plt.plot(n, yf2);
plt.show()

#%% Filtros pasaaltas y pasabanda

b1 = signal.firwin(31, 0.6, pass_zero=False);
b2 = signal.firwin(31, [0.1, 0.6],pass_zero=False);
w1, h1 = signal.freqz(b1);
w2, h2 = signal.freqz(b2);
plt.title('Respuesta en frecuencia del filtro digital');
plt.plot(w1, 20*np.log10(np.abs(h1)),'b');
plt.plot(w2, 20*np.log10(np.abs(h2)),'r');
plt.ylabel('Amplitud (dB)');
plt.xlabel('Frecuencia (rad/muestra)');
plt.legend(('Filtro pasabajas', 'Filtro pasabanda'), loc='best')
plt.savefig('fig1.png')
plt.show();


#%%
fs=500;
s = np.loadtxt('senal_filtros.txt')
s = s[:,1];
s1 = s[500:750]; 

f, pw =  signal.welch(s, fs)
plt.figure()
plt.semilogy(f, np.sqrt(pw))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Densidad espectral de potencia')
plt.show()

Fn = fs/2;
Fc1 = 60;
Wn1 = Fc1/Fn;
N=31;
b=signal.firwin(N, Wn1)
w, h = signal.freqz(b);
h = 20*np.log10(np.abs(h));
w = w/np.max(w);
plt.figure();
plt.plot(w, h,'b');
plt.title('Respuesta en frecuencia del filtro digital');
plt.xlabel('Frecuencia (rad/muestra)');
plt.ylabel('Amplitud (dB)');
plt.show()

n = np.arange(0,len(s));
sf= signal.filtfilt(b, 1, s);
plt.figure()
plt.subplot(211)
plt.plot(n, s)
plt.subplot(212)
plt.plot(n, sf);

f, pw =  signal.welch(sf, fs)
plt.figure()
plt.semilogy(f, np.sqrt(pw))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Densidad espectral de potencia')
plt.show()

s1f= sf[500:750]; 
plt.subplot(121); plt.plot(s1); plt.xlim(0, 250)
plt.subplot(122); plt.plot(s1f); plt.xlim(0, 250)
plt.show()



