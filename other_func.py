import numpy as np
import warnings
warnings.filterwarnings("ignore")

def SNR(x,x_true):# Calculate the SNR
    return 20*np.log10(np.linalg.norm(x_true,2)/np.linalg.norm(x_true-x,2))

def add_noise(snr,signal):
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power/(10**(snr/10))
    noise = np.random.randn(*signal.shape)*np.sqrt(noise_power)
    noisy_signal = signal+noise
    noisy_signal_power = np.mean(np.abs(noisy_signal)**2)
    new_snr = 20*np.log10(np.linalg.norm(signal,2)/np.linalg.norm(signal-noisy_signal,2))
    return noisy_signal,new_snr

def Error(x, x_true):# Calculate the Error
    return (np.linalg.norm(x-x_true)/np.linalg.norm(x_true))