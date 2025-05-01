import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pywt
from scipy.signal import butter, filtfilt, find_peaks

# Cargamos los datos del archivo capturados y guardados anteriormente
señal_EMG = np.loadtxt("datos_señaljoseph (3).txt", skiprows = 1)
tiempo = señal_EMG[:, 0]
voltaje = señal_EMG[:, 1]
fs = 100
duracion2=len(voltaje)/fs
print("duracion de la señal", duracion2)
fs = 1000 # Frecuencia de muestreo (Hz)
lowcout = 20 # Frecuencia mínima de corte (Hz)
highcut = 1000 # Frecuencia máxima de corte (Hz)

# 1. Graficar la señal original
plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(0,120)
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(115,120)
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(0,5)
plt.legend()
plt.show()

# DETECCIÓN DE PICOS R EN   0 A 5 SEGUNDOS
inicio = 0
fin = 5
mask = (tiempo >= inicio) & (tiempo <= fin)
tiempo_5s = tiempo[mask]
voltaje_5s = voltaje[mask]

# Filtro pasa banda 0.5–40 Hz
def butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=1000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

ecg_filtrada_5s = butter_bandpass_filter(voltaje_5s)

# Detección de picos R
peaks_5s, _ = find_peaks(ecg_filtrada_5s, distance=int(0.6 * fs), height=np.mean(ecg_filtrada_5s))

# Graficar ECG filtrada + picos
plt.figure(figsize=(15, 4))
plt.plot(tiempo_5s, ecg_filtrada_5s, label="ECG filtrada (0–5 s)")
plt.plot(tiempo_5s[peaks_5s], ecg_filtrada_5s[peaks_5s], "ro", label="Picos R")
plt.title("Detección de Picos R en ECG (0–5 segundos)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.show()
inicio = 0
fin = 5
mask = (tiempo >= inicio) & (tiempo <= fin)
tiempo_5s = tiempo[mask]
voltaje_5s = voltaje[mask]

# Filtro pasa banda: 0.5 a 40 Hz
def butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

ecg_filtrada_5s = butter_bandpass_filter(voltaje_5s)
peaks_5s, _ = find_peaks(ecg_filtrada_5s, distance=int(0.6 * fs), height=np.mean(ecg_filtrada_5s))
rr_intervals_5s = np.diff(tiempo_5s[peaks_5s])  # en segundos
media_rr = np.mean(rr_intervals_5s)
std_rr = np.std(rr_intervals_5s)

print(f"Media R-R (0–5 s): {media_rr:.3f} s")
print(f"Desviación estándar R-R (0–5 s): {std_rr:.3f} s")
if media_rr > 1:
    interpretacion = "frecuencia cardíaca baja (bradicardia o reposo profundo)"
elif media_rr < 0.6:
    interpretacion = "frecuencia cardíaca alta (posible estrés, ejercicio o error)"
else:
    interpretacion = "frecuencia cardíaca normal en reposo"
print("Interpretación fisiológica:", interpretacion)
wavelet = 'cmor1.5-1.0'  # Wavelet compleja Morlet
scales = np.arange(1, 512)
coef, freqs = pywt.cwt(ecg_filtrada_5s, scales, wavelet, sampling_period=1/fs)

rr_intervals = rr_intervals_5s
rr_times = tiempo_5s[peaks_5s][1:]
fs_interp = 4  # Frecuencia de muestreo de interpolación
tiempo_uniforme = np.linspace(rr_times[0], rr_times[-1], int((rr_times[-1] - rr_times[0]) * fs_interp))
f_interp = interp1d(rr_times, rr_intervals, kind='cubic', fill_value='extrapolate')
rr_interp = f_interp(tiempo_uniforme)
#  Transformada Wavelet Continua
wavelet = 'cmor1.5-1.0'
scales = np.arange(1, 512)
coef, freqs = pywt.cwt(rr_interp, scales, wavelet, sampling_period=1/fs_interp)
power = np.abs(coef)**2
# Graficar espectrograma
plt.figure(figsize=(12, 6))
plt.imshow(power, extent=[tiempo_uniforme[0], tiempo_uniforme[-1], freqs[-1], freqs[0]],
           cmap='plasma', aspect='auto', origin='lower')
plt.colorbar(label='Potencia Wavelet |W(t, f)|²')
plt.axhspan(0.04, 0.15, color='cyan', alpha=0.5, label='LF (0.04–0.15 Hz)')
plt.axhspan(0.15, 0.4, color='lightgreen', alpha=0.5, label='HF (0.15–0.4 Hz)')
plt.axhline(0.04, color='white', linestyle='--', linewidth=1)
plt.axhline(0.15, color='white', linestyle='--', linewidth=1)
plt.axhline(0.4, color='white', linestyle='--', linewidth=1)
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.title("Espectrograma Wavelet (CWT) de la Serie R-R\n(Wavelet: cmor1.5-1.0)")
plt.ylim(0, 0.5)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
