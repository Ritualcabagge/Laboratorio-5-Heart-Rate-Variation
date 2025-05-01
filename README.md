# Laboratorio-5-Heart-Rate-Variation

- Introducción:
  
La variabilidad de la frecuencia cardíaca (HRV, por sus siglas en inglés) es un indicador fundamental del estado del sistema nervioso autónomo, ya que permite evaluar el equilibrio entre la actividad simpática y parasimpática que regula el ritmo cardíaco. En esta práctica de laboratorio, se abordará el análisis de la HRV mediante la aplicación de la transformada wavelet, una herramienta eficaz para observar la evolución temporal y frecuencial de señales biológicas. Esta técnica permite detectar cambios sutiles en la actividad cardíaca que no son evidentes mediante métodos tradicionales del dominio del tiempo. El conocimiento y análisis de la HRV no solo es relevante en contextos clínicos, sino también en campos como el deporte, la psicología y el monitoreo de estrés.

- Metodología, desarrollo y análisis:

Primero se descargaron los correspondientes programas para la adquisicion de datos DAQ NI USB, en el cual instalamos una aplicacion y ampliacion en Python, luego se conectan los electrodos al amplificador y al sistema DAQ. Luego realizamos la correspondiente investigación de acuerdo a la guia de laboratorio con respecto a la actividad simpática y parasimpática del sistema nervioso autónomo, transformada wavelet y Variabilidad de la frecuencia cardiaca (HRV) medida como fluctuaciones en el intervalo R-R, y las frecuencias de interés en este análisis.

![image](https://github.com/user-attachments/assets/ab3df84d-3256-4978-8a06-895365832d51)
Fig. 1 Diagrama de flujo plan de acción.

- Montaje:

Se conectaron los electrodos al sensor y este a su vez al dispositivo NU, el cual de igual manera proporcionaba la energia suficiente para su correcto funcionamiento. Cabe recalcar que se utilizo un código parecido al de la captura de la señal de la actividad múscular desde un sensor de EMG, con cambios en la frecuencia ya que para caputrar una señal de ECG es recomendado usar desde 250 Hz a 1000 Hz, en nuestro caso utilizamos el máximo de 1000 Hz ya que nuestro objetivo es lograr identificar y análizar el comportamiento de los picos R.

![image](https://github.com/user-attachments/assets/b26ffaf4-68fe-436c-9667-3464a90e272d)
Fig. 2 Imagen ilustrativa montaje.

Ahora, ya elegido el sujeto de prueba realizamos la captura de la señal electrocardiográfica durante 5 minutos, al principio en reposo y luego exaltado, para capturar y análizar.
Utilizamos el siguiente código para capturar la señal en tiempo real y guardar los datos en un archivo .txt para realizar su correspondiente filtro y análisis.

```python 
import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
```
En este apartado llamamos e importamos las diferentes librerias con las que vamos trabaja, en este caso nidaqx para interactuar con el hardware de adquisició NI DAQ, por otro
lado importamos para definir el tipo de adquisición (finita, continua, etc) y por último para manejar operaciones numéricas, en este caso como crear el eje de tiempo.

```python
# Parámetros de adquisición 
sample_rate = 1000        # Frecuencia de muestreo en Hz
duration_minutes = 5  # Duración de la adquisición en minutos
duration_seconds = duration_minutes * 60 # Duración en segundos
num_samples = int(sample_rate * duration_seconds) # Número total de muestras
```
Se definen la frecuencia de muestreo (1000 Hz) y la duración de la adquisición (5 minutos). A partir de estos datos, se calcula el número total de muestras necesarias para capturar la señal sin pérdida de información.

```python
with nidaqmx.Task() as task:
    # Configuramos canal a utilizar
    task.ai_channels.add_ai_voltage_chan("Dev3/ai0")
    
    # Adquisición finita de muestras
    task.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=num_samples
    
    )
```
Dentro de un bloque with, se crea una tarea DAQ para garantizar que se inicie y cierre correctamente. Se configura el canal analógico de entrada (Dev3/ai0) y se define una adquisición finita del número exacto de muestras con el temporizador interno.
```python
    task.start()
    task.wait_until_done(timeout=duration_seconds + 10)
    data = task.read(number_of_samples_per_channel = num_samples)
```
Una vez finalizada la captura, los datos se leen en un arreglo. Se crea un eje de tiempo paralelo usando numpy.linspace que representa cada instante de muestra, desde 0 hasta el final de la adquisición.

```python
# Crear un eje de tiempo para la gráfica
time_axis = np.linspace(0, duration_seconds, num_samples, endpoint=False)
with open("datos_señal1.txt", "w") as archivo_txt:
    archivo_txt.write("Tiempo (s)\tVoltaje (V)\n")
    for t, v in zip(time_axis, data):
        archivo_txt.write(f"{t:.6f}\t{v:.6f}\n")
```
Finalmente, los datos de tiempo y voltaje se guardan en un archivo .txt, separados por tabulaciones y con encabezado, listos para ser graficados en el siguiente apartado.

Ahora ya teniendo listo el código para el filtrado y la adquisición de datos, procedemos a utilizar el siguiente código para gráficar los datos filtrados obtenidos anteriormente, de tal manera podemos identificar de manera clara los diferentes intervalos y picos. Mediante este código gráficamos toda la señal y luego más a fondo al principio en reposo y luego en excitación. Con esto podemos analizar el comportamiento de los picos y sus diferencias. 

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import nidaqmx
from scipy.signal import butter, filtfilt, windows
from scipy.fftpack import fft, fftfreq
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind, norm
```
Se importan librerías para: Procesamiento numérico (numpy), Gráficas (matplotlib), Filtros y análisis de frecuencia (scipy.signal, fft)

```python
# Cargamos los datos del archivo capturados y guardados anteriormente
señal_EMG = np.loadtxt("datos_señaljoseph.txt", skiprows = 1)
tiempo = señal_EMG[:, 0]
voltaje = señal_EMG[:, 1]
```
Primero se importa la señal desde el archivo datos_señaljoseph.txt, que contiene dos columnas: el tiempo en segundos y el voltaje de la señal en voltios. Para ello, se utiliza numpy.loadtxt con skiprows=1, que omite la primera fila del archivo (el encabezado con los nombres de las columnas). Luego, se separan los datos en dos arreglos: tiempo, que representa los puntos temporales, y voltaje, que corresponde a las amplitudes registradas de la señal ECG.

```python
fs = 1000
duracion2=len(voltaje)/fs
print("duracion de la señal", duracion2)
fs = 1000 # Frecuencia de muestreo (Hz)
lowcout = 20 # Frecuencia mínima de corte (Hz)
highcut = 1000 # Frecuencia máxima de corte (Hz)
```
Una vez cargados los datos, se define la frecuencia de muestreo como fs = 1000 Hz, que es adecuada para señales ECG, ya que permite capturar adecuadamente componentes hasta los 100 Hz sin aliasing. Se establecen también las frecuencias de corte del filtro: una frecuencia mínima (lowcut = 20 Hz) que elimina componentes de muy baja frecuencia como artefactos de movimiento o tendencia basal, y una frecuencia máxima (highcut = 100 Hz) que elimina el ruido de alta frecuencia como interferencias eléctricas. El orden del filtro se define como 4, lo que garantiza una buena pendiente de atenuación sin hacer el sistema demasiado sensible a pequeñas variaciones.

```python
# === Función para diseñar y aplicar el filtro pasa banda ===
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# === Aplicar el filtro a la señal ===
voltaje_filtrado = butter_bandpass_filter(voltaje, lowcut, highcut, fs, order)
```
La función butter_bandpass_filter es responsable de diseñar y aplicar el filtro pasa banda. Dentro de esta función, primero se calcula la frecuencia de Nyquist, que es la mitad de la frecuencia de muestreo y representa el límite superior que puede ser representado correctamente en una señal digital. Luego, se normalizan las frecuencias de corte dividiéndolas por la frecuencia de Nyquist. Con estos valores, se utiliza la función butter de scipy.signal para obtener los coeficientes del filtro Butterworth. Finalmente, se aplica el filtro con la función filtfilt, que filtra la señal en ambas direcciones (adelante y atrás), lo que tiene la ventaja de no introducir desfase en la señal, una propiedad fundamental cuando se requiere precisión temporal como en el caso del ECG.

```python
plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(0,120)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(0,5)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(115,120)
plt.grid(True)
plt.legend()
plt.show()
```
Después de aplicar el filtro, se generan tres gráficas con matplotlib para observar el comportamiento de la señal:

Gráfica completa (0 a 120 s): Permite observar la señal a lo largo de toda su duración, útil para detectar cambios globales, artefactos o eventos cardíacos atípicos a lo largo del tiempo.

Zoom en los primeros 5 segundos: Ayuda a visualizar con más detalle la morfología de las ondas del ECG, como los complejos P-QRS-T, que se presentan de forma cíclica.

Zoom en los últimos 5 segundos: Similar a la anterior, pero muestra los instantes finales de la adquisición, lo cual es útil para comparar si la señal se mantiene estable en el tiempo.

Cada gráfica está debidamente etiquetada con títulos, ejes y leyendas, y se le aplica una cuadrícula para facilitar la lectura de valores.

```python

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
```

Análisis de la HRV en el dominio del tiempo
```python
# Análisis HRV en dominio del tiempo

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
```
Este bloque calcula la media y variabilidad de los intervalos entre latidos (R-R), lo cual permite evaluar el estado del sistema nervioso autónomo obteniendo asi Interpreta automáticamente el valor de la media R–R donde 

-1 s - ritmo lento - posible bradicardia.


-0.6 s - ritmo acelerado - posible estrés o ejercicio.


Entre 0.6 y 1 s - frecuencia cardíaca normal.


obteniendo :

Media R-R (0–5 s): 0.943 s


Desviación estándar R-R (0–5 s): 0.072 s


Interpretación fisiológica: frecuencia cardíaca normal en reposo


e. Aplicación de transformada Wavelet

```phyton
wavelet = 'cmor1.5-1.0'  # Wavelet compleja Morlet
scales = np.arange(1, 512)
coef, freqs = pywt.cwt(ecg_filtrada_5s, scales, wavelet, sampling_period=1/fs)
rr_intervals = rr_intervals_5s
rr_times = tiempo_5s[peaks_5s][1:]

# configurar RR a frecuencia constante (para CWT)
fs_interp = 4  # Frecuencia de muestreo de interpolación
tiempo_uniforme = np.linspace(rr_times[0], rr_times[-1], int((rr_times[-1] - rr_times[0]) * fs_interp))
f_interp = interp1d(rr_times, rr_intervals, kind='cubic', fill_value='extrapolate')
rr_interp = f_interp(tiempo_uniforme)

# Transformada Wavelet Continua
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
```

Se aplica una Transformada Wavelet Continua a la serie R–R obtenida, permitiendo visualizar cómo varía la actividad del sistema nervioso autónomo en el tiempo y en distintas bandas de frecuencia luego se realiza una i adaptacion a la señal R–R a una frecuencia uniforme y luego se obtiene el espectrograma que muestra la potencia de la señal HRV en las bandas de interés (LF y HF) en donde:


Las frecuencias bajas (LF) entre 0.04 y 0.15 Hz están asociadas a una combinación de actividad simpática y parasimpática


Las frecuencias altas (HF) entre 0.15 y 0.4 Hz reflejan principalmente la actividad parasimpática (vagal), relacionada con la respiración

RESULTADOS:


-SEÑAL ECG


![image](https://github.com/user-attachments/assets/c25bba38-dfb5-4e9a-9ab3-212cf31aafca)


-SEÑAL ECG(RANGO DE TIEMPO)


![image](https://github.com/user-attachments/assets/e33a8597-ac88-47f0-82bd-6206841cb2f3)


-SEÑAL FILTRADA CON SU DETECCION DE PICOS R-R


![image](https://github.com/user-attachments/assets/acb11eb4-8689-44bd-8190-9ede069d38c0)


-ESPECTOGRAMA WAVELET(CTW)


![image](https://github.com/user-attachments/assets/0a8a5491-c82b-4d25-8f14-457bb4d844f9)








