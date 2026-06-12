"""
Exploracion inicial de la base de datos
=========================================================

Objetivo de este script: Nos permite VERIFICAR que la base de datos se descargo 
y se descomprimio bien, CARGARLA con wfdb, y VISUALIZAR un registro para confirmar 
que las señales y las anotaciones estan donde esperamos que esten. 
Es un paso fundamental antes de empezar a procesar, porque si la base no se carga bien, todo lo que hagamos despues va a estar mal.
---

Sobre la base de datos (Penzel et al., Apnea-ECG Database, PhysioNet):

  - 70 registros nocturnos en total, divididos en:
      * Learning set (35): a01-a20, b01-b05, c01-c10. Tienen .apn (apnea).
      * Test set (35)   : x01-x35. NO tienen .apn.
  - Duracion: ~7 a 10 horas por registro.
  - ECG de UNA sola derivacion, muestreado a 100 Hz.
  - Cada registro trae 4 archivos:
      .dat -> senal de ECG
      .hea -> header (frecuencia de muestreo, ganancia, etc.)
      .apn -> anotaciones de apnea minuto a minuto (simbolo 'N' o 'A')
      .qrs -> anotaciones de QRS detectadas automaticamente (sin auditar)
  - Solo 8 registros (a01-a04, b01, c01-c03) tienen ademas 4 senales extra:
    respiracion toracica/abdominal (Resp C/A), flujo oronasal (Resp N) y SpO2.
    Vienen en un archivo aparte (rNNr.dat).

"""

# =============================================================================
# 1. Importamos las librerias
# =============================================================================
import os

import numpy as np
import scipy.signal
import scipy.interpolate
from scipy.fft import fft
import math as m

import matplotlib.pyplot as plt

import wfdb
from wfdb import processing


# =============================================================================
# 2. Configuracion: donde esta la base de datos
# =============================================================================
# Comentario: cambiar DATA_DIR si la carpeta de la base esta en otro lado.

DATA_DIR = 'apnea-ecg-database-1.0.0'

print('=' * 70)
print('Configuracion')
print('=' * 70)
print('DATA_DIR             :', DATA_DIR)
print('Existe la carpeta?   :', os.path.isdir(DATA_DIR))
if os.path.isdir(DATA_DIR):
    print('Cantidad de archivos :', len(os.listdir(DATA_DIR)))
else:
    raise FileNotFoundError(
        f'No encuentro la carpeta {DATA_DIR}. Revisa el ZIP de la base.')


# =============================================================================
# 3. Inventario: que registros tenemos?
# =============================================================================
# Antes de cargar nada, hagamos un inventario rapido para saber que hay
# adentro de DATA_DIR. Cada registro se identifica por su nombre base (ej.
# a01) y esta compuesto por los archivos a01.dat, a01.hea, a01.apn, a01.qrs.
#
# Listamos los nombres base (todo lo que tenga un .hea) y los separamos por
# grupo: learning (a*, b*, c*) y test (x*).

def inventariar_base(data_dir):
    """Lista los registros disponibles en data_dir y los agrupa.

    Devuelve un dict con claves 'apnea' (a01-a20), 'borderline' (b01-b05),
    'control' (c01-c10) y 'test' (x01-x35). El criterio es el nombre base.
    Descartamos los rNNr.hea (esos son los headers de los archivos
    combinados de respiracion + ECG, no los principales).
    """
    archivos_hea = sorted(f for f in os.listdir(data_dir) if f.endswith('.hea'))
    registros = [
        f[:-4] for f in archivos_hea
        if not f.endswith('r.hea') and not f.endswith('er.hea')
    ]

    grupos = {'apnea': [], 'borderline': [], 'control': [], 'test': [], 'otros': []}
    for r in registros:
        if r.startswith('a'):
            grupos['apnea'].append(r)
        elif r.startswith('b'):
            grupos['borderline'].append(r)
        elif r.startswith('c'):
            grupos['control'].append(r)
        elif r.startswith('x'):
            grupos['test'].append(r)
        else:
            grupos['otros'].append(r)
    return grupos


grupos = inventariar_base(DATA_DIR)

print()
print('=' * 70)
print('Inventario de la base')
print('=' * 70)
for nombre, lista in grupos.items():
    print(f'{nombre:12s} ({len(lista):2d}): {lista}')

# Si la base se descomprimio bien deberiamos ver 20 a*, 5 b*, 10 c* y 35 x*,
# total 70. Si falta alguno, el ZIP no se descomprimio completo.


# =============================================================================
# 4. Cargamos un registro
# =============================================================================
# Vamos a tomar un registro del grupo apnea y cargar la senal de ECG. 

REGISTRO = 'a01'                            # <- cambiar para explorar otro sujeto
path = os.path.join(DATA_DIR, REGISTRO)    

signal, fields = wfdb.rdsamp(path)

print()
print('=' * 70)
print(f'Registro {REGISTRO} - carga')
print('=' * 70)
print('Tipo de signal:', type(signal), '| shape:', signal.shape)
print('Tipo de fields:', type(fields))
print('Contenido de fields:')
for k, v in fields.items():
    print(f'  {k:12s}: {v}')

# De ahi sacamos los datos que vamos a usar todo el tiempo: la frecuencia de
# muestreo y la cantidad de muestras totales.
fs = fields['fs']
n_sig = fields['n_sig']
sig_len = fields['sig_len']
nombres_canales = fields['sig_name']
unidades = fields['units']

print()
print(f'Frecuencia de muestreo : {fs} Hz')
print(f'Cantidad de canales    : {n_sig}')
print(f'Largo de la senal      : {sig_len} muestras  '
      f'= {sig_len/fs/60:.1f} min  = {sig_len/fs/3600:.2f} h')
print(f'Nombre de los canales  : {nombres_canales}')
print(f'Unidades               : {unidades}')

# Como cada registro principal trae UN solo canal (el ECG), nos quedamos con
# ese unico canal en una variable ecg de una sola dimension.
ecg = signal[:, 0]
print(f'shape de ecg           : {ecg.shape}')
print(f'primeros valores       : {ecg[:5]}')


# =============================================================================
# 5. Visualizamos un segmento del ECG
# =============================================================================
# Vamos a mirar un segmento corto para confirmar que es un ECG y que tiene la forma esperada.

t = np.linspace(0, len(ecg)/fs, len(ecg))   # vector tiempo en segundos

t_inicio_min = 5        # arrancamos a los 5 min del registro
duracion_seg = 20       # mostramos 20 s
i0 = int(t_inicio_min*60*fs)
i1 = i0 + int(duracion_seg*fs)

plt.figure(figsize=(20, 4))
plt.plot(t[i0:i1], ecg[i0:i1])
plt.title(f'ECG - registro {REGISTRO} - segmento de {duracion_seg} s a partir del minuto {t_inicio_min}')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.grid(True, alpha=0.3)
plt.show()


# =============================================================================
# 6. Anotaciones de apnea (.apn)
# =============================================================================
# Para los registros del learning set (a*, b*, c*), hay un archivo .apn con
# una anotacion POR CADA MINUTO del registro:
#   'N' -> minuto SIN apnea (respiracion normal)
#   'A' -> minuto CON evento de apnea / hipopnea
#
# Las hicieron expertos usando las senales de respiracion. Son el ground truth contra el que vamos a comparar
# nuestras detecciones en el script 05.

ann_apnea = wfdb.rdann(path, 'apn')

print()
print('=' * 70)
print(f'Anotaciones de apnea (.apn) - registro {REGISTRO}')
print('=' * 70)
print('tipo                       :', type(ann_apnea))
print('cantidad de anotaciones    :', len(ann_apnea.symbol))
print('primeros 10 simbolos       :', ann_apnea.symbol[:10])
print('ultimos 10 simbolos        :', ann_apnea.symbol[-10:])
print('primeras 5 muestras        :', ann_apnea.sample[:5])
print('ultimas 5 muestras         :', ann_apnea.sample[-5:])
print('diff entre anotaciones     :', np.unique(np.diff(ann_apnea.sample)),
      '(en muestras)')
print('diff en segundos           :', np.unique(np.diff(ann_apnea.sample))/fs)

# La diferencia deberia dar 6000 muestras (= 60 s * 100 Hz), confirmando que
# efectivamente hay una marca por minuto.

# Contamos cuantos minutos son apnea y cuantos normales:
es_apnea = np.array([s == 'A' for s in ann_apnea.symbol])  

minutos_apnea  = int(es_apnea.sum())
minutos_normal = int((~es_apnea).sum())
minutos_total  = len(es_apnea)

print()
print(f'Total de minutos anotados : {minutos_total}')
print(f'Minutos con apnea (A)     : {minutos_apnea}  ({100*minutos_apnea/minutos_total:.1f}%)')
print(f'Minutos sin apnea (N)     : {minutos_normal}  ({100*minutos_normal/minutos_total:.1f}%)')


# =============================================================================
# 7. Linea de tiempo de apnea a lo largo de toda la noche
# =============================================================================
# Ahora visualizamos las anotaciones minuto a minuto como una serie binaria
# (1 = apnea, 0 = normal). Esto nos da una vista global del registro.

t_minutos = ann_apnea.sample / fs / 60     # muestras -> seg -> min
y_apnea = es_apnea.astype(int)             # 1 = A, 0 = N

plt.figure(figsize=(18, 3))
plt.step(t_minutos, y_apnea, where='post')
plt.fill_between(t_minutos, 0, y_apnea, step='post', alpha=0.3)
plt.yticks([0, 1], ['Normal (N)', 'Apnea (A)'])
plt.xlabel('Tiempo [min]')
plt.title(f'Anotaciones de apnea minuto a minuto - registro {REGISTRO}')
plt.grid(True, alpha=0.3)
plt.show()

# En un sujeto del grupo apnea (como a01) esperamos ver muchas zonas en 1,
# distribuidas a lo largo de toda la noche. En un control (c*) practicamente
# todo deberia estar en 0.


# =============================================================================
# 8. Anotaciones de QRS (.qrs)
# =============================================================================
# Ademas de las anotaciones de apnea, cada registro trae un .qrs con las
# ubicaciones de los QRS detectados AUTOMATICAMENTE por sqrs125. 

ann_qrs = wfdb.rdann(path, 'qrs')

print()
print('=' * 70)
print(f'Anotaciones de QRS (.qrs) - registro {REGISTRO}')
print('=' * 70)
print('cantidad de QRS detectados :', len(ann_qrs.sample))
print('primeros simbolos          :', ann_qrs.symbol[:10])
print('primeras posiciones        :', ann_qrs.sample[:10])

# Frecuencia cardiaca aproximada a partir de los RR:
rr_muestras = np.diff(ann_qrs.sample)   # diferencias entre QRS consecutivos
rr_segundos = rr_muestras / fs
fc_aprox = 60.0 / rr_segundos           # latidos por minuto

print()
print(f'Intervalos RR detectados : {len(rr_muestras)}')
print(f'RR mediano               : {np.median(rr_segundos):.3f} s')
print(f'FC mediana               : {np.median(fc_aprox):.1f} lpm')
print(f'FC min / max             : {fc_aprox.min():.1f} / {fc_aprox.max():.1f} lpm')

#Queremos ver que FC esté dentro del rango fisiológico esperado (ej. 40-180 lpm) y 
# que no haya intervalos RR muy cortos (< 0.3 s) o muy largos (> 2 s), lo cual indicaría errores
# en la detección de QRS.

# =============================================================================
# 9. Visualizacion integrada: ECG + QRS detectados
# =============================================================================
# Para confirmar visualmente que las marcas .qrs caen sobre los picos R, las
# superponemos sobre el segmento de ECG que ya graficamos.

mask_seg = (ann_qrs.sample >= i0) & (ann_qrs.sample < i1)
qrs_en_segmento = ann_qrs.sample[mask_seg]

plt.figure(figsize=(20, 4))
plt.plot(t[i0:i1], ecg[i0:i1], 'b', label='ECG')
plt.plot(t[qrs_en_segmento], ecg[qrs_en_segmento], 'ro', label='QRS (.qrs)')
plt.title(f'ECG + QRS provistos por la base - registro {REGISTRO}')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# =============================================================================
# 10. Vista combinada: ECG + linea de tiempo de apnea
# =============================================================================
# Juntamos las dos vistas en una sola figura para ver como se relaciona la
# senal con las anotaciones de apnea. Tramo mas largo (~30 min) para que se
# aprecien transiciones N <-> A.

t_inicio_min2 = 30
duracion_min2 = 30
j0 = int(t_inicio_min2*60*fs)
j1 = j0 + int(duracion_min2*60*fs)

# si el registro es mas corto que eso, ajustamos
if j1 > len(ecg):
    j0 = 0
    j1 = len(ecg)
    t_inicio_min2 = 0
    duracion_min2 = (j1 - j0) / fs / 60

fig, axes = plt.subplots(2, 1, figsize=(18, 5), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})

# Panel 1: ECG 
paso = 10
axes[0].plot(t[j0:j1:paso]/60, ecg[j0:j1:paso], linewidth=0.5)
axes[0].set_ylabel('ECG [mV]')
axes[0].set_title(f'Registro {REGISTRO} - ECG (decimado solo para visualizar) y anotaciones de apnea')
axes[0].grid(True, alpha=0.3)

# Panel 2: anotaciones de apnea en el mismo tramo
mask_ann = (t_minutos >= t_inicio_min2) & (t_minutos < t_inicio_min2 + duracion_min2)
axes[1].step(t_minutos[mask_ann], y_apnea[mask_ann], where='post')
axes[1].fill_between(t_minutos[mask_ann], 0, y_apnea[mask_ann], step='post', alpha=0.3)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['N', 'A'])
axes[1].set_xlabel('Tiempo [min]')
axes[1].set_ylabel('Apnea')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



print()
print('=' * 70)
print('Fin del script. Si llegamos hasta aca sin errores, la base se lee bien.')
print('=' * 70)
