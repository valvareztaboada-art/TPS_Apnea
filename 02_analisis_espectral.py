# -*- coding: utf-8 -*-
"""
TPS - Apnea-ECG: analisis espectral y justificacion del filtrado
=================================================================

Objetivo: ver QUE TIENE la señal cruda en banda y de ahi DECIDIR y JUSTIFICAR
que filtros aplicar en el preprocesamiento (script 03).

La idea no es decidir a ojo "esta sucia / esta limpia", sino mirar la
densidad espectral de potencia (PSD) con Welch en varios sujetos de distintos
grupos y verificar:

  1. Cuanta energia hay en 0-0.5 Hz (baseline wander: respiracion, movimiento).
  2. Donde se concentra la energia del QRS (tipicamente 5-25 Hz).
  3. Si hay pico de red (50 Hz, porque la base se grabo en Alemania).
  4. Si la PSD del ECG cambia entre minutos de apnea y minutos normales del
     mismo sujeto (si cambia mucho, hay que tener cuidado al filtrar).

El resultado del script es una "lectura" impresa por consola con las
decisiones de filtrado justificadas, mas las figuras correspondientes que
van como evidencia en el informe.

Restricciones: solo numpy, scipy.signal, matplotlib, wfdb. Nada de wavelets,
ICA ni ML.
"""

import os
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import wfdb


# =============================================================================
# Configuracion
# =============================================================================

DATA_DIR = 'apnea-ecg-database-1.0.0'

# Sujetos a comparar: uno de cada grupo, asi el analisis no es anecdotico
SUJETOS = ['a01', 'c01', 'x01']      # apnea, control, test

# Segmento a analizar: 1 minuto del medio de la noche (asi evitamos los
# primeros minutos donde suele haber transitorios de colocacion de electrodos)
MINUTO_REFERENCIA = 30      # minuto a contar desde el inicio del registro
DURACION_SEG = 60           # 1 minuto

FS = 100                    # Hz (toda la base es a 100 Hz)


# =============================================================================
# Funciones auxiliares
# =============================================================================

def cargar_segmento(registro, minuto, duracion_seg, data_dir=DATA_DIR):
    """Carga `duracion_seg` segundos de ECG empezando en el minuto indicado.

    Devuelve (ecg_1d, fs).
    """
    path = os.path.join(data_dir, registro)
    sampfrom = int(minuto * 60 * FS)
    sampto = sampfrom + int(duracion_seg * FS)
    signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto)
    return signal[:, 0], fields['fs']


def psd_welch(x, fs, nperseg=1024):
    """Wrapper sobre scipy.signal.welch con parametros sensatos para ECG.

    nperseg=1024 a 100 Hz -> ventanas de 10.24 s.
    Resolucion frecuencial = fs/nperseg ~= 0.098 Hz (suficiente para ver
    baseline wander). Overlap 50% por defecto.

    Devuelve (f, Pxx) en Hz y (mV^2 / Hz) respectivamente.
    """
    return sg.welch(x, fs=fs, nperseg=nperseg, detrend='constant')


def potencia_en_banda(f, Pxx, f_min, f_max):
    """Integra la PSD en una banda [f_min, f_max] (regla del trapecio)."""
    mask = (f >= f_min) & (f <= f_max)
    if not mask.any():
        return 0.0
    # numpy 2.0+ usa trapezoid, anteriores usan trapz
    integrar = getattr(np, 'trapezoid', None) or np.trapz
    return float(integrar(Pxx[mask], f[mask]))


# =============================================================================
# Seccion 1: cargamos un segmento limpio de cada sujeto
# =============================================================================

print('=' * 70)
print(f'Cargando segmentos de referencia')
print(f'  minuto de inicio  : {MINUTO_REFERENCIA}')
print(f'  duracion          : {DURACION_SEG} s')
print('=' * 70)

segmentos = {}      # dict: nombre_registro -> array de ECG (1 minuto)
for r in SUJETOS:
    try:
        ecg, fs = cargar_segmento(r, MINUTO_REFERENCIA, DURACION_SEG)
        if fs != FS:
            print(f'  {r}: ATENCION, fs={fs} Hz, esperaba {FS}. Salto.')
            continue
        segmentos[r] = ecg
        print(f'  {r}: ok, {len(ecg)} muestras, '
              f'media={ecg.mean():+.4f} mV, std={ecg.std():.4f} mV, '
              f'rango=[{ecg.min():+.3f}, {ecg.max():+.3f}] mV')
    except FileNotFoundError:
        print(f'  {r}: NO ENCONTRADO en {DATA_DIR}/, lo salto.')
    except Exception as e:
        print(f'  {r}: ERROR al cargar ({e}). Salto.')

if not segmentos:
    raise RuntimeError('No pude cargar ningun sujeto. Revisa DATA_DIR y los nombres.')


# =============================================================================
# Seccion 2: vista en tiempo del segmento crudo
# =============================================================================
# Esto sirve sobre todo para que veas a ojo si hay deriva visible (baseline
# wander), saturaciones, o tramos planos. La inspeccion visual nunca alcanza
# como justificacion, pero es el primer indicio.

t = np.arange(DURACION_SEG * FS) / FS

fig, axes = plt.subplots(len(segmentos), 1,
                         figsize=(18, 2.2 * len(segmentos)),
                         sharex=True)
if len(segmentos) == 1:
    axes = [axes]

for ax, (r, ecg) in zip(axes, segmentos.items()):
    ax.plot(t, ecg, linewidth=0.7)
    ax.axhline(0, color='k', alpha=0.2, linewidth=0.5)
    ax.set_ylabel(f'{r}\n[mV]')
    ax.grid(True, alpha=0.3)

axes[0].set_title(
    f'ECG crudo - segmento de {DURACION_SEG} s desde el minuto {MINUTO_REFERENCIA}'
)
axes[-1].set_xlabel('Tiempo [s]')
plt.tight_layout()
plt.show()


# =============================================================================
# Seccion 3: PSD por Welch - comparacion entre sujetos
# =============================================================================
# Welch divide la señal en ventanas solapadas, calcula el periodograma de cada
# una y las promedia. Eso reduce la varianza del estimador (un periodograma
# crudo "vibra" mucho) y permite ver la estructura de fondo.

print()
print('=' * 70)
print('Calculo de PSD con metodo de Welch')
print('=' * 70)
print(f'  nperseg              : 1024 (10.24 s por ventana a 100 Hz)')
print(f'  overlap              : 50% (default)')
print(f'  resolucion frecuencial: {FS/1024:.4f} Hz/bin')
print(f'  rango analizable     : 0 a {FS/2} Hz (Nyquist)')

# Calculamos la PSD de cada sujeto una sola vez y la guardamos
psds = {r: psd_welch(ecg, FS) for r, ecg in segmentos.items()}

plt.figure(figsize=(16, 6))
for r, (f, Pxx) in psds.items():
    plt.semilogy(f, Pxx, label=r, linewidth=1.3)

# Marcamos las bandas de interes
plt.axvspan(0, 0.5, alpha=0.15, color='red',
            label='Baseline wander (0 - 0.5 Hz)')
plt.axvspan(5, 25, alpha=0.10, color='green',
            label='Banda QRS (5 - 25 Hz)')
plt.axvline(50, color='orange', linestyle='--', alpha=0.7,
            label='Red electrica (50 Hz = Nyquist)')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [mV² / Hz]  (escala log)')
plt.title('PSD de la señal cruda - comparacion entre sujetos')
plt.legend(loc='upper right')
plt.grid(True, which='both', alpha=0.3)
plt.xlim(0, FS / 2)
plt.tight_layout()
plt.show()


# =============================================================================
# Seccion 4: zoom en baja frecuencia (0 - 2 Hz) - baseline wander
# =============================================================================
# Aca es donde se ven la respiracion (0.15 - 0.4 Hz aprox), el movimiento
# corporal (< 0.1 Hz) y la deriva electrodica. Si hay una "joroba" alta en
# esta zona, hay que aplicar un pasa-altos.

plt.figure(figsize=(15, 5))
for r, (f, Pxx) in psds.items():
    plt.semilogy(f, Pxx, label=r, linewidth=1.5, marker='.', markersize=3)

plt.axvspan(0, 0.5, alpha=0.2, color='red', label='Baseline wander')
plt.axvline(0.5, color='red', linestyle='--', alpha=0.5,
            label='Corte sugerido (0.5 Hz)')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [mV² / Hz]')
plt.title('Zoom 0-2 Hz: baseline wander y respiracion')
plt.xlim(0, 2)
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# Seccion 5: zoom en alta frecuencia (35 - 50 Hz) - red electrica
# =============================================================================
# La base se grabo en Marburg (Alemania), donde la red es 50 Hz. Pero como
# muestreamos a 100 Hz, Nyquist es 50 Hz exactos, lo cual significa que el
# adquisidor TUVO que tener un filtro anti-aliasing por debajo de 50 Hz. Eso
# en general atenua mucho la red. Si igual aparece un pico aca, hay que
# sumar un notch.

plt.figure(figsize=(15, 5))
for r, (f, Pxx) in psds.items():
    plt.semilogy(f, Pxx, label=r, linewidth=1.5, marker='.', markersize=4)

plt.axvline(50, color='orange', linestyle='--', alpha=0.7,
            label='Red (50 Hz = Nyquist)')
plt.axvline(40, color='gray', linestyle=':', alpha=0.5,
            label='Pasa-bajos sugerido (40 Hz)')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [mV² / Hz]')
plt.title('Zoom 35-50 Hz: posible interferencia de red')
plt.xlim(35, 50)
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# Seccion 6: tabla de potencias por banda
# =============================================================================
# Numeros concretos para el informe. La idea: si la potencia en 0-0.5 Hz es
# del mismo orden o mayor que la de la banda QRS, hay baseline serio. Si la
# de 45-50 Hz es muy chica frente a la del QRS, el anti-aliasing ya hizo su
# trabajo.

print()
print('=' * 70)
print('Potencias integradas por banda (en mV^2)')
print('=' * 70)
print(f"{'Sujeto':<8} {'0-0.5 Hz':>12} {'5-25 Hz (QRS)':>16} "
      f"{'45-50 Hz':>12} {'ratio BL/QRS':>15}")
print('-' * 70)
for r, (f, Pxx) in psds.items():
    p_bl = potencia_en_banda(f, Pxx, 0, 0.5)
    p_qrs = potencia_en_banda(f, Pxx, 5, 25)
    p_50 = potencia_en_banda(f, Pxx, 45, 50)
    ratio = p_bl / p_qrs if p_qrs > 0 else float('inf')
    print(f"{r:<8} {p_bl:>12.5f} {p_qrs:>16.5f} {p_50:>12.5f} {ratio:>15.3f}")
print()
print('Lectura del ratio BL/QRS:')
print('  < 0.1  : baseline despreciable -> el pasa-altos es casi cosmetico')
print('  0.1-1  : baseline notable      -> el pasa-altos ayuda')
print('  > 1    : baseline domina       -> el pasa-altos es OBLIGATORIO')


# =============================================================================
# Seccion 7: apnea vs normal en el MISMO sujeto (a01)
# =============================================================================
# Queremos asegurarnos de que el filtro elegido no este "comiendose"
# diferencias que despues vamos a usar para detectar apnea.
# Comparamos PSD de un minuto anotado 'A' contra uno anotado 'N' en a01.
#
# Como la modulacion por apnea se manifiesta en la FRECUENCIA CARDIACA y su
# variabilidad (-> serie RR -> HRV), y no en la FORMA del latido, esperamos
# que la PSD del ECG en si sea parecida entre los dos estados. Si fuera muy
# distinta tendriamos que repensar el filtrado.

if 'a01' in segmentos:
    path_a01 = os.path.join(DATA_DIR, 'a01')
    try:
        ann = wfdb.rdann(path_a01, 'apn')
        es_apnea = np.array([s == 'A' for s in ann.symbol])

        minutos_n = np.where(~es_apnea)[0]
        minutos_a = np.where(es_apnea)[0]

        # nos saltamos los primeros 5 min por las dudas (transitorios)
        minutos_n = minutos_n[minutos_n >= 5]
        minutos_a = minutos_a[minutos_a >= 5]

        if len(minutos_n) > 0 and len(minutos_a) > 0:
            # tomamos uno bien al medio de cada lista
            min_n = int(np.median(minutos_n))
            min_a = int(np.median(minutos_a))

            print()
            print('=' * 70)
            print(f'Comparacion apnea vs normal en a01:')
            print(f'  minuto N elegido: {min_n}')
            print(f'  minuto A elegido: {min_a}')
            print('=' * 70)

            ecg_n, _ = cargar_segmento('a01', min_n, 60)
            ecg_a, _ = cargar_segmento('a01', min_a, 60)

            f_n, Pxx_n = psd_welch(ecg_n, FS)
            f_a, Pxx_a = psd_welch(ecg_a, FS)

            # imprimimos las potencias en bandas para comparar
            for nombre, f, Pxx in [('Normal', f_n, Pxx_n),
                                    ('Apnea', f_a, Pxx_a)]:
                p_bl = potencia_en_banda(f, Pxx, 0, 0.5)
                p_qrs = potencia_en_banda(f, Pxx, 5, 25)
                print(f'  {nombre:<7} -> P(0-0.5 Hz)={p_bl:.5f}  '
                      f'P(5-25 Hz)={p_qrs:.5f}  ratio={p_bl/p_qrs:.3f}')

            # grafico comparativo
            plt.figure(figsize=(15, 5))
            plt.semilogy(f_n, Pxx_n, label=f'a01 min {min_n} (Normal)',
                         linewidth=1.4)
            plt.semilogy(f_a, Pxx_a, label=f'a01 min {min_a} (Apnea)',
                         linewidth=1.4)
            plt.axvspan(0, 0.5, alpha=0.15, color='red',
                        label='Baseline wander')
            plt.axvspan(5, 25, alpha=0.10, color='green', label='Banda QRS')
            plt.xlabel('Frecuencia [Hz]')
            plt.ylabel('PSD [mV² / Hz]')
            plt.title('PSD del ECG: apnea vs normal en el mismo sujeto (a01)')
            plt.legend()
            plt.grid(True, which='both', alpha=0.3)
            plt.xlim(0, FS / 2)
            plt.tight_layout()
            plt.show()
        else:
            print('a01 no tiene minutos de ambas clases para comparar. Salto seccion 7.')
    except FileNotFoundError:
        print('No encuentro a01.apn, salto la comparacion apnea/normal.')
else:
    print('a01 no esta cargado, salto la comparacion apnea/normal.')


# =============================================================================
# Seccion 8: lectura del analisis y decisiones de filtrado
# =============================================================================

print()
print('=' * 70)
print('LECTURA DEL ANALISIS Y JUSTIFICACION DEL FILTRADO')
print('=' * 70)
print("""
Lo que hay que mirar en los graficos / numeros de arriba:

1) Baseline wander (0 - 0.5 Hz):
   - Si en la PSD ves una "joroba" alta abajo del 0.5 Hz y/o el ratio BL/QRS
     de la tabla es > ~0.3, hay que aplicar un PASA-ALTOS.
   - Si el ratio queda << 0.1, el pasa-altos sigue siendo buena idea para
     que Pan-Tompkins y el detector funcionen mejor, pero ya no es critico.
   - La alternativa al pasa-altos (la que muestra la profe en clase) es
     estimar la deriva con CUBIC SPLINES sobre los Q-onset y restarsela a
     la señal. Las dos cosas son validas y se pueden comparar en el informe.

2) Banda QRS (5 - 25 Hz):
   - Es donde queremos PRESERVAR la energia. Cualquier filtro elegido tiene
     que tener una banda de paso que cubra esto sin atenuacion.
   - Para el DETECTOR de QRS (Pan-Tompkins) se usa una banda mas estrecha
     (5 - 15 Hz). Eso es solo para el detector, no para el ECG que usamos
     para visualizar / reportar.

3) Red electrica (cerca de 50 Hz):
   - Como Nyquist es 50 Hz, el adquisidor tuvo que filtrar antes. Si en el
     zoom 35-50 Hz no se ve un pico angosto, no hace falta notch.
   - Si igual aparece un pico en 50 Hz, agregamos un notch (sg.iirnotch).

4) Apnea vs normal (a01):
   - Esperamos PSDs parecidas en la banda del ECG: la modulacion por apnea
     esta en la FC y su variabilidad, no en la forma del latido. Si las dos
     PSDs son visualmente similares, el filtro generico es seguro y no
     introduce sesgo entre clases.
   - Si fueran muy distintas, habria que justificar mas finamente que el
     filtro no afecta las features de HRV que usamos despues.

DECISIONES PROPUESTAS (a confirmar mirando los graficos en tu corrida):

   - Pasa-altos en 0.5 Hz, Butterworth orden 4, aplicado con filtfilt
     (fase cero, no introduce retardo => no corre los QRS).
     -> elimina baseline wander.

   - Pasa-bajos en 40 Hz, Butterworth orden 4, con filtfilt.
     -> atenua EMG residual y deja el QRS intacto.

   - Notch en 50 Hz (iirnotch con Q=30): SOLO si en el zoom 35-50 Hz se ve
     un pico angosto. Probable que no haga falta, pero lo dejamos disponible.

   - Para el detector de QRS especificamente: banda 5 - 15 Hz (Pan-Tompkins
     clasico). Esto va a un script aparte, no es el "ECG limpio" general.

Todo esto va al script 03_preprocesamiento.py.
""")

print('=' * 70)
print('Fin del analisis espectral.')
print('=' * 70)
