# -*- coding: utf-8 -*-
"""
Analisis espectral para la justificacion del filtrado
=================================================================

Objetivo de este script: realizar un análisis espectral para así DECIDIR y JUSTIFICAR
que filtros aplicar en el preprocesamiento (script 03).

Se busca mirar la densidad espectral de potencia (PSD) con Welch y verificar:
  1. Cuanta energia hay en 0-0.5 Hz (baseline wander: respiracion, movimiento).
  2. Donde se concentra la energia del QRS (5-25 Hz).
  3. Si hay pico de red (50 Hz, porque la base se grabo en Alemania).
  4. Si la PSD del ECG cambia entre minutos de apnea y minutos normales del
     mismo sujeto.

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

# Sujetos a comparar: uno de cada grupo
SUJETOS = ['a01', 'c01', 'x01']      # apnea, control, test

# Segmento a analizar: 1 minuto del medio de la noche (para evitar transitorios de inicio)
MINUTO_REFERENCIA = 30     
DURACION_SEG = 60           

FS = 100                    # Hz (la base se grabo a 100 Hz)


# =============================================================================
# Funciones auxiliares
# =============================================================================

def cargar_segmento(registro, minuto, duracion_seg, data_dir=DATA_DIR):
    path = os.path.join(data_dir, registro)
    sampfrom = int(minuto * 60 * FS)
    sampto = sampfrom + int(duracion_seg * FS)
    signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto)
    return signal[:, 0], fields['fs']


def psd_welch(x, fs, nperseg=1024):
    """
    nperseg=1024 a 100 Hz -> ventanas de 10.24 s.
    Resolucion frecuencial = fs/nperseg ~= 0.098 Hz (suficiente para ver
    baseline wander). Overlap 50% por defecto.
    """
    return sg.welch(x, fs=fs, nperseg=nperseg, detrend='constant')


def potencia_en_banda(f, Pxx, f_min, f_max):
    """Integra la PSD en una banda [f_min, f_max] ."""
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

segmentos = {}      
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
# Seccion 2: PSD por Welch - comparacion entre sujetos
# =============================================================================
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
plt.xlim(0, FS / 2 + 5)
plt.tight_layout()
plt.show()


# =============================================================================
# Seccion 3: zoom en baja frecuencia (0 - 2 Hz) - baseline wander
# =============================================================================
# Se observa la respiracion (0.15 - 0.4 Hz), el movimiento
# corporal (< 0.1 Hz) y la deriva electrodica.

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
# Seccion 4: zoom en alta frecuencia (35 - 50 Hz) - red electrica
# =============================================================================
# La red es 50 Hz. Pero como muestreamos a 100 Hz, Nyquist es 50 Hz, lo cual significa que el
# adquisidor tuvo que tener un filtro anti-aliasing por debajo de 50 Hz. Eso en general atenua mucho la red.

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
plt.xlim(30, FS / 2 + 5)
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# Seccion 5: tabla de potencias por banda
# =============================================================================
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
# Seccion 6: apnea vs normal en el MISMO sujeto (a01)
# =============================================================================
if 'a01' in segmentos:
    path_a01 = os.path.join(DATA_DIR, 'a01')
    try:
        ann = wfdb.rdann(path_a01, 'apn')
        es_apnea = np.array([s == 'A' for s in ann.symbol])

        minutos_n = np.where(~es_apnea)[0]
        minutos_a = np.where(es_apnea)[0]

        # no incluímos los primeros 5 min (transitorios)
        minutos_n = minutos_n[minutos_n >= 5]
        minutos_a = minutos_a[minutos_a >= 5]

        if len(minutos_n) > 0 and len(minutos_a) > 0:
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
            plt.xlim(0, FS / 2 + 5)
            plt.tight_layout()
            plt.show()
        else:
            print('a01 no tiene minutos de ambas clases para comparar. Salto seccion 7.')
    except FileNotFoundError:
        print('No encuentro a01.apn, salto la comparacion apnea/normal.')
else:
    print('a01 no esta cargado, salto la comparacion apnea/normal.')
