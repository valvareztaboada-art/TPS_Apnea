# -*- coding: utf-8 -*-
"""
Preprocesamiento + Pan-Tompkins + limpieza de RR (para un sujeto)
=============================

Script EXPLORATORIO sobre 1 sujeto. Usa las funciones del modulo
src.pipeline. El objetivo de este script es generar las figuras y mensajes que van al informe.

Para procesar TODOS los sujetos y llenar el cache/, ver 03b_procesar_todos.py.

"""

import os
import sys

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.pipeline import (
    # carga
    cargar_segmento,
    cargar_anotaciones_qrs,
    # filtros generales
    filtrar_ecg_general,
    disenar_pasaaltos, disenar_pasabajos,
    # Pan-Tompkins (etapas separadas para visualizar paso a paso)
    pan_tompkins,
    comparar_detecciones,
    # RR
    serie_rr,
    filtro_rango_fisiologico, filtro_malik, filtro_mediana_local,
    interpolar_nan,
    limpiar_rr,
    # constantes 
    FC_PASAALTOS, FC_PASABAJOS, ORDEN_BUTTER,
    PT_BANDA_BAJA, PT_BANDA_ALTA, PT_VENTANA_INT_MS,
    PT_REFRACTARIO_MS, PT_ADAPT_ALPHA, PT_GUARDA_BORDE_MS,
    RR_MIN_FISIOL, RR_MAX_FISIOL,
    MALIK_UMBRAL, MEDIANA_VENTANA, MEDIANA_UMBRAL,
)


# =============================================================================
# Configuracion local del exploratorio
# =============================================================================

DATA_DIR = 'apnea-ecg-database-1.0.0'
REGISTRO = 'a01'

MINUTO_INICIO = 30
DURACION_MIN = 10
VENTANA_VIZ_SEG = 10


# =============================================================================
# Carga
# =============================================================================

print('=' * 70)
print(f'Preprocesamiento + Pan-Tompkins + limpieza de RR (exploratorio)')
print(f'  registro       : {REGISTRO}')
print(f'  desde minuto   : {MINUTO_INICIO}')
print(f'  duracion       : {DURACION_MIN} min')
print('=' * 70)

ecg_crudo, fs, _ = cargar_segmento(REGISTRO, MINUTO_INICIO,
                                   DURACION_MIN * 60, DATA_DIR)
sampfrom = int(MINUTO_INICIO * 60 * 100)   # para alinear .qrs despues
print(f'  fs             : {fs} Hz')
print(f'  muestras       : {len(ecg_crudo)}')
t = np.arange(len(ecg_crudo)) / fs


# =============================================================================
# ETAPA A: filtros generales
# =============================================================================

print()
print('=' * 70)
print(f'ETAPA A - Filtros del ECG (HP {FC_PASAALTOS} Hz + LP {FC_PASABAJOS} Hz)')
print('=' * 70)

ecg_filtrado = filtrar_ecg_general(ecg_crudo, fs)

# Respuesta en frecuencia
b_hp, a_hp = disenar_pasaaltos(FC_PASAALTOS, fs, ORDEN_BUTTER)
b_lp, a_lp = disenar_pasabajos(FC_PASABAJOS, fs, ORDEN_BUTTER)
w_hp, h_hp = sg.freqz(b_hp, a_hp, worN=2048, fs=fs)
w_lp, h_lp = sg.freqz(b_lp, a_lp, worN=2048, fs=fs)

plt.figure(figsize=(14, 4))
plt.semilogy(w_hp, np.abs(h_hp), label=f'HP {FC_PASAALTOS} Hz')
plt.semilogy(w_lp, np.abs(h_lp), label=f'LP {FC_PASABAJOS} Hz')
plt.axhline(1/np.sqrt(2), color='red', linestyle=':', alpha=0.4, label='-3 dB')
plt.xlabel('Frecuencia [Hz]'); plt.ylabel('|H(f)|')
plt.title('Respuesta en frecuencia de los filtros')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.xlim(0, fs/2); plt.ylim(1e-3, 2)
plt.tight_layout(); plt.show()

# Antes / despues en tiempo
i0v = 0
i1v = int(VENTANA_VIZ_SEG * fs)
fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
axes[0].plot(t[i0v:i1v], ecg_crudo[i0v:i1v], linewidth=0.7)
axes[0].set_ylabel('Crudo [mV]'); axes[0].grid(True, alpha=0.3)
axes[0].set_title(f'ECG antes y despues del filtrado general ({VENTANA_VIZ_SEG} s)')
axes[1].plot(t[i0v:i1v], ecg_filtrado[i0v:i1v], color='orange', linewidth=0.7)
axes[1].set_ylabel('Filtrado [mV]'); axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Tiempo [s]')
plt.tight_layout(); plt.show()

print(f'  std crudo / filtrado : {ecg_crudo.std():.4f} / {ecg_filtrado.std():.4f} mV')


# =============================================================================
# ETAPA B: Pan-Tompkins
# =============================================================================

print()
print('=' * 70)
print('ETAPA B - Pan-Tompkins')
print('=' * 70)
print(f'  bandpass            : {PT_BANDA_BAJA}-{PT_BANDA_ALTA} Hz')
print(f'  ventana integrador  : {PT_VENTANA_INT_MS} ms')
print(f'  refractario         : {PT_REFRACTARIO_MS} ms')
print(f'  guarda de borde     : {PT_GUARDA_BORDE_MS} ms en cada extremo')
print(f'  umbral adaptativo   : mediana + {PT_ADAPT_ALPHA} * (P99 - mediana)')

pt = pan_tompkins(ecg_filtrado, fs)

fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
axes[0].plot(t[i0v:i1v], ecg_filtrado[i0v:i1v]); axes[0].set_ylabel('ECG filtrado')
axes[1].plot(t[i0v:i1v], pt['bandpass'][i0v:i1v], color='C1')
axes[1].set_ylabel('1) Bandpass\n5-15 Hz')
axes[2].plot(t[i0v:i1v], pt['derivada'][i0v:i1v], color='C2')
axes[2].set_ylabel('2) Derivada')
axes[3].plot(t[i0v:i1v], pt['cuadrado'][i0v:i1v], color='C3')
axes[3].set_ylabel('3) Cuadrado')
axes[4].plot(t[i0v:i1v], pt['integrada'][i0v:i1v], color='C4')
axes[4].set_ylabel('4) Integrador\n150 ms')

mR = (pt['picos_R'] >= i0v) & (pt['picos_R'] < i1v)
mI = (pt['picos_int'] >= i0v) & (pt['picos_int'] < i1v)
axes[4].plot(t[pt['picos_int'][mI]], pt['integrada'][pt['picos_int'][mI]],
             'rv', label='picos sobre integrador')
axes[4].plot(t[i0v:i1v], pt['umbral'][i0v:i1v], 'k--', alpha=0.6,
             label='umbral local')
borde_seg = PT_GUARDA_BORDE_MS/1000
axes[4].axvspan(t[i0v], t[i0v] + borde_seg, alpha=0.15, color='gray')
axes[4].axvspan(t[i1v-1] - borde_seg, t[i1v-1], alpha=0.15, color='gray',
                label='guarda de borde')
axes[0].plot(t[pt['picos_R'][mR]], ecg_filtrado[pt['picos_R'][mR]],
             'ro', label='R refinados', markersize=6)
axes[0].legend(loc='upper right'); axes[4].legend(loc='upper right')
axes[-1].set_xlabel('Tiempo [s]')
for ax in axes: ax.grid(True, alpha=0.3)
fig.suptitle(f'Pan-Tompkins - etapas (primeros {VENTANA_VIZ_SEG} s)')
plt.tight_layout(); plt.show()

print(f'  QRS detectados en {DURACION_MIN} min : {len(pt["picos_R"])}')
print(f'  FC media estimada    : {len(pt["picos_R"])/DURACION_MIN:.1f} lpm')
print(f'  Umbral local         : {pt["umbral"].min():.1f} - {pt["umbral"].max():.1f} '
      f'(media {pt["umbral"].mean():.1f})')


# Comparacion contra .qrs de la base
print()
print('=' * 70)
print('Comparacion contra .qrs de la base')
print('=' * 70)

qrs_full = cargar_anotaciones_qrs(REGISTRO, DATA_DIR)
mask_seg = ((qrs_full >= sampfrom) & (qrs_full < sampfrom + len(ecg_crudo)))
qrs_ref = qrs_full[mask_seg] - sampfrom
comp = comparar_detecciones(pt['picos_R'], qrs_ref, fs)

print(f'  QRS de la base (.qrs) : {len(qrs_ref)}')
print(f'  Nuestro Pan-Tompkins  : {len(pt["picos_R"])}')
print(f'  TP / FP / FN          : {comp["TP"]} / {comp["FP"]} / {comp["FN"]}')
print(f'  Sensibilidad          : {100*comp["sensibilidad"]:.2f} %')
print(f'  Precision             : {100*comp["precision"]:.2f} %')


# =============================================================================
# ETAPA C: limpieza temporal de la serie RR
# =============================================================================

print()
print('=' * 70)
print('ETAPA C - Limpieza de la serie RR (filtros temporales)')
print('=' * 70)
print(f'  1) Rango fisiologico  : [{RR_MIN_FISIOL}, {RR_MAX_FISIOL}] s')
print(f'  2) Malik (Task Force) : |dRR|/RR_prev > {int(MALIK_UMBRAL*100)} %')
print(f'  3) Mediana local      : |RR - med_loc|/med_loc > {int(MEDIANA_UMBRAL*100)} %,'
      f' ventana {MEDIANA_VENTANA} RRs')

rr = serie_rr(pt['picos_R'], fs)
rr_interp, flags = limpiar_rr(rr)

n_r = int(flags['rango'].sum())
n_m = int(flags['malik'].sum())
n_med = int(flags['mediana'].sum())
n_total = int(flags['total'].sum())

print(f'\n  RR totales              : {len(rr)}')
print(f'  RR mediano (crudo)      : {np.median(rr):.3f} s '
      f'(FC = {60/np.median(rr):.1f} lpm)')
print(f'  std (crudo)             : {rr.std():.3f} s')
print(f'\n  Marcados por cada criterio:')
print(f'    rango fisiologico   : {n_r:4d}  ({100*n_r/len(rr):.2f} %)')
print(f'    Malik               : {n_m:4d}  ({100*n_m/len(rr):.2f} %)')
print(f'    mediana local       : {n_med:4d}  ({100*n_med/len(rr):.2f} %)')
print(f'    UNION (OR)          : {n_total:4d}  ({100*n_total/len(rr):.2f} %)')
print(f'\n  RR mediano (limpio)     : {np.median(rr_interp):.3f} s '
      f'(FC = {60/np.median(rr_interp):.1f} lpm)')
print(f'  std (limpio)            : {rr_interp.std():.3f} s')

# Tacograma antes / despues
t_rr = pt['picos_R'][1:] / fs
fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
axes[0].plot(t_rr/60, rr, '.-', markersize=2, linewidth=0.5, color='C0',
             alpha=0.6, label='RR crudos')
if n_r > 0:
    axes[0].plot(t_rr[flags['rango']]/60, rr[flags['rango']], 'rx', markersize=10,
                 label=f'fuera de rango (n={n_r})')
if n_m > 0:
    axes[0].plot(t_rr[flags['malik']]/60, rr[flags['malik']], 'o', markersize=8,
                 markerfacecolor='none', markeredgecolor='orange',
                 markeredgewidth=2, label=f'Malik (n={n_m})')
if n_med > 0:
    axes[0].plot(t_rr[flags['mediana']]/60, rr[flags['mediana']], 's', markersize=8,
                 markerfacecolor='none', markeredgecolor='green',
                 markeredgewidth=2, label=f'mediana local (n={n_med})')
axes[0].set_ylabel('RR [s]')
axes[0].set_title(f'Serie RR cruda con outliers marcados por criterio '
                  f'(union: n={n_total})')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[1].plot(t_rr/60, rr_interp, '.-', markersize=2, linewidth=0.5, color='C2')
axes[1].set_xlabel('Tiempo [min]')
axes[1].set_ylabel('RR [s]')
axes[1].set_title('Serie RR limpia (interpolacion lineal en los outliers)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()


# =============================================================================
# Resumen
# =============================================================================

print()
print('=' * 70)
print('Resumen')
print('=' * 70)
print(f'  Registro                : {REGISTRO}')
print(f'  Segmento                : {DURACION_MIN} min desde el min {MINUTO_INICIO}')
print(f'  QRS detectados (PT)     : {len(pt["picos_R"])}')
print(f'  Sens / Prec vs .qrs     : '
      f'{100*comp["sensibilidad"]:.1f} % / {100*comp["precision"]:.1f} %')
print(f'  RR outliers (union)     : {n_total} ({100*n_total/len(rr):.2f} %)')
print(f'  RR finales (limpios)    : {len(rr_interp)}')
print(f'  FC mediana (limpia)     : {60/np.median(rr_interp):.1f} lpm')
print('=' * 70)