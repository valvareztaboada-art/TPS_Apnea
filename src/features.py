# -*- coding: utf-8 -*-
"""
src/features.py
================

Calculo de features HRV por minuto, listo para detectar apnea.

Las features siguen el estandar Task Force ESC/NASPE 1996 mas algunas
especificas de la firma espectral de la apnea (CVHR).

- Time-domain: ventana de 1 minuto (alineada con las anotaciones .apn que son
  por minuto).
- Frequency-domain: ventana de 5 minutos centrada (1 min es demasiado corto
  para resolver la banda LF que arranca en 0.04 Hz = 25 s de periodo).
- PSD calculada con Lomb-Scargle (estandar para series no uniformemente
  sampleadas como RR).
"""

import numpy as np
import pandas as pd
from scipy.signal import lombscargle


# =============================================================================
# Constantes: bandas espectrales
# =============================================================================

# Bandas estandar Task Force ESC/NASPE 1996
BAND_VLF = (0.0033, 0.04)    # Very Low Frequency
BAND_LF = (0.04, 0.15)        # Low Frequency  (mayormente simpatico)
BAND_HF = (0.15, 0.40)        # High Frequency (mayormente parasimpatico, respiratorio)

# Banda CVHR (Cyclical Variation of Heart Rate) - firma espectral de la apnea
# Penzel 2000, Mendez 2010: los eventos apneicos duran 30-60 s -> ciclo
# de FC con frecuencia 0.017-0.033 Hz. Usamos 0.01-0.04 Hz para ser
# inclusivos sin solapar con LF.
BAND_CVHR = (0.01, 0.04)


# =============================================================================
# Features time-domain
# =============================================================================

def features_tiempo(rr):
    """Features HRV en el dominio del tiempo a partir de RRs (en segundos).

    Devuelve un dict. Si no hay suficientes datos (< 2 latidos), llena con NaN.
    """
    n = len(rr)
    if n < 2:
        return {
            'n_beats': n,
            'mean_rr': np.nan, 'sdnn': np.nan, 'rmssd': np.nan,
            'nn50': 0, 'pnn50': np.nan,
            'mean_hr': np.nan, 'sd_hr': np.nan,
        }

    drr = np.diff(rr)
    n50 = int(np.sum(np.abs(drr) > 0.050))

    return {
        'n_beats': n,
        'mean_rr': float(np.mean(rr)),
        'sdnn': float(np.std(rr, ddof=1)),
        'rmssd': float(np.sqrt(np.mean(drr ** 2))),
        'nn50': n50,
        'pnn50': float(100 * n50 / len(drr)),
        'mean_hr': float(60 / np.mean(rr)),
        'sd_hr': float(np.std(60 / rr, ddof=1)),
    }


# =============================================================================
# Features frequency-domain (Lomb-Scargle)
# =============================================================================

def lomb_psd(rr, t_rr, f_min=0.003, f_max=0.5, n_freqs=256):
    """Periodograma de Lomb-Scargle de la serie RR.

    Lomb-Scargle es la opcion estandar para series no uniformemente sampleadas
    (cada RR viene a un instante distinto). Devuelve (f, psd) en Hz / unidades
    arbitrarias de potencia.

    Si hay muy pocos datos o el span temporal es chico, devuelve arrays vacios.
    """
    if len(rr) < 4 or (t_rr[-1] - t_rr[0]) < 30:
        return np.array([]), np.array([])

    y = rr - np.mean(rr)
    f = np.linspace(f_min, f_max, n_freqs)
    omega = 2 * np.pi * f
    try:
        pgram = lombscargle(t_rr, y, omega, normalize=False)
        return f, pgram
    except Exception:
        return np.array([]), np.array([])


def band_power(f, psd, f_low, f_high):
    """Integra la PSD entre [f_low, f_high]."""
    if len(f) == 0:
        return np.nan
    mask = (f >= f_low) & (f <= f_high)
    if not mask.any():
        return np.nan
    integrar = getattr(np, 'trapezoid', None) or np.trapz
    return float(integrar(psd[mask], f[mask]))


def features_frecuencia(rr, t_rr):
    """Features HRV en el dominio de la frecuencia.

    Calcula la PSD por Lomb-Scargle y extrae potencias en bandas VLF, LF, HF
    y CVHR, mas ratios derivados.
    """
    nan_result = {
        'vlf_power': np.nan, 'lf_power': np.nan, 'hf_power': np.nan,
        'total_power': np.nan,
        'lf_hf_ratio': np.nan, 'lf_norm': np.nan, 'hf_norm': np.nan,
        'cvhr_power': np.nan, 'cvhr_norm': np.nan,
    }
    if len(rr) < 4:
        return nan_result

    f, psd = lomb_psd(rr, t_rr)
    if len(f) == 0:
        return nan_result

    vlf = band_power(f, psd, *BAND_VLF)
    lf = band_power(f, psd, *BAND_LF)
    hf = band_power(f, psd, *BAND_HF)

    if any(np.isnan(x) for x in (vlf, lf, hf)):
        return nan_result

    total = vlf + lf + hf
    lf_hf = lf / hf if hf > 0 else np.nan
    sum_lf_hf = lf + hf
    lf_norm = lf / sum_lf_hf if sum_lf_hf > 0 else np.nan
    hf_norm = hf / sum_lf_hf if sum_lf_hf > 0 else np.nan

    cvhr = band_power(f, psd, *BAND_CVHR)
    cvhr_norm = cvhr / total if total > 0 else np.nan

    return {
        'vlf_power': vlf, 'lf_power': lf, 'hf_power': hf,
        'total_power': total,
        'lf_hf_ratio': lf_hf, 'lf_norm': lf_norm, 'hf_norm': hf_norm,
        'cvhr_power': cvhr, 'cvhr_norm': cvhr_norm,
    }


# =============================================================================
# Funcion principal: features para cada minuto del registro
# =============================================================================

def features_por_minuto(picos_R, rr_interp, fs, duracion_s,
                         ventana_freq_seg=300):
    """Calcula features HRV para cada minuto de un registro.

    Parameters
    ----------
    picos_R : np.ndarray
        Indices de muestra de cada R detectado.
    rr_interp : np.ndarray
        Serie RR limpia + interpolada, en segundos.
        Largo: len(picos_R) - 1.
    fs : int
        Frecuencia de muestreo.
    duracion_s : float
        Duracion total del registro en segundos.
    ventana_freq_seg : int
        Tamano (en segundos) de la ventana CENTRADA usada para el calculo
        de features espectrales. Default 300 (5 min).

    Returns
    -------
    pd.DataFrame con una fila por minuto, columna 'minute' como indice
    posicional (no se setea como indice de DataFrame). Cada fila tiene
    todas las features de tiempo y frecuencia.
    """
    t_rr = picos_R[1:] / fs       # tiempo (s) del final de cada RR
    n_minutos = int(duracion_s // 60)

    rows = []
    for m in range(n_minutos):
        # Ventana de 1 min (alineada con .apn): minuto m
        mask_t = (t_rr >= m * 60) & (t_rr < (m + 1) * 60)
        rr_t = rr_interp[mask_t]

        # Ventana centrada de ventana_freq_seg segundos para espectro
        t_centro = (m + 0.5) * 60
        mask_f = ((t_rr >= t_centro - ventana_freq_seg / 2)
                  & (t_rr < t_centro + ventana_freq_seg / 2))
        rr_f = rr_interp[mask_f]
        t_rr_f = t_rr[mask_f]

        rows.append({
            'minute': m,
            **features_tiempo(rr_t),
            **features_frecuencia(rr_f, t_rr_f),
        })

    return pd.DataFrame(rows)