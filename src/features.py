# -*- coding: utf-8 -*-
"""
Funciones para calcular features HRV y EDR por minuto 
================

Calculo de features HRV por minuto, listo para detectar apnea.

- Time-domain: ventana de 1 minuto (alineada con las anotaciones .apn que son
  por minuto).
- Frequency-domain: ventana de 5 minutos centrada (1 min es demasiado corto
  para resolver la banda LF que arranca en 0.04 Hz = 25 s de periodo).
- PSD calculada con Lomb-Scargle (estandar para series no uniformemente
  sampleadas como RR).
  - EDR (ECG-Derived Respiration): a partir de la amplitud de los picos R se
  extrae una señal sustituta de la respiración. Se calcula su PSD (también
  por Lomb-Scargle) y se obtienen potencias en la banda respiratoria normal
  (0.15-0.40 Hz) y en una banda de modulación lenta asociada a apnea
  (0.01-0.04 Hz), junto con sus versiones normalizadas y el ratio entre
  ambas.
"""

import numpy as np
import pandas as pd
from scipy.signal import lombscargle


# =============================================================================
# Constantes: bandas espectrales
# =============================================================================

# Bandas estandar 
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

    Devuelve (f, psd) en Hz / unidades arbitrarias de potencia.
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
# EDR (ECG-Derived Respiration)
# =============================================================================

# Bandas en la PSD del EDR
BAND_EDR_RESP = (0.15, 0.40)    # respiratoria normal
BAND_EDR_APNEA = (0.01, 0.04)   # modulacion apneica lenta


def amplitudes_R(ecg_filtrado, picos_R, fs, ventana_ms=25):
    """Calcula la amplitud del ECG en cada R detectado.

    Toma el maximo en una ventana de +-ventana_ms alrededor de cada pico,
    para tolerar pequenas desalineaciones del detector.

    Returns
    -------
    np.ndarray
        Amplitudes (una por R), mismo largo que picos_R.
    """
    n = int(ventana_ms * fs / 1000)
    amps = np.zeros(len(picos_R), dtype=float)
    L = len(ecg_filtrado)
    for i, r in enumerate(picos_R):
        s = max(0, r - n)
        e = min(L, r + n + 1)
        amps[i] = float(np.max(ecg_filtrado[s:e]))
    return amps


def features_edr(amplitudes, t_picos):
    """Features espectrales del EDR.

    Parameters
    ----------
    amplitudes : np.ndarray
        Serie de amplitudes de R en la ventana.
    t_picos : np.ndarray
        Tiempo (s) de cada R en la ventana. Mismo largo que amplitudes.

    Returns
    -------
    dict
        edr_resp_power, edr_apnea_power, edr_resp_norm, edr_apnea_norm,
        edr_apnea_resp_ratio. Si hay datos insuficientes devuelve NaNs.
    """
    nan_result = {
        'edr_resp_power': np.nan,
        'edr_apnea_power': np.nan,
        'edr_resp_norm': np.nan,
        'edr_apnea_norm': np.nan,
        'edr_apnea_resp_ratio': np.nan,
    }
    if len(amplitudes) < 10:
        return nan_result

    f, psd = lomb_psd(amplitudes, t_picos, f_min=0.005, f_max=0.5, n_freqs=256)
    if len(f) == 0:
        return nan_result

    resp = band_power(f, psd, *BAND_EDR_RESP)
    apnea = band_power(f, psd, *BAND_EDR_APNEA)
    total = band_power(f, psd, 0.005, 0.5)

    if np.isnan(total) or total <= 0:
        return nan_result

    resp_norm = resp / total if not np.isnan(resp) else np.nan
    apnea_norm = apnea / total if not np.isnan(apnea) else np.nan
    apnea_resp_ratio = (apnea / resp) if (not np.isnan(resp) and resp > 0
                                            and not np.isnan(apnea)) else np.nan

    return {
        'edr_resp_power': resp,
        'edr_apnea_power': apnea,
        'edr_resp_norm': resp_norm,
        'edr_apnea_norm': apnea_norm,
        'edr_apnea_resp_ratio': apnea_resp_ratio,
    }


# =============================================================================
# Funcion principal: features para cada minuto del registro
# =============================================================================

def features_por_minuto(picos_R, rr_interp, fs, duracion_s,
                         amplitudes_picos=None,
                         ventana_freq_seg=300):
    """Calcula features HRV (y opcionalmente EDR) para cada minuto.

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
    amplitudes_picos : np.ndarray, opcional
        Amplitud del ECG en cada R (mismo largo que picos_R). Si se pasa,
        se calculan tambien features EDR por minuto. Si es None, EDR queda
        sin computar (las columnas no aparecen en el output).
    ventana_freq_seg : int
        Tamano (en segundos) de la ventana CENTRADA usada para el calculo
        de features espectrales (HRV y EDR). Default 300 (5 min).

    Returns
    -------
    pd.DataFrame con una fila por minuto.
    """
    t_rr = picos_R[1:] / fs       # tiempo (s) del final de cada RR
    t_R = picos_R / fs            # tiempo (s) de cada R
    n_minutos = int(duracion_s // 60)
    incluir_edr = amplitudes_picos is not None

    rows = []
    for m in range(n_minutos):
        # Ventana de 1 min (alineada con .apn): minuto m
        mask_t = (t_rr >= m * 60) & (t_rr < (m + 1) * 60)
        rr_t = rr_interp[mask_t]

        # Ventana centrada de ventana_freq_seg segundos para espectro HRV
        t_centro = (m + 0.5) * 60
        t_lo = t_centro - ventana_freq_seg / 2
        t_hi = t_centro + ventana_freq_seg / 2
        mask_f = (t_rr >= t_lo) & (t_rr < t_hi)
        rr_f = rr_interp[mask_f]
        t_rr_f = t_rr[mask_f]

        fila = {
            'minute': m,
            **features_tiempo(rr_t),
            **features_frecuencia(rr_f, t_rr_f),
        }

        # EDR: misma ventana centrada de 5 min, sobre las amplitudes de R
        if incluir_edr:
            mask_R_f = (t_R >= t_lo) & (t_R < t_hi)
            amps_f = amplitudes_picos[mask_R_f]
            t_R_f = t_R[mask_R_f]
            fila.update(features_edr(amps_f, t_R_f))

        rows.append(fila)

    return pd.DataFrame(rows)