# -*- coding: utf-8 -*-
"""
src/pipeline.py
================

Pipeline reutilizable para Apnea-ECG: filtros del ECG + Pan-Tompkins +
limpieza temporal de la serie RR. Sin side effects (no plots, no prints,
no I/O de archivos al disco). Listo para ser usado desde:

  - scripts batch (procesar_registro sobre los 70 sujetos)
  - scripts exploratorios (03_preprocesamiento_y_qrs.py)
  - una interfaz PySide6 (que carga el cache o procesa una senal subida)

Decisiones tecnicas y justificacion: ver scripts 02_analisis_espectral.py
y 03_preprocesamiento_y_qrs.py.
"""

import os
import numpy as np
import scipy.signal as sg
import wfdb


# =============================================================================
# Constantes del pipeline
# =============================================================================

# Filtros generales
FC_PASAALTOS = 0.5
FC_PASABAJOS = 40.0
ORDEN_BUTTER = 4

# Pan-Tompkins
PT_BANDA_BAJA = 5.0
PT_BANDA_ALTA = 15.0
PT_ORDEN_BANDA = 2
PT_VENTANA_INT_MS = 150
PT_REFRACTARIO_MS = 200
PT_ADAPT_ALPHA = 0.3
PT_GUARDA_BORDE_MS = 200

# Filtros temporales sobre RR
RR_MIN_FISIOL = 0.3
RR_MAX_FISIOL = 2.0
MALIK_UMBRAL = 0.20
MEDIANA_VENTANA = 5
MEDIANA_UMBRAL = 0.30


# =============================================================================
# Etapa A: filtros del ECG
# =============================================================================

def disenar_pasaaltos(fc, fs, orden=ORDEN_BUTTER):
    return sg.butter(orden, fc/(fs/2), btype='highpass')


def disenar_pasabajos(fc, fs, orden=ORDEN_BUTTER):
    return sg.butter(orden, fc/(fs/2), btype='lowpass')


def filtrar_ecg_general(ecg, fs,
                        fc_hp=FC_PASAALTOS, fc_lp=FC_PASABAJOS,
                        orden=ORDEN_BUTTER):
    """HP + LP Butterworth con filtfilt (fase cero, no corre los QRS)."""
    b_hp, a_hp = disenar_pasaaltos(fc_hp, fs, orden)
    b_lp, a_lp = disenar_pasabajos(fc_lp, fs, orden)
    return sg.filtfilt(b_lp, a_lp, sg.filtfilt(b_hp, a_hp, ecg))


# =============================================================================
# Etapa B: Pan-Tompkins
# =============================================================================

def pt_paso1_bandpass(x, fs, f_low=PT_BANDA_BAJA, f_high=PT_BANDA_ALTA,
                       orden=PT_ORDEN_BANDA):
    b, a = sg.butter(orden, [f_low/(fs/2), f_high/(fs/2)], btype='band')
    return sg.filtfilt(b, a, x)


def pt_paso2_derivada(x, fs):
    """Derivada centrada con kernel de Pan-Tompkins (1/8)*[1,2,0,-2,-1]."""
    h = np.array([1, 2, 0, -2, -1]) * fs / 8.0
    return np.convolve(x, h, mode='same')


def pt_paso3_cuadrado(x):
    return x ** 2


def pt_paso4_integrador(x, fs, ventana_ms=PT_VENTANA_INT_MS):
    N = max(1, int(ventana_ms * fs / 1000))
    return np.convolve(x, np.ones(N)/N, mode='same')


def pt_detectar_picos(integrada, fs, refractario_ms=PT_REFRACTARIO_MS,
                       alpha=PT_ADAPT_ALPHA,
                       guarda_borde_ms=PT_GUARDA_BORDE_MS):
    """Deteccion adaptativa: umbral = mediana + alpha * (P99 - mediana).
    Enmascara los bordes para evitar artefactos de convolucion."""
    distancia = int(refractario_ms * fs / 1000)
    borde = int(guarda_borde_ms * fs / 1000)
    noise = float(np.median(integrada))
    peak = float(np.percentile(integrada, 99))
    altura = noise + alpha * (peak - noise)
    integrada_mask = integrada.copy()
    if borde > 0:
        integrada_mask[:borde] = -np.inf
        integrada_mask[-borde:] = -np.inf
    picos, _ = sg.find_peaks(integrada_mask, distance=distancia, height=altura)
    return picos, altura


def pt_refinar_a_R(picos_int, ecg_filtrado, fs, ventana_ms=75):
    """Para cada pico sobre la integrada, busca max local del ECG en +-75 ms."""
    half_w = int(ventana_ms * fs / 1000)
    picos_R = []
    for p in picos_int:
        i0 = max(0, p - half_w)
        i1 = min(len(ecg_filtrado), p + half_w + 1)
        picos_R.append(i0 + int(np.argmax(ecg_filtrado[i0:i1])))
    return np.array(picos_R, dtype=int)


def pan_tompkins(ecg_filtrado, fs):
    """Pan-Tompkins completo. Devuelve dict con las 4 etapas + picos R."""
    bp = pt_paso1_bandpass(ecg_filtrado, fs)
    der = pt_paso2_derivada(bp, fs)
    cua = pt_paso3_cuadrado(der)
    integ = pt_paso4_integrador(cua, fs)
    picos_int, umbral = pt_detectar_picos(integ, fs)
    picos_R = pt_refinar_a_R(picos_int, ecg_filtrado, fs)
    return {'bandpass': bp, 'derivada': der, 'cuadrado': cua,
            'integrada': integ, 'picos_int': picos_int, 'picos_R': picos_R,
            'umbral': umbral}


def comparar_detecciones(picos_propios, picos_referencia, fs, tol_ms=100):
    """TP / FP / FN con tolerancia temporal."""
    tol = int(tol_ms * fs / 1000)
    referencia = np.asarray(picos_referencia)
    propios = np.asarray(picos_propios)
    matched_ref = np.zeros(len(referencia), dtype=bool)
    matched_propio = np.zeros(len(propios), dtype=bool)
    for i, r in enumerate(referencia):
        if len(propios) == 0:
            break
        diffs = np.abs(propios - r)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol and not matched_propio[j]:
            matched_ref[i] = True
            matched_propio[j] = True
    TP = int(matched_ref.sum())
    FN = int((~matched_ref).sum())
    FP = int((~matched_propio).sum())
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return {'TP': TP, 'FP': FP, 'FN': FN,
            'sensibilidad': sens, 'precision': prec}


# =============================================================================
# Etapa C: filtros temporales sobre la serie RR
# =============================================================================

def serie_rr(picos, fs):
    return np.diff(picos) / fs


def filtro_rango_fisiologico(rr, rr_min=RR_MIN_FISIOL, rr_max=RR_MAX_FISIOL):
    return (rr < rr_min) | (rr > rr_max)


def filtro_malik(rr, umbral=MALIK_UMBRAL):
    """Filtro de cambio relativo (Task Force ESC/NASPE 1996)."""
    out = np.zeros(len(rr), dtype=bool)
    for i in range(1, len(rr)):
        if rr[i-1] > 0 and abs(rr[i] - rr[i-1]) / rr[i-1] > umbral:
            out[i] = True
    return out


def filtro_mediana_local(rr, ventana=MEDIANA_VENTANA, umbral=MEDIANA_UMBRAL):
    """Outliers respecto a la mediana de la ventana local (excluyendo el propio)."""
    out = np.zeros(len(rr), dtype=bool)
    half = ventana // 2
    for i in range(len(rr)):
        i0 = max(0, i - half)
        i1 = min(len(rr), i + half + 1)
        idx_vecinos = list(range(i0, i)) + list(range(i+1, i1))
        if len(idx_vecinos) < 2:
            continue
        m = float(np.median(rr[idx_vecinos]))
        if m > 0 and abs(rr[i] - m) / m > umbral:
            out[i] = True
    return out


def interpolar_nan(rr):
    rr_out = rr.copy().astype(float)
    nans = np.isnan(rr_out)
    if not nans.any() or nans.all():
        return rr_out
    idx = np.arange(len(rr_out))
    rr_out[nans] = np.interp(idx[nans], idx[~nans], rr_out[~nans])
    return rr_out


def limpiar_rr(rr):
    """Aplica los tres filtros y devuelve la serie limpia + los flags.

    Returns
    -------
    rr_interp : np.ndarray con NaN reemplazados por interpolacion lineal
    flags : dict con 'rango', 'malik', 'mediana' y 'total' (bool arrays)
    """
    flag_rango = filtro_rango_fisiologico(rr)
    flag_malik = filtro_malik(rr)
    flag_mediana = filtro_mediana_local(rr)
    flag_total = flag_rango | flag_malik | flag_mediana

    rr_nan = rr.copy().astype(float)
    rr_nan[flag_total] = np.nan
    rr_interp = interpolar_nan(rr_nan)

    return rr_interp, {
        'rango': flag_rango,
        'malik': flag_malik,
        'mediana': flag_mediana,
        'total': flag_total,
    }


# =============================================================================
# Funciones de I/O y de alto nivel
# =============================================================================

def cargar_ecg(record_name, data_dir, sampfrom=0, sampto=None):
    """Carga el ECG y los metadatos del registro.

    Returns
    -------
    ecg : 1D array
    fs : int
    fields : dict (header de wfdb)
    """
    path = os.path.join(data_dir, record_name)
    if sampto is None:
        signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom)
    else:
        signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto)
    return signal[:, 0], fields['fs'], fields


def cargar_segmento(record_name, minuto, duracion_seg, data_dir):
    """Wrapper convenience: carga `duracion_seg` segundos desde `minuto`."""
    fs_aprox = 100
    sampfrom = int(minuto * 60 * fs_aprox)
    sampto = sampfrom + int(duracion_seg * fs_aprox)
    return cargar_ecg(record_name, data_dir,
                       sampfrom=sampfrom, sampto=sampto)


def cargar_anotaciones_qrs(record_name, data_dir):
    """Carga las anotaciones .qrs (machine-generated). Devuelve array de
    indices de muestra. Lanza FileNotFoundError si no existe."""
    path = os.path.join(data_dir, record_name)
    ann = wfdb.rdann(path, 'qrs')
    return ann.sample


def cargar_anotaciones_apn(record_name, data_dir):
    """Carga las anotaciones .apn (apnea por minuto). Devuelve (samples, symbols).
    Solo existen para el learning set (a*, b*, c*). Lanza FileNotFoundError
    para los test (x*)."""
    path = os.path.join(data_dir, record_name)
    ann = wfdb.rdann(path, 'apn')
    return ann.sample, np.array(ann.symbol)


def procesar_registro(record_name, data_dir, sampfrom=0, sampto=None,
                      comparar_qrs=True):
    """Pipeline completo sobre un registro.

    Carga el ECG (o segmento), aplica filtros generales, Pan-Tompkins, y
    limpieza temporal de la serie RR.

    Parameters
    ----------
    record_name : str         (ej 'a01')
    data_dir : str
    sampfrom, sampto : int    para procesar solo un segmento (default todo)
    comparar_qrs : bool       si True intenta comparar contra .qrs de la base

    Returns
    -------
    dict con:
      'fs'           : frecuencia de muestreo (int)
      'duracion_s'   : duracion del segmento procesado (float)
      'sampfrom'     : offset usado al cargar
      'ecg_crudo'    : senal cruda
      'ecg_filtrado' : senal despues de HP+LP
      'pt'           : dict completo de pan_tompkins (incluye picos_R)
      'rr_crudo'     : serie RR antes de limpiar
      'rr_interp'    : serie RR despues de limpieza + interpolacion
      'flags'        : dict con 'rango', 'malik', 'mediana', 'total'
      'comparacion_qrs' : dict con TP/FP/FN/sens/prec (si comparar_qrs=True
                          y existen .qrs)
    """
    ecg_crudo, fs, _ = cargar_ecg(record_name, data_dir,
                                   sampfrom=sampfrom, sampto=sampto)
    ecg_filtrado = filtrar_ecg_general(ecg_crudo, fs)
    pt = pan_tompkins(ecg_filtrado, fs)
    rr = serie_rr(pt['picos_R'], fs)
    rr_interp, flags = limpiar_rr(rr)

    result = {
        'fs': fs,
        'duracion_s': len(ecg_crudo) / fs,
        'sampfrom': sampfrom,
        'ecg_crudo': ecg_crudo,
        'ecg_filtrado': ecg_filtrado,
        'pt': pt,
        'rr_crudo': rr,
        'rr_interp': rr_interp,
        'flags': flags,
    }

    if comparar_qrs:
        try:
            qrs_ref = cargar_anotaciones_qrs(record_name, data_dir)
            # alinear al frame del segmento si procesamos un trozo
            if sampto is not None:
                m = (qrs_ref >= sampfrom) & (qrs_ref < sampto)
                qrs_ref = qrs_ref[m] - sampfrom
            result['comparacion_qrs'] = comparar_detecciones(
                pt['picos_R'], qrs_ref, fs)
            result['qrs_ref'] = qrs_ref
        except (FileNotFoundError, Exception):
            result['comparacion_qrs'] = None
            result['qrs_ref'] = None

    return result


# =============================================================================
# Utilidades para batch
# =============================================================================

def listar_registros(data_dir, incluir_respiracion=False):
    """Lista los registros principales (no los rNNr de respiracion).

    Returns
    -------
    list ordenada de nombres base (ej ['a01', 'a02', ..., 'x35'])
    """
    archivos_hea = sorted(f for f in os.listdir(data_dir) if f.endswith('.hea'))
    registros = []
    for f in archivos_hea:
        nombre = f[:-4]
        # filtrar archivos auxiliares de respiracion (rNNr y rNNer)
        if not incluir_respiracion and (nombre.endswith('r') or nombre.endswith('er')):
            continue
        registros.append(nombre)
    return sorted(registros)


def clasificar_grupo(record_name):
    """Devuelve 'apnea', 'borderline', 'control', 'test' u 'otros'."""
    if not record_name:
        return 'otros'
    c = record_name[0].lower()
    return {'a': 'apnea', 'b': 'borderline',
            'c': 'control', 'x': 'test'}.get(c, 'otros')


def resumen_registro(record_name, result):
    """Construye un dict resumen con metricas escalares de un registro
    procesado, listo para agregar a una tabla pandas."""
    rr = result['rr_crudo']
    rr_interp = result['rr_interp']
    flags = result['flags']
    n_rr = len(rr)

    fila = {
        'record': record_name,
        'grupo': clasificar_grupo(record_name),
        'duracion_h': result['duracion_s'] / 3600,
        'fs': result['fs'],
        'n_qrs': len(result['pt']['picos_R']),
        'fc_mediana_lpm': float(60 / np.median(rr_interp)) if len(rr_interp) > 0 else None,
        'rr_mediano_s': float(np.median(rr_interp)) if len(rr_interp) > 0 else None,
        'rr_std_s': float(np.std(rr_interp)) if len(rr_interp) > 0 else None,
        'n_outliers_rango': int(flags['rango'].sum()),
        'n_outliers_malik': int(flags['malik'].sum()),
        'n_outliers_mediana': int(flags['mediana'].sum()),
        'n_outliers_total': int(flags['total'].sum()),
        'pct_outliers': float(100 * flags['total'].sum() / n_rr) if n_rr > 0 else None,
    }

    comp = result.get('comparacion_qrs')
    if comp is not None:
        fila.update({
            'qrs_sens_pct': 100 * comp['sensibilidad'],
            'qrs_prec_pct': 100 * comp['precision'],
            'qrs_tp': comp['TP'],
            'qrs_fp': comp['FP'],
            'qrs_fn': comp['FN'],
        })
    else:
        fila.update({
            'qrs_sens_pct': None, 'qrs_prec_pct': None,
            'qrs_tp': None, 'qrs_fp': None, 'qrs_fn': None,
        })
    return fila
