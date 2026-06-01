# -*- coding: utf-8 -*-
"""
03b_procesar_todos.py
======================

Procesa todos los registros de la base Apnea-ECG aplicando el pipeline
completo (filtros + Pan-Tompkins + limpieza de RR). Para cada sujeto guarda
un .npz en la carpeta cache/ con los resultados, y al final escribe
cache/resumen.csv con metricas escalares por sujeto.

Salida que se guarda por sujeto (cache/<record>.npz):
  - fs               : frecuencia de muestreo
  - duracion_s       : duracion del registro
  - picos_R          : indices de muestra de cada R detectado
  - rr_crudo         : serie RR original (en segundos)
  - rr_interp        : serie RR limpia + interpolada
  - flag_rango       : booleano, true si el RR fue marcado por rango
  - flag_malik       : booleano, true si fue marcado por Malik
  - flag_mediana     : booleano, true si fue marcado por mediana local
  - flag_total       : OR de los anteriores
  - qrs_ref          : (si existen) anotaciones .qrs de la base

No se guarda la senal cruda ni la filtrada (~20 MB por sujeto). Cuando se
necesita (en la GUI o exploracion), se recalcula con pipeline.filtrar_ecg_general
(es rapido: < 1 s por sujeto).

Uso:
    python 03b_procesar_todos.py

Opcional: para procesar solo un subconjunto, modificar SUBSET abajo.
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

# permitir importar src.pipeline sin importar desde donde se corra
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.pipeline import (
    procesar_registro,
    listar_registros,
    clasificar_grupo,
    resumen_registro,
)


# =============================================================================
# Configuracion
# =============================================================================

DATA_DIR = 'apnea-ecg-database-1.0.0'
CACHE_DIR = 'cache'

# Para correr solo un subconjunto (debug), poner ej ['a01', 'a02', 'c01'].
# Para procesar todo dejar en None.
SUBSET = None


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    registros = listar_registros(DATA_DIR)
    if SUBSET is not None:
        registros = [r for r in registros if r in SUBSET]

    print(f'Registros a procesar: {len(registros)}')
    print(f'Cache dir: {CACHE_DIR}')
    print()

    filas_resumen = []
    t_inicio = time.time()

    for i, r in enumerate(registros, 1):
        t0 = time.time()
        print(f'[{i:2d}/{len(registros)}] {r:5s} ...', end=' ', flush=True)
        try:
            result = procesar_registro(r, DATA_DIR, comparar_qrs=True)

            # guardar arrays escenciales en .npz comprimido
            np.savez_compressed(
                os.path.join(CACHE_DIR, f'{r}.npz'),
                fs=result['fs'],
                duracion_s=result['duracion_s'],
                picos_R=result['pt']['picos_R'],
                rr_crudo=result['rr_crudo'],
                rr_interp=result['rr_interp'],
                flag_rango=result['flags']['rango'],
                flag_malik=result['flags']['malik'],
                flag_mediana=result['flags']['mediana'],
                flag_total=result['flags']['total'],
                qrs_ref=result.get('qrs_ref') if result.get('qrs_ref') is not None
                         else np.array([], dtype=int),
            )

            # acumular fila para resumen.csv
            fila = resumen_registro(r, result)
            fila['procesado_ok'] = True
            fila['error'] = None
            fila['tiempo_s'] = time.time() - t0
            filas_resumen.append(fila)

            n_qrs = len(result['pt']['picos_R'])
            pct = 100 * result['flags']['total'].sum() / max(1, len(result['rr_crudo']))
            print(f'ok ({n_qrs:5d} QRS, {pct:.2f}% outliers, '
                  f'{time.time()-t0:.1f}s)')
        except Exception as e:
            filas_resumen.append({
                'record': r,
                'grupo': clasificar_grupo(r),
                'procesado_ok': False,
                'error': str(e),
                'tiempo_s': time.time() - t0,
            })
            print(f'ERROR: {e}')
            traceback.print_exc(limit=2)

    # guardar resumen
    df = pd.DataFrame(filas_resumen)
    csv_path = os.path.join(CACHE_DIR, 'resumen.csv')
    df.to_csv(csv_path, index=False)

    print()
    print('=' * 70)
    print(f'Procesamiento completo en {time.time()-t_inicio:.1f} s')
    print(f'Resumen guardado en {csv_path}')
    print('=' * 70)

    # tabla resumida por grupo
    if df['procesado_ok'].any():
        ok = df[df['procesado_ok']].copy()
        print()
        print('Resumen por grupo:')
        agg = ok.groupby('grupo').agg(
            n_sujetos=('record', 'count'),
            duracion_h_media=('duracion_h', 'mean'),
            fc_mediana=('fc_mediana_lpm', 'mean'),
            pct_outliers_med=('pct_outliers', 'median'),
            pct_outliers_max=('pct_outliers', 'max'),
            qrs_sens=('qrs_sens_pct', 'mean'),
            qrs_prec=('qrs_prec_pct', 'mean'),
        ).round(2)
        print(agg.to_string())

        # alertas: sujetos con metricas raras
        print()
        print('Sujetos para revisar manualmente (outliers > 5% o sens < 95%):')
        sospechosos = ok[
            (ok['pct_outliers'] > 5) |
            (ok['qrs_sens_pct'].notna() & (ok['qrs_sens_pct'] < 95))
        ]
        if len(sospechosos) == 0:
            print('  ninguno, todo dentro de rangos esperados')
        else:
            cols = ['record', 'grupo', 'pct_outliers',
                    'qrs_sens_pct', 'qrs_prec_pct']
            print(sospechosos[cols].to_string(index=False))

    n_err = (~df['procesado_ok']).sum()
    if n_err > 0:
        print()
        print(f'ERRORES: {n_err} sujetos fallaron')
        print(df[~df['procesado_ok']][['record', 'error']].to_string(index=False))


if __name__ == '__main__':
    main()
