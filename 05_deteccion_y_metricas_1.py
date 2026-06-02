# -*- coding: utf-8 -*-
"""
05_deteccion_y_metricas.py
===========================

Deteccion de apnea por minuto usando 3 tecnicas basadas en thresholds y
distancias sobre las features HRV de 04_features_por_minuto.py.

Estrategia (train/test split simple):
  - LEARNING SET (35 sujetos a/b/c*, con labels por minuto y por sujeto):
    ajustar umbrales/centroides.
  - TEST SET (35 sujetos x*, distribucion esperada 20 A / 5 B / 10 C):
    aplicar los umbrales y validar que la distribucion sea razonable.

Tres tecnicas, ordenadas de mas simple a mas elaborada:
  T1: umbral unico sobre cvhr_norm (la feature mas discriminante).
  T2: regla AND sobre cvhr_norm Y lf_hf_ratio (dos features combinadas).
  T3: distancia a centroides de clase usando cvhr_norm, lf_hf_ratio, sdnn,
      rmssd (multivariada, similar al espiritu del K-means K=2 visto en
      clase, pero supervisado).

Bonus: ensemble por mayoria de votos de las tres tecnicas.

Evaluacion:
  - Per-minuto sobre learning set: confusion matrix, sens, spec, acc.
  - Per-sujeto sobre learning set: clase predicha vs real (matriz 3x3).
  - Per-sujeto sobre test set: distribucion predicha vs (20, 5, 10).

Lee:    cache/features.csv
Guarda: cache/clasificacion.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# =============================================================================
# Configuracion
# =============================================================================

CACHE_DIR = 'cache'

# Features usadas por cada tecnica
# T1 usa cvhr_power (la mas discriminante: A/N = 2.78x vs 1.56x de cvhr_norm)
T1_FEATURE = 'cvhr_power'
T2_FEATURES = ('cvhr_norm', 'lf_hf_ratio')
T3_FEATURES = ['cvhr_norm', 'lf_hf_ratio', 'sdnn', 'rmssd']

# Thresholds de AHI por defecto (los del CinC 2000 / Penzel 2000):
#   >= 100 minutos predichos como A -> clase A (apnea)
#   < 5 minutos predichos como A    -> clase C (control)
#   intermedio                      -> clase B (borderline)
# OJO: estos son para el conteo REAL de minutos apneicos. Como nuestro
# detector tiene falsos positivos, los conteos predichos viven en otra
# escala. Por eso CALIBRAMOS estos thresholds sobre el learning set
# (donde tenemos las clases reales) maximizando accuracy per-sujeto.
AHI_A_DEFAULT = 100
AHI_C_DEFAULT = 5

# Distribucion esperada en test set (Penzel 2000)
TEST_DIST_ESPERADA = {'A': 20, 'B': 5, 'C': 10}


# =============================================================================
# Helpers genericos
# =============================================================================

def metricas_binarias(y_true, y_pred):
    """Calcula metricas estandar de clasificacion binaria."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    n = TP + TN + FP + FN
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    acc = (TP + TN) / n if n > 0 else 0.0
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'sens': sens, 'spec': spec, 'acc': acc,
        'youden': sens + spec - 1,
    }


def encontrar_umbral_1d(valores, y_true, n_grid=200):
    """Busca el threshold T (en el rango [P1, P99] de los valores) que
    maximiza el J de Youden = sens + spec - 1."""
    v = np.asarray(valores)
    y = np.asarray(y_true).astype(int)
    grid = np.linspace(np.percentile(v, 1), np.percentile(v, 99), n_grid)
    mejor_t = grid[0]
    mejor_j = -np.inf
    for t in grid:
        m = metricas_binarias(y, (v > t).astype(int))
        if m['youden'] > mejor_j:
            mejor_j = m['youden']
            mejor_t = t
    return float(mejor_t), float(mejor_j)


def encontrar_umbrales_2d(v1, v2, y_true, n_grid=40):
    """Grid search 2D para regla AND. Maximiza Youden's J."""
    v1 = np.asarray(v1); v2 = np.asarray(v2); y = np.asarray(y_true).astype(int)
    g1 = np.linspace(np.percentile(v1, 1), np.percentile(v1, 99), n_grid)
    g2 = np.linspace(np.percentile(v2, 1), np.percentile(v2, 99), n_grid)
    mejor_t1, mejor_t2 = g1[0], g2[0]
    mejor_j = -np.inf
    for t1 in g1:
        # vectorizado en t2
        for t2 in g2:
            y_pred = ((v1 > t1) & (v2 > t2)).astype(int)
            m = metricas_binarias(y, y_pred)
            if m['youden'] > mejor_j:
                mejor_j = m['youden']
                mejor_t1, mejor_t2 = t1, t2
    return float(mejor_t1), float(mejor_t2), float(mejor_j)


def clasificar_ahi(n_apnea, ahi_a, ahi_c):
    """Asigna clase A/B/C segun cantidad de minutos predichos como apnea."""
    if n_apnea >= ahi_a:
        return 'A'
    if n_apnea < ahi_c:
        return 'C'
    return 'B'


def calibrar_thresholds_ahi(ahis_learning, clases_reales_learning):
    """Encuentra (AHI_C, AHI_A) que maximizan accuracy per-sujeto en el
    learning set. Usa los propios valores observados como candidatos
    (grid impulsado por los datos, no arbitrario).

    Returns
    -------
    ahi_c, ahi_a : floats
        AHI_C < AHI_A. clase A si AHI >= ahi_a, C si < ahi_c, B intermedio.
    acc : float
        Accuracy per-sujeto en learning set.
    """
    candidatos = sorted(set(ahis_learning))
    n = len(ahis_learning)
    mejor_acc = 0.0
    mejor_c, mejor_a = candidatos[0], candidatos[-1]

    for t_c in candidatos:
        for t_a in candidatos:
            if t_c >= t_a:
                continue
            preds = ['A' if v >= t_a else ('C' if v < t_c else 'B')
                     for v in ahis_learning]
            acc = sum(p == r for p, r in zip(preds, clases_reales_learning)) / n
            if acc > mejor_acc:
                mejor_acc = acc
                mejor_c, mejor_a = t_c, t_a
    return float(mejor_c), float(mejor_a), float(mejor_acc)


def grupo_a_clase(grupo):
    """Mapea grupo del registro (apnea/borderline/control) a clase A/B/C."""
    return {'apnea': 'A', 'borderline': 'B', 'control': 'C'}.get(grupo, None)


# =============================================================================
# Main
# =============================================================================

def main():
    # ---------- Cargar features ----------
    feat_path = os.path.join(CACHE_DIR, 'features.csv')
    df = pd.read_csv(feat_path)
    print('=' * 70)
    print('Carga de features')
    print('=' * 70)
    print(f'  archivo : {feat_path}')
    print(f'  filas   : {len(df)}')
    print(f'  sujetos : {df["record"].nunique()}')
    print(f'  con label (learning set): {df["label"].notna().sum()}')

    # Separar learning vs test
    learning = df[df['label'].notna()].copy()
    test = df[df['label'].isna()].copy()

    # Quitamos minutos con NaN en las features que usaremos
    features_usadas = sorted(set([T1_FEATURE] + list(T2_FEATURES) + T3_FEATURES))
    learning_clean = learning.dropna(subset=features_usadas)
    y_train = (learning_clean['label'] == 'A').astype(int).values

    print(f'\n  learning post-NaN : {len(learning_clean)} minutos '
          f'({(y_train==1).sum()} A, {(y_train==0).sum()} N)')
    print(f'  test post-NaN     : {len(test.dropna(subset=features_usadas))} minutos')

    # =========================================================
    # TECNICA 1: umbral unico sobre cvhr_norm
    # =========================================================
    print()
    print('=' * 70)
    print(f'TECNICA 1: umbral sobre {T1_FEATURE}')
    print('=' * 70)
    T1_t, T1_j = encontrar_umbral_1d(learning_clean[T1_FEATURE], y_train)
    print(f'  Threshold optimo (Youden J)  : {T1_t:.4f}')
    print(f'  Youden J en learning         : {T1_j:.3f}')

    # Predict en TODO el dataframe (NaN -> False -> 0)
    df['pred_T1'] = (df[T1_FEATURE] > T1_t).fillna(False).astype(int)

    m1_train = metricas_binarias(y_train,
                                  (learning_clean[T1_FEATURE] > T1_t).astype(int))
    print(f'  Per-minuto learning: sens {100*m1_train["sens"]:.1f}%, '
          f'spec {100*m1_train["spec"]:.1f}%, acc {100*m1_train["acc"]:.1f}%')

    # =========================================================
    # TECNICA 2: regla AND sobre dos features
    # =========================================================
    print()
    print('=' * 70)
    print(f'TECNICA 2: ({T2_FEATURES[0]} > T1) AND ({T2_FEATURES[1]} > T2)')
    print('=' * 70)
    T2_t1, T2_t2, T2_j = encontrar_umbrales_2d(
        learning_clean[T2_FEATURES[0]], learning_clean[T2_FEATURES[1]],
        y_train, n_grid=40)
    print(f'  Thresholds: {T2_FEATURES[0]} > {T2_t1:.4f}'
          f' AND {T2_FEATURES[1]} > {T2_t2:.4f}')
    print(f'  Youden J en learning : {T2_j:.3f}')

    df['pred_T2'] = (
        (df[T2_FEATURES[0]] > T2_t1) & (df[T2_FEATURES[1]] > T2_t2)
    ).fillna(False).astype(int)

    y_pred_T2_train = ((learning_clean[T2_FEATURES[0]] > T2_t1) &
                        (learning_clean[T2_FEATURES[1]] > T2_t2)).astype(int)
    m2_train = metricas_binarias(y_train, y_pred_T2_train)
    print(f'  Per-minuto learning: sens {100*m2_train["sens"]:.1f}%, '
          f'spec {100*m2_train["spec"]:.1f}%, acc {100*m2_train["acc"]:.1f}%')

    # =========================================================
    # TECNICA 3: distancia a centroides
    # =========================================================
    print()
    print('=' * 70)
    print(f'TECNICA 3: distancia a centroides (features Z-norm: {T3_FEATURES})')
    print('=' * 70)
    medias_T3 = learning_clean[T3_FEATURES].mean()
    stds_T3 = learning_clean[T3_FEATURES].std()

    train_z = (learning_clean[T3_FEATURES] - medias_T3) / stds_T3
    centroide_A = train_z[y_train == 1].mean()
    centroide_N = train_z[y_train == 0].mean()

    print('  Centroide A:', dict((k, round(v, 3)) for k, v in centroide_A.items()))
    print('  Centroide N:', dict((k, round(v, 3)) for k, v in centroide_N.items()))

    # Predict en TODO
    df_z = (df[T3_FEATURES] - medias_T3) / stds_T3
    d_A = np.sqrt(((df_z - centroide_A) ** 2).sum(axis=1))
    d_N = np.sqrt(((df_z - centroide_N) ** 2).sum(axis=1))
    df['pred_T3'] = (d_A < d_N).fillna(False).astype(int)

    # Metricas en learning
    train_z_clean = (learning_clean[T3_FEATURES] - medias_T3) / stds_T3
    d_A_tr = np.sqrt(((train_z_clean - centroide_A) ** 2).sum(axis=1))
    d_N_tr = np.sqrt(((train_z_clean - centroide_N) ** 2).sum(axis=1))
    y_pred_T3_train = (d_A_tr < d_N_tr).astype(int)
    m3_train = metricas_binarias(y_train, y_pred_T3_train)
    print(f'  Per-minuto learning: sens {100*m3_train["sens"]:.1f}%, '
          f'spec {100*m3_train["spec"]:.1f}%, acc {100*m3_train["acc"]:.1f}%')

    # =========================================================
    # BONUS: ensemble por mayoria
    # =========================================================
    df['pred_ENS'] = (
        (df['pred_T1'] + df['pred_T2'] + df['pred_T3']) >= 2
    ).astype(int)
    y_pred_ens = ((m1_t := (learning_clean[T1_FEATURE] > T1_t).astype(int)) +
                  y_pred_T2_train + y_pred_T3_train >= 2).astype(int)
    m_ens_train = metricas_binarias(y_train, y_pred_ens)
    print()
    print('=' * 70)
    print('BONUS: ensemble por mayoria (T1+T2+T3 >= 2 votos)')
    print('=' * 70)
    print(f'  Per-minuto learning: sens {100*m_ens_train["sens"]:.1f}%, '
          f'spec {100*m_ens_train["spec"]:.1f}%, acc {100*m_ens_train["acc"]:.1f}%')

    # =========================================================
    # CLASIFICACION PER-SUJETO
    # =========================================================
    print()
    print('=' * 70)
    print('CLASIFICACION PER-SUJETO')
    print('=' * 70)

    per_sujeto = df.groupby(['record', 'grupo']).agg(
        pred_T1=('pred_T1', 'sum'),
        pred_T2=('pred_T2', 'sum'),
        pred_T3=('pred_T3', 'sum'),
        pred_ENS=('pred_ENS', 'sum'),
        n_minutos=('minute', 'count'),
    ).reset_index()
    per_sujeto['clase_real'] = per_sujeto['grupo'].apply(grupo_a_clase)

    # Calibrar AHI thresholds para cada tecnica sobre learning set
    learning_sub = per_sujeto[per_sujeto['clase_real'].notna()].copy()
    test_sub = per_sujeto[per_sujeto['grupo'] == 'test'].copy()

    thresholds_calibrados = {}
    print()
    print(f'Thresholds AHI calibrados sobre learning set (maximizando acc per-sujeto):')
    print(f'  (default literatura: AHI_A={AHI_A_DEFAULT}, AHI_C={AHI_C_DEFAULT})\n')

    for tec in ['T1', 'T2', 'T3', 'ENS']:
        ahis = learning_sub[f'pred_{tec}'].values
        clases = learning_sub['clase_real'].values
        c_thr, a_thr, acc_lvl = calibrar_thresholds_ahi(ahis, clases)
        thresholds_calibrados[tec] = (c_thr, a_thr)
        print(f'  Tecnica {tec:3s}: AHI_C = {c_thr:6.1f}, AHI_A = {a_thr:6.1f}'
              f'   -> acc learning = {100*acc_lvl:.1f}%')

    # Aplicar thresholds calibrados a todos los sujetos
    for tec in ['T1', 'T2', 'T3', 'ENS']:
        c_thr, a_thr = thresholds_calibrados[tec]
        per_sujeto[f'clase_{tec}'] = per_sujeto[f'pred_{tec}'].apply(
            lambda n, c=c_thr, a=a_thr: clasificar_ahi(n, ahi_a=a, ahi_c=c))

    # Imprimir distribuciones de AHI por clase real (diagnostico)
    print()
    print('Rango de AHI predicho por clase real (learning set, ayuda a entender separabilidad):')
    for tec in ['T1', 'T2', 'T3', 'ENS']:
        c_thr, a_thr = thresholds_calibrados[tec]
        print(f'\n  Tecnica {tec} (C<{c_thr:.0f}, A>={a_thr:.0f}):')
        for clase in 'ABC':
            sub = learning_sub[learning_sub['clase_real'] == clase][f'pred_{tec}']
            if len(sub) > 0:
                print(f'    {clase}: n={len(sub):2d}, '
                      f'min={sub.min():4.0f}, mediana={sub.median():5.0f}, '
                      f'max={sub.max():4.0f}')

    print()
    cols_print = ['record', 'grupo', 'clase_real',
                  'pred_T1', 'clase_T1', 'pred_T2', 'clase_T2',
                  'pred_T3', 'clase_T3', 'pred_ENS', 'clase_ENS']
    print(per_sujeto[cols_print].to_string(index=False))

    # Recargar learning_sub y test_sub con los clase_* ya asignados
    learning_sub = per_sujeto[per_sujeto['clase_real'].notna()].copy()
    test_sub = per_sujeto[per_sujeto['grupo'] == 'test'].copy()

    # =========================================================
    # CONFUSION MATRIX PER-SUJETO (learning set)
    # =========================================================
    print()
    print('=' * 70)
    print('Matriz de confusion per-sujeto (learning set, 35 sujetos)')
    print('=' * 70)
    for tec in ['T1', 'T2', 'T3', 'ENS']:
        print(f'\n--- Tecnica {tec} ---')
        cm = pd.crosstab(learning_sub['clase_real'], learning_sub[f'clase_{tec}'],
                          rownames=['Real'], colnames=['Pred'],
                          margins=True, margins_name='Total')
        print(cm.to_string())
        n_ok = int((learning_sub['clase_real'] == learning_sub[f'clase_{tec}']).sum())
        n_tot = len(learning_sub)
        print(f'  Accuracy: {n_ok}/{n_tot} = {100*n_ok/n_tot:.1f}%')

    # =========================================================
    # DISTRIBUCION EN TEST SET
    # =========================================================
    print()
    print('=' * 70)
    print('Distribucion en test set (esperada: 20 A, 5 B, 10 C)')
    print('=' * 70)
    print(f'Total test set: {len(test_sub)} sujetos\n')

    for tec in ['T1', 'T2', 'T3', 'ENS']:
        counts = test_sub[f'clase_{tec}'].value_counts()
        na = int(counts.get('A', 0))
        nb = int(counts.get('B', 0))
        nc = int(counts.get('C', 0))
        # Error = suma de |observado - esperado| / 2 (porque la suma se duplica)
        error = (abs(na - TEST_DIST_ESPERADA['A']) +
                 abs(nb - TEST_DIST_ESPERADA['B']) +
                 abs(nc - TEST_DIST_ESPERADA['C'])) / 2
        coincide = error / len(test_sub) * 100
        print(f'  Tecnica {tec}: A={na:2d} B={nb:2d} C={nc:2d}  '
              f'(error = {error:.0f} sujetos, ~{coincide:.1f}% mal asignados)')

    # =========================================================
    # GUARDAR RESULTADOS
    # =========================================================
    out_csv = os.path.join(CACHE_DIR, 'clasificacion.csv')
    per_sujeto.to_csv(out_csv, index=False)
    print(f'\nResultados per-sujeto guardados en {out_csv}')

    # =========================================================
    # VISUALIZACIONES
    # =========================================================

    # 1. Histogramas de las features clave con los thresholds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, feat, t_val in zip(axes,
                                 [T1_FEATURE, T2_FEATURES[1]],
                                 [T1_t, T2_t2]):
        ax.hist(learning_clean[learning_clean['label']=='N'][feat],
                 bins=60, alpha=0.55, label=f'Normal (n={(y_train==0).sum()})',
                 color='C0')
        ax.hist(learning_clean[learning_clean['label']=='A'][feat],
                 bins=60, alpha=0.55, label=f'Apnea (n={(y_train==1).sum()})',
                 color='C3')
        ax.axvline(t_val, color='k', linestyle='--', linewidth=1.5,
                    label=f'threshold = {t_val:.3f}')
        ax.set_xlabel(feat); ax.set_ylabel('Cuenta')
        ax.set_title(f'Distribucion de {feat} en learning set')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # 2. Scatter 2D del learning set con la regla T2
    fig, ax = plt.subplots(figsize=(10, 8))
    n_show = 3000
    sample_N = learning_clean[learning_clean['label']=='N'].sample(
        min(n_show, (y_train==0).sum()), random_state=0)
    sample_A = learning_clean[learning_clean['label']=='A'].sample(
        min(n_show, (y_train==1).sum()), random_state=0)
    ax.scatter(sample_N[T2_FEATURES[0]], sample_N[T2_FEATURES[1]],
                s=4, alpha=0.25, color='C0', label='Normal')
    ax.scatter(sample_A[T2_FEATURES[0]], sample_A[T2_FEATURES[1]],
                s=4, alpha=0.25, color='C3', label='Apnea')
    ax.axvline(T2_t1, color='k', linestyle='--', alpha=0.7)
    ax.axhline(T2_t2, color='k', linestyle='--', alpha=0.7)
    ax.axvspan(T2_t1, ax.get_xlim()[1], ymin=(T2_t2 - ax.get_ylim()[0]) /
                (ax.get_ylim()[1] - ax.get_ylim()[0]),
                alpha=0.10, color='C3')
    ax.set_xlabel(T2_FEATURES[0]); ax.set_ylabel(T2_FEATURES[1])
    ax.set_title(f'Tecnica 2: regla AND con [{T2_t1:.3f}, {T2_t2:.3f}]')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # 3. AHI per sujeto por tecnica (learning set), coloreado por clase real
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, tec in zip(axes, ['T1', 'T2', 'T3', 'ENS']):
        c_thr, a_thr = thresholds_calibrados[tec]
        for clase, color in [('A', 'C3'), ('B', 'C1'), ('C', 'C2')]:
            sub = learning_sub[learning_sub['clase_real'] == clase]
            x_offset = {'A': 0, 'B': 1, 'C': 2}[clase]
            jitter = np.random.RandomState(0).uniform(-0.2, 0.2, len(sub))
            ax.scatter([x_offset]*len(sub) + jitter, sub[f'pred_{tec}'],
                       color=color, s=50, alpha=0.7,
                       label=f'Real={clase} (n={len(sub)})')
        ax.axhline(a_thr, color='k', linestyle='--', alpha=0.5,
                    label=f'AHI_A = {a_thr:.0f} (calibrado)')
        ax.axhline(c_thr, color='gray', linestyle='--', alpha=0.5,
                    label=f'AHI_C = {c_thr:.0f} (calibrado)')
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['A', 'B', 'C'])
        ax.set_xlabel('Clase real')
        ax.set_ylabel('Minutos predichos como apnea (AHI)')
        ax.set_title(f'Tecnica {tec}')
        ax.set_yscale('symlog')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Learning set: AHI predicho con thresholds calibrados por tecnica')
    plt.tight_layout(); plt.show()

    # 4. Distribucion en test set comparada con esperada
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    x_pos = np.arange(3)
    width = 0.35
    esperado = [TEST_DIST_ESPERADA[c] for c in 'ABC']
    for ax, tec in zip(axes, ['T1', 'T2', 'T3', 'ENS']):
        counts = test_sub[f'clase_{tec}'].value_counts()
        obs = [int(counts.get(c, 0)) for c in 'ABC']
        ax.bar(x_pos - width/2, esperado, width, label='Esperado',
                color='gray', alpha=0.7)
        ax.bar(x_pos + width/2, obs, width, label=f'Predicho ({tec})',
                color='C0')
        for i, (e, o) in enumerate(zip(esperado, obs)):
            ax.text(i - width/2, e + 0.3, str(e), ha='center', fontsize=10)
            ax.text(i + width/2, o + 0.3, str(o), ha='center', fontsize=10)
        ax.set_xticks(x_pos); ax.set_xticklabels(['A', 'B', 'C'])
        ax.set_xlabel('Clase'); ax.set_ylabel('Sujetos')
        ax.set_title(f'Tecnica {tec}')
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('Test set: distribucion predicha vs esperada (20 A, 5 B, 10 C)')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
