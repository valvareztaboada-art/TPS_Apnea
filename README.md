# TPS Apnea-ECG: Detección automática de apnea obstructiva del sueño desde ECG

Trabajo Práctico de Procesamiento de Señales Biomédicas. Implementa un pipeline completo
para detectar episodios de apnea obstructiva del sueño a partir de una única derivación
de ECG nocturno, sobre la base pública **Apnea-ECG** de PhysioNet. No se utilizan
bibliotecas de machine learning ni wavelets: el sistema está basado en filtros clásicos,
detector de QRS por Pan-Tompkins, features HRV temporales y espectrales, features EDR
(ECG-derived respiration), y clasificación por umbrales y distancias a centroides.

Incluye una interfaz gráfica interactiva (PySide6 + pyqtgraph) para inspeccionar
visualmente el funcionamiento del algoritmo sobre cualquiera de los 70 sujetos.


## Resultados principales

Sobre el conjunto de entrenamiento (35 sujetos a/b/c\*):

| Técnica | Accuracy ternaria (A/B/C) | Binaria A vs no-A | Binaria C vs no-C |
|---|---|---|---|
| T1 — Umbral sobre `cvhr_norm` | **82,9 %** | **88,6 %** | 88,6 % |
| T2 — AND sobre `cvhr_norm` y `lf_hf_ratio` | 77,1 % | 85,7 % | 85,7 % |
| T3 — Distancia a centroides HRV+EDR | 80,0 % | 82,9 % | **91,4 %** |
| ENS — Mayoría de votos | 80,0 % | 85,7 % | 88,6 % |

Sobre el conjunto de test (35 sujetos x\*, distribución poblacional conocida 20A/5B/10C):
todas las técnicas predicen distribuciones cercanas a la esperada con error de 4–5 sujetos.


## Estructura del proyecto

```
tps_apnea/
├── base_de_datos/                ← PhysioNet Apnea-ECG (descargar por separado)
│   ├── a01.dat, a01.hea, a01.apn, a01.qrs
│   └── ... (70 sujetos)
│
├── cache/                        ← generado automáticamente por los scripts
│   ├── <record>.npz              ← picos R, RR limpio, flags por sujeto
│   ├── resumen.csv               ← métricas por sujeto del batch
│   ├── features.csv              ← features HRV+EDR por minuto (todos los sujetos)
│   ├── clasificacion.csv         ← clase predicha por sujeto y técnica
│   └── predicciones_por_minuto.csv  ← predicciones per-minuto (para la interfaz)
│
├── src/                          ← lógica reutilizable (sin side effects)
│   ├── __init__.py
│   ├── pipeline.py               ← filtros, Pan-Tompkins, limpieza RR
│   └── features.py               ← features HRV + EDR (Lomb-Scargle)
│
├── 01_exploracion_inicial.py     ← visualización inicial de los registros
├── 02_analisis_espectral.py      ← PSD del ECG, justificación de filtros
├── 03_preprocesamiento_y_qrs.py  ← pipeline completo sobre 1 sujeto
├── 03b_procesar_todos.py         ← BATCH: procesa los 70 sujetos → cache/
├── 04_features_por_minuto.py     ← BATCH: calcula features → features.csv
├── 05_deteccion_y_metricas.py    ← clasificación, métricas, plots
├── interfaz_apnea.py             ← interfaz gráfica interactiva
│
├── requirements.txt              ← dependencias de Python
└── README.md                     ← este archivo
```


## Qué hace cada archivo

### Lógica reutilizable (`src/`)

- **`src/pipeline.py`**: filtros Butterworth para el ECG (HP 0,5 Hz + LP 40 Hz),
  detector de QRS Pan-Tompkins desde cero con umbral local adaptativo, limpieza
  temporal de la serie R–R por filtros encadenados (rango fisiológico + filtro
  de Malik + mediana local). Sin código de visualización; solo funciones puras.
- **`src/features.py`**: cómputo de features HRV temporales (SDNN, RMSSD, NN50,
  pNN50…) y espectrales (potencias VLF/LF/HF, CVHR, ratios) por Lomb-Scargle.
  Cálculo de amplitudes de R y features EDR (potencia respiratoria normal vs
  banda apneica).

### Scripts numerados (ejecutables en orden)

- **`01_exploracion_inicial.py`**: lee la base, verifica integridad de los
  archivos, plotea ECG y anotaciones de minutos apneicos. Sirve para confirmar
  que la base está bien descargada antes de seguir.
- **`02_analisis_espectral.py`**: estima la PSD del ECG (Welch) y justifica
  cuantitativamente la elección de los filtros (corte en 0,5 Hz y 40 Hz).
- **`03_preprocesamiento_y_qrs.py`**: aplica el pipeline completo a un único
  sujeto (a01 por defecto) mostrando todos los pasos intermedios con plots.
  Útil para inspeccionar visualmente el funcionamiento.
- **`03b_procesar_todos.py`**: corre el pipeline sobre los 70 sujetos y
  persiste el resultado en `cache/*.npz`. Tarda aprox. 5 minutos. Genera
  también `cache/resumen.csv` con métricas por sujeto comparadas contra la
  referencia oficial `.qrs` de PhysioNet.
- **`04_features_por_minuto.py`**: lee `cache/`, carga el ECG filtrado, calcula
  amplitudes de R y features HRV+EDR para cada minuto de cada sujeto.
  Resultado: `cache/features.csv` (~34 000 filas, 26 columnas). Tarda ~5 min.
- **`05_deteccion_y_metricas.py`**: implementa las 3 técnicas (T1, T2, T3),
  calibra umbrales sobre el conjunto de entrenamiento, evalúa per-minuto y
  per-sujeto, reporta matrices de confusión y métricas binarias y de
  sensibilidad. Genera `cache/clasificacion.csv` y
  `cache/predicciones_por_minuto.csv`.

### Interfaz gráfica

- **`interfaz_apnea.py`**: aplicación interactiva con PySide6 + pyqtgraph.
  Permite seleccionar cualquiera de los 70 sujetos del combo desplegable y
  visualiza simultáneamente:
  - ECG completo de la noche con los picos R detectados.
  - Tacograma R–R con los intervalos descartados marcados.
  - Features por minuto con el fondo coloreado según la clase predicha.
  - Tabla de todos los minutos con sus features y predicciones; el clic en
    una fila lleva los paneles superiores a ese minuto.
  - Panel lateral con grupo real, clase predicha por cada técnica e índice
    AHI calculado.


## Cómo instalarlo

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/tps_apnea.git
cd tps_apnea
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Cómo correrlo

Los scripts están pensados para ejecutarse **en orden** porque cada uno consume
las salidas del anterior (vía el directorio `cache/`).

```bash
# 1. Verificación inicial (opcional, solo plots exploratorios)
python 01_exploracion_inicial.py

# 2. Análisis espectral (opcional, justifica los filtros)
python 02_analisis_espectral.py

# 3. Pipeline sobre un solo sujeto (opcional, para inspeccionar visualmente)
python 03_preprocesamiento_y_qrs.py

# 4. BATCH: procesa los 70 sujetos (~4 min) - REQUERIDO
python 03b_procesar_todos.py

# 5. BATCH: calcula features (~9 min) - REQUERIDO
python 04_features_por_minuto.py

# 6. Clasificación y métricas - REQUERIDO
python 05_deteccion_y_metricas.py

# 7. Interfaz gráfica (opcional)
python interfaz_apnea.py
```

Los scripts marcados como REQUERIDO son los que generan los archivos del
directorio `cache/` que después usa la interfaz. Los scripts 01, 02 y 03 son
exploratorios; útiles para entender el pipeline pero no obligatorios para
llegar al resultado.


## Decisiones metodológicas relevantes

- **Detección de QRS**: implementación propia de Pan-Tompkins. Se usó **umbral
  local adaptativo** (recalculado cada 30 s) en lugar de global, porque varios
  registros presentan cambios de amplitud del QRS a lo largo de la noche que
  hacían perder muchos latidos al detector con umbral fijo.
- **Limpieza de la serie R–R**: estrategia **temporal** (rango + Malik +
  mediana local) en lugar de morfológica. La frecuencia de muestreo de la base
  (100 Hz) introduce cuantización temporal de 10 ms en la posición del pico R,
  que es comparable a las diferencias morfológicas reales entre latidos
  normales y ectópicos. 
- **Análisis espectral**: se usa **Lomb-Scargle** en lugar de Welch porque la
  serie R–R no está uniformemente muestreada en el tiempo.
- **Ventana para features espectrales**: 5 minutos centrados en el minuto
  evaluado, porque 1 minuto es insuficiente para resolver la banda LF (que
  empieza en 0,04 Hz, período de 25 s).
- **Calibración de umbrales AHI**: los thresholds del paper original (100, 5)
  son para conteos *reales* de minutos apneicos; aplicados a las salidas de un
  detector imperfecto requieren recalibración. Los umbrales se ajustan en el
  conjunto de entrenamiento maximizando accuracy por sujeto, sin tocar el
  conjunto de test.


## Limitaciones conocidas

- La frecuencia de muestreo de 100 Hz es relativamente baja para análisis de
  HRV; idealmente la base debería estar a 250 Hz o más. Esto introduce
  cuantización temporal que limita la precisión de las features.
- La clase B (apnea *borderline*) es inherentemente ambigua, según el propio
  paper original. Las métricas binarias A vs no-A y C vs no-C resultan más
  representativas que la accuracy ternaria estricta.


## Referencias

- Penzel *et al.* (2000) — *The Apnea-ECG Database*, Computers in Cardiology.
- Pan & Tompkins (1985) — *A Real-Time QRS Detection Algorithm*, IEEE TBME.
- Task Force ESC/NASPE (1996) — *Heart Rate Variability Standards*, Circulation.
- de Chazal *et al.* (2003) — *Automated Processing of the Single-Lead ECG for
  Detection of Sleep Apnea*, IEEE TBME.
- Moody *et al.* (1985) — *Derivation of Respiratory Signals from Multi-Lead
  ECGs*, Computers in Cardiology.


[Especificar licencia si aplica, ej. MIT, GPL, etc.]


## Autores

[Nombre del/los autor/es] — Trabajo Práctico de Procesamiento de Señales
Biomédicas, [fecha].
