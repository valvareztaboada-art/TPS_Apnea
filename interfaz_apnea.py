# -*- coding: utf-8 -*-
"""
interfaz_apnea.py
==================

Interfaz interactiva (PySide6 + pyqtgraph) para visualizar el pipeline
completo de deteccion de apnea sobre cualquier sujeto de la base
Apnea-ECG.

Funcionalidad:
  - Selector de sujeto (los 70 de la base).
  - Panel resumen lateral con grupo, clase real, duracion, FC, y
    predicciones (AHI y clase) por cada una de las 4 tecnicas.
  - Panel ECG con R detectados marcados, zoom horizontal sincronizado
    con el tacograma.
  - Panel tacograma RR con los outliers temporales (descartados por la
    limpieza) marcados.
  - Panel features por minuto con cvhr_norm, lf_hf_ratio y
    edr_apnea_resp_ratio Z-normalizados. Fondo coloreado por la
    prediccion (T1 por defecto). Lineas verticales marcando los minutos
    de apnea real (cuando hay label disponible).
  - Tabla de todos los minutos: click en una fila hace zoom a ese
    minuto en el ECG/tacograma y mueve la marca en el panel de features.

Pre-requisitos:
  - cache/*.npz       (de 03b_procesar_todos.py)
  - cache/features.csv      (de 04_features_por_minuto.py)
  - cache/clasificacion.csv (de 05_deteccion_y_metricas.py)
  - base_de_datos/<record>.{dat,hea}  (los registros wfdb originales)

Uso:
    python interfaz_apnea.py
"""

import os
import sys

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QComboBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableView,
    QHeaderView, QGroupBox, QGridLayout, QStatusBar, QMessageBox,
)
import pyqtgraph as pg

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.pipeline import (
    cargar_ecg, filtrar_ecg_general, clasificar_grupo,
)


# =============================================================================
# Constantes
# =============================================================================

DATA_DIR = 'apnea-ecg-database-1.0.0'
CACHE_DIR = 'cache'

# Colores por clase (RGB 0-255)
CLASS_COLORS = {
    'A': (200, 60, 60),     # rojo - apnea
    'B': (220, 150, 50),    # naranja - borderline
    'C': (60, 160, 80),     # verde - control
}

# Tema general
pg.setConfigOption('background', '#fafafa')
pg.setConfigOption('foreground', '#222222')
pg.setConfigOption('antialias', True)


# =============================================================================
# Modelo Qt para la tabla de minutos
# =============================================================================

class TablaMinutosModel(QtCore.QAbstractTableModel):
    """Tabla con todas las features y predicciones por minuto del
    sujeto seleccionado.
    """

    COLUMNS = [
        ('minute', 'Min'),
        ('label', 'Real'),
        ('cvhr_norm', 'cvhr_norm'),
        ('lf_hf_ratio', 'LF/HF'),
        ('edr_apnea_resp_ratio', 'EDR ap/resp'),
        ('pred_T1', 'T1'),
        ('pred_T2', 'T2'),
        ('pred_T3', 'T3'),
        ('pred_ENS', 'ENS'),
    ]

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df.reset_index(drop=True)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section][1]
        if role == Qt.DisplayRole and orientation == Qt.Vertical:
            return str(section)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col_key = self.COLUMNS[index.column()][0]
        val = self.df.iloc[index.row()].get(col_key) if col_key in self.df.columns else None

        if role == Qt.DisplayRole:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return '-'
            if isinstance(val, float):
                return f'{val:.3f}'
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            return str(val)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        if role == Qt.BackgroundRole:
            # Color de fondo segun label/prediccion
            if col_key == 'label':
                if val == 'A':
                    return QtGui.QColor(255, 225, 225)
                if val == 'N':
                    return QtGui.QColor(225, 245, 225)
            if col_key in ('pred_T1', 'pred_T2', 'pred_T3', 'pred_ENS'):
                if val == 1:
                    return QtGui.QColor(255, 225, 225)
                if val == 0:
                    return QtGui.QColor(225, 245, 225)

        return None


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):

    # Tecnica usada para colorear el fondo del panel de features
    TECNICA_FONDO = 'T1'

    def __init__(self):
        super().__init__()
        self.setWindowTitle('TPS Apnea-ECG — Detector multimodal HRV+EDR')
        self.resize(1500, 900)

        # Estado del registro actual
        self._record = None
        self._cache = None
        self._ecg = None
        self._feat = None
        self._clasif = None
        self._marca_minuto = None  # InfiniteLine en panel features

        self._cargar_datos_estaticos()
        self._build_ui()

        # Cargar primer registro automaticamente
        if self.records:
            self.combo_sujeto.setCurrentIndex(0)
            # Forzar carga (currentTextChanged ya disparado por addItems primero
            # pero la senal de currentIndexChanged sera mas confiable aca)
            self.cargar_registro(self.records[0])

    # -------------------------------------------------------------------------
    # Construccion de la UI
    # -------------------------------------------------------------------------

    def _cargar_datos_estaticos(self):
        feat_path = os.path.join(CACHE_DIR, 'features.csv')
        clas_path = os.path.join(CACHE_DIR, 'clasificacion.csv')
        pred_path = os.path.join(CACHE_DIR, 'predicciones_por_minuto.csv')

        faltan = [p for p in (feat_path, clas_path, pred_path)
                  if not os.path.exists(p)]
        if faltan:
            QMessageBox.critical(self, 'Error',
                f'Faltan archivos del cache:\n' + '\n'.join('  - ' + p for p in faltan) +
                '\n\nCorrer antes:\n'
                '  python 04_features_por_minuto.py\n'
                '  python 05_deteccion_y_metricas.py')
            sys.exit(1)

        self.features_df = pd.read_csv(feat_path)
        self.clasif_df = pd.read_csv(clas_path)
        self.pred_min_df = pd.read_csv(pred_path)

        # Merge: agregar predicciones per-minuto al dataframe de features
        merge_cols = ['record', 'minute', 'pred_T1', 'pred_T2',
                       'pred_T3', 'pred_ENS']
        merge_cols = [c for c in merge_cols if c in self.pred_min_df.columns]
        self.features_df = self.features_df.merge(
            self.pred_min_df[merge_cols],
            on=['record', 'minute'], how='left')

        self.records = sorted(self.features_df['record'].unique().tolist())

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # --- Panel izquierdo: selector + resumen + predicciones ----
        left_widget = QWidget()
        left_widget.setMinimumWidth(280)
        left_widget.setMaximumWidth(330)
        ll = QVBoxLayout(left_widget)
        ll.setSpacing(8)

        # Selector
        ll.addWidget(QLabel('<b>Sujeto:</b>'))
        self.combo_sujeto = QComboBox()
        self.combo_sujeto.addItems(self.records)
        self.combo_sujeto.currentTextChanged.connect(self.cargar_registro)
        ll.addWidget(self.combo_sujeto)

        # Resumen
        gb_resumen = QGroupBox('Resumen')
        gl = QGridLayout(gb_resumen)
        gl.setVerticalSpacing(4)
        self.lbl_grupo = QLabel('-')
        self.lbl_clase_real = QLabel('-')
        self.lbl_duracion = QLabel('-')
        self.lbl_fc_media = QLabel('-')
        self.lbl_n_outliers = QLabel('-')
        for i, (k, v) in enumerate([
            ('Grupo', self.lbl_grupo),
            ('Clase real', self.lbl_clase_real),
            ('Duracion', self.lbl_duracion),
            ('FC media', self.lbl_fc_media),
            ('Outliers RR', self.lbl_n_outliers),
        ]):
            gl.addWidget(QLabel(f'{k}:'), i, 0)
            gl.addWidget(v, i, 1)
        ll.addWidget(gb_resumen)

        # Predicciones
        gb_pred = QGroupBox('Predicciones por tecnica')
        gp = QGridLayout(gb_pred)
        gp.setVerticalSpacing(4)
        gp.addWidget(QLabel('<b>Tec.</b>'), 0, 0)
        gp.addWidget(QLabel('<b>AHI</b>'), 0, 1)
        gp.addWidget(QLabel('<b>Clase</b>'), 0, 2)
        self.lbl_tec = {}
        descripciones = {
            'T1': 'T1 (HRV)',
            'T2': 'T2 (HRV+combo)',
            'T3': 'T3 (HRV+EDR)',
            'ENS': 'ENS (mayoria)',
        }
        for i, tec in enumerate(['T1', 'T2', 'T3', 'ENS'], 1):
            lbl_nombre = QLabel(descripciones[tec])
            lbl_nombre.setToolTip(descripciones[tec])
            gp.addWidget(lbl_nombre, i, 0)
            self.lbl_tec[tec] = {
                'ahi': QLabel('-'),
                'clase': QLabel('-'),
            }
            self.lbl_tec[tec]['clase'].setAlignment(Qt.AlignCenter)
            self.lbl_tec[tec]['clase'].setMinimumWidth(40)
            gp.addWidget(self.lbl_tec[tec]['ahi'], i, 1)
            gp.addWidget(self.lbl_tec[tec]['clase'], i, 2)
        ll.addWidget(gb_pred)

        ll.addStretch()

        # Legend chico
        legend = QLabel(
            '<small><b>Leyenda</b><br>'
            '<span style="background-color: rgb(200,60,60); color: white;'
            ' padding: 1px 4px;">A</span> apnea &nbsp;'
            '<span style="background-color: rgb(220,150,50); color: white;'
            ' padding: 1px 4px;">B</span> border. &nbsp;'
            '<span style="background-color: rgb(60,160,80); color: white;'
            ' padding: 1px 4px;">C</span> control</small>'
        )
        legend.setWordWrap(True)
        ll.addWidget(legend)

        main_layout.addWidget(left_widget)

        # --- Panel derecho: plots + tabla --------------------------
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # Plot ECG
        self.plot_ecg = pg.PlotWidget(title='ECG con R detectados')
        self.plot_ecg.setLabel('left', 'ECG (mV)')
        self.plot_ecg.setLabel('bottom', 'Tiempo (s)')
        self.plot_ecg.setDownsampling(auto=True, mode='peak')
        self.plot_ecg.setClipToView(True)
        self.plot_ecg.showGrid(x=True, y=True, alpha=0.3)
        splitter.addWidget(self.plot_ecg)

        # Plot tacograma
        self.plot_rr = pg.PlotWidget(title='Tacograma RR (con outliers marcados)')
        self.plot_rr.setLabel('left', 'RR (s)')
        self.plot_rr.setLabel('bottom', 'Tiempo (s)')
        self.plot_rr.showGrid(x=True, y=True, alpha=0.3)
        # X sincronizado con ECG
        self.plot_rr.setXLink(self.plot_ecg)
        splitter.addWidget(self.plot_rr)

        # Plot features por minuto
        self.plot_features = pg.PlotWidget(
            title=f'Features por minuto y clasificacion '
                  f'(fondo = {self.TECNICA_FONDO})')
        self.plot_features.setLabel('left', 'Feature (z-norm robusto)')
        self.plot_features.setLabel('bottom', 'Minuto')
        self.plot_features.showGrid(x=True, y=True, alpha=0.3)
        self.plot_features.addLegend(offset=(10, 10))
        splitter.addWidget(self.plot_features)

        # Tabla
        self.tabla = QTableView()
        self.tabla.setSelectionBehavior(QTableView.SelectRows)
        self.tabla.setSelectionMode(QTableView.SingleSelection)
        self.tabla.setSortingEnabled(False)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.tabla.clicked.connect(self._on_tabla_click)
        splitter.addWidget(self.tabla)

        splitter.setSizes([220, 160, 280, 220])
        main_layout.addWidget(splitter, stretch=1)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Listo')

    # -------------------------------------------------------------------------
    # Carga y actualizacion de un registro
    # -------------------------------------------------------------------------

    def cargar_registro(self, record):
        if not record:
            return
        self.statusBar().showMessage(f'Cargando {record}...')
        QApplication.processEvents()
        try:
            self._record = record

            # Cache npz
            cache_path = os.path.join(CACHE_DIR, f'{record}.npz')
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f'No se encontro {cache_path}. '
                    f'Correr 03b_procesar_todos.py primero.')
            self._cache = np.load(cache_path)
            self._fs = int(self._cache['fs'])

            # ECG raw + filtrado
            ecg_raw, _, _ = cargar_ecg(record, DATA_DIR)
            self._ecg = filtrar_ecg_general(ecg_raw, self._fs)

            # Features de este registro
            self._feat = self.features_df[
                self.features_df['record'] == record
            ].reset_index(drop=True)

            # Clasificacion de este registro (una sola fila)
            cl_row = self.clasif_df[self.clasif_df['record'] == record]
            self._clasif = cl_row.iloc[0] if len(cl_row) > 0 else None

            # Merge labels y predicciones a self._feat para la tabla
            for tec in ['T1', 'T2', 'T3', 'ENS']:
                if f'pred_{tec}' not in self._feat.columns:
                    self._feat[f'pred_{tec}'] = 0  # placeholder

            self._actualizar_resumen()
            self._actualizar_plot_ecg()
            self._actualizar_plot_rr()
            self._actualizar_plot_features()
            self._actualizar_tabla()

            self.statusBar().showMessage(
                f'Cargado: {record}  ({len(self._ecg)/self._fs/3600:.1f}h)',
                5000)
        except Exception as e:
            self.statusBar().showMessage(f'Error: {e}', 10000)
            QMessageBox.warning(self, 'Error cargando registro', str(e))

    # -------------------------------------------------------------------------
    # Updaters por panel
    # -------------------------------------------------------------------------

    def _actualizar_resumen(self):
        if self._clasif is None:
            return
        c = self._clasif

        self.lbl_grupo.setText(str(c['grupo']))

        clase_real = c.get('clase_real')
        if pd.isna(clase_real) or clase_real is None or clase_real == '':
            self.lbl_clase_real.setText('N/A (test)')
            self.lbl_clase_real.setStyleSheet('color: gray;')
        else:
            color = CLASS_COLORS.get(str(clase_real), (128, 128, 128))
            self.lbl_clase_real.setText(str(clase_real))
            self.lbl_clase_real.setStyleSheet(
                f'background-color: rgb{color}; color: white; '
                f'padding: 1px 6px; border-radius: 3px;')

        dur_s = float(self._cache['duracion_s'])
        h = int(dur_s // 3600); m = int((dur_s % 3600) // 60)
        self.lbl_duracion.setText(f'{h}h {m}min')

        rr = np.asarray(self._cache['rr_interp'])
        if len(rr) > 0:
            fc = 60.0 / np.mean(rr)
            self.lbl_fc_media.setText(f'{fc:.1f} bpm')

        flag_total = np.asarray(self._cache['flag_total'])
        if len(flag_total) > 0:
            pct = 100 * flag_total.sum() / len(flag_total)
            self.lbl_n_outliers.setText(
                f'{int(flag_total.sum())} ({pct:.1f}%)')

        # Predicciones por tecnica
        for tec in ['T1', 'T2', 'T3', 'ENS']:
            ahi = int(c[f'pred_{tec}'])
            clase = str(c[f'clase_{tec}'])
            self.lbl_tec[tec]['ahi'].setText(str(ahi))
            self.lbl_tec[tec]['clase'].setText(clase)
            color = CLASS_COLORS.get(clase, (128, 128, 128))
            self.lbl_tec[tec]['clase'].setStyleSheet(
                f'background-color: rgb{color}; color: white; '
                f'padding: 1px 6px; border-radius: 3px;')

    def _actualizar_plot_ecg(self):
        self.plot_ecg.clear()
        # Senal completa
        t = np.arange(len(self._ecg)) / self._fs
        self.plot_ecg.plot(t, self._ecg,
                            pen=pg.mkPen('#2f5d9e', width=1))
        # R detectados encima
        picos = np.asarray(self._cache['picos_R'])
        t_R = picos / self._fs
        ecg_R = self._ecg[picos]
        self.plot_ecg.plot(t_R, ecg_R,
                            pen=None, symbol='o',
                            symbolBrush=(200, 60, 60),
                            symbolSize=4, symbolPen=None)
        # Vista por defecto: primeros 60s
        self.plot_ecg.setXRange(0, 60)
        # Auto-rango Y
        self.plot_ecg.enableAutoRange(axis='y')

    def _actualizar_plot_rr(self):
        self.plot_rr.clear()
        picos = np.asarray(self._cache['picos_R'])
        rr_interp = np.asarray(self._cache['rr_interp'])
        rr_crudo = np.asarray(self._cache['rr_crudo'])
        flag_total = np.asarray(self._cache['flag_total'])

        # Tiempos: cada RR se asocia al instante del 2do pico (fin del intervalo)
        t_rr = picos[1:] / self._fs

        # Linea de RR interpolado (continuo)
        self.plot_rr.plot(t_rr, rr_interp,
                          pen=pg.mkPen('#444444', width=1))

        # Puntos: validos (azul) y outliers (rojo) sobre el rr_crudo
        valido = ~flag_total
        # Algunos valores crudos podrian ser muy chicos/grandes; los puntos
        # outliers los marcamos sobre el rr_interp para que esten en la curva
        if np.any(valido):
            self.plot_rr.plot(t_rr[valido], rr_crudo[valido],
                              pen=None, symbol='o',
                              symbolBrush=(50, 100, 180),
                              symbolSize=3, symbolPen=None,
                              name='RR valido')
        if np.any(flag_total):
            self.plot_rr.plot(t_rr[flag_total], rr_interp[flag_total],
                              pen=None, symbol='x',
                              symbolBrush=(200, 60, 60),
                              symbolSize=8,
                              symbolPen=pg.mkPen((200, 60, 60), width=1.5),
                              name='outlier (interpolado)')

    def _actualizar_plot_features(self):
        self.plot_features.clear()
        if self._feat is None or len(self._feat) == 0:
            return

        minutos = self._feat['minute'].values

        # ---- Fondo coloreado por la prediccion (T1 default) ----
        col_pred = f'pred_{self.TECNICA_FONDO}'
        # Usamos pred_T1 que es 0/1 per-minuto
        if col_pred in self._feat.columns:
            preds = self._feat[col_pred].fillna(0).astype(int).values
            # Construir una imagen RGBA 1xN para usar como fondo
            n = len(preds)
            img = np.zeros((n, 1, 4), dtype=np.ubyte)
            # Color apnea (predicho como A)
            apnea_rgb = CLASS_COLORS['A']
            normal_rgb = CLASS_COLORS['C']
            for i, p in enumerate(preds):
                c = apnea_rgb if p == 1 else normal_rgb
                img[i, 0] = (c[0], c[1], c[2], 35)  # alpha bajo
            img_item = pg.ImageItem(img)
            # ImageItem en coordenadas (x: 0..n, y: ymin..ymax)
            # Vamos a ponerlo en y de -5 a +5 para cubrir el rango z-norm tipico
            img_item.setRect(QtCore.QRectF(0, -6, n, 12))
            img_item.setZValue(-10)
            self.plot_features.addItem(img_item)

        # ---- Features Z-normalizadas (robustas) ----
        feats_to_plot = [
            ('cvhr_norm', '#aa3333'),
            ('lf_hf_ratio', '#2e7d32'),
            ('edr_apnea_resp_ratio', '#1565c0'),
        ]
        for feat_name, color in feats_to_plot:
            if feat_name not in self._feat.columns:
                continue
            vals = self._feat[feat_name].values.astype(float)
            med = np.nanmedian(vals)
            iqr = (np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
            z = (vals - med) / iqr if iqr > 0 else np.zeros_like(vals)
            # Clip para no romper el grafico
            z = np.clip(z, -5, 5)
            self.plot_features.plot(minutos, z,
                                     pen=pg.mkPen(color, width=1.5),
                                     name=feat_name)

        # ---- Marcas verticales: minutos con label = 'A' (ground truth) ----
        if 'label' in self._feat.columns:
            apnea_min = self._feat[self._feat['label'] == 'A']['minute'].values
            if len(apnea_min) > 0:
                # Pequenos triangulos arriba
                y_marker = 4.5
                self.plot_features.plot(
                    apnea_min, np.full(len(apnea_min), y_marker),
                    pen=None, symbol='t', symbolBrush=(200, 60, 60, 200),
                    symbolSize=6, symbolPen=None,
                    name='label = A (real)'
                )

        # ---- Marca de minuto seleccionado (line infinite vertical) ----
        self._marca_minuto = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#000', width=2, style=Qt.DashLine))
        self._marca_minuto.setPos(-1)  # off-screen al inicio
        self.plot_features.addItem(self._marca_minuto)

        self.plot_features.setYRange(-6, 6, padding=0)

    def _actualizar_tabla(self):
        modelo = TablaMinutosModel(self._feat)
        self.tabla.setModel(modelo)
        self.tabla.resizeColumnsToContents()

    # -------------------------------------------------------------------------
    # Interacciones
    # -------------------------------------------------------------------------

    def _on_tabla_click(self, index):
        """Click en una fila de la tabla => zoom al minuto en ECG/RR y
        marca en panel features."""
        row = index.row()
        if self._feat is None or row >= len(self._feat):
            return
        minuto = int(self._feat.iloc[row]['minute'])
        t0 = minuto * 60
        t1 = (minuto + 1) * 60

        # Zoom 1 min en ECG (RR lo sigue por el setXLink)
        self.plot_ecg.setXRange(t0, t1, padding=0)

        # Marca vertical en features
        if self._marca_minuto is not None:
            self._marca_minuto.setPos(minuto)

        self.statusBar().showMessage(
            f'Minuto {minuto}  (t = {t0}s - {t1}s)', 5000)


# =============================================================================
# Entry point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
