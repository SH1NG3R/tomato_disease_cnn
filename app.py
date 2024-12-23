import sys
import cv2
import json
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                           QFileDialog, QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from model_architecture import create_model

class TomatoDiseaseDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Enfermedades en Tomates")
        self.setGeometry(100, 100, 1200, 800)
        
        # Cargar el modelo y la información
        self.load_model()
        
        # Inicializar variables
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.captured_frames = []
        self.predictions_history = []
        
        # Crear la interfaz principal
        self.init_ui()
        
    def load_model(self):
        """Cargar el modelo y la información relacionada"""
        try:
            # Crear el modelo con la arquitectura correcta
            self.model = create_model()
            
            # Cargar los pesos
            self.model.load_weights('models/TomatoDisease_acc_0.9871_weights')
            
            # Cargar información adicional
            with open('models/TomatoDisease_acc_0.9871_info.json', 'r') as f:
                self.model_info = json.load(f)
                
            print("Modelo cargado exitosamente")
                
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            sys.exit(1)
    
    def init_ui(self):
        """Inicializar la interfaz de usuario"""
        # Crear el widget principal y el layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Crear el widget de pestañas
        tabs = QTabWidget()
        
        # Pestaña 1: Detección
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)
        
        # Área de visualización
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detection_layout.addWidget(self.image_label)
        
        # Botones de control
        button_layout = QHBoxLayout()
        self.camera_button = QPushButton("Iniciar Cámara")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.capture_button = QPushButton("Capturar")
        self.capture_button.clicked.connect(self.capture_frame)
        self.load_button = QPushButton("Cargar Imagen")
        self.load_button.clicked.connect(self.load_image)
        
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.load_button)
        detection_layout.addLayout(button_layout)
        
        # Etiqueta de predicción
        self.prediction_label = QLabel("Predicción: ")
        detection_layout.addWidget(self.prediction_label)
        
        # Pestaña 2: Estadísticas
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        # Información del modelo
        model_info = f"""
        Precisión del modelo: {self.model_info['accuracy']:.2%}
        Tamaño de imagen: {self.model_info['image_size'][0]}x{self.model_info['image_size'][1]}
        Número de clases: {len(self.model_info['classes'])}
        """
        stats_layout.addWidget(QLabel(model_info))
        
        # Tabla de clases
        class_table = QTableWidget()
        class_table.setColumnCount(2)
        class_table.setHorizontalHeaderLabels(['Clase', 'Índice'])
        class_table.setRowCount(len(self.model_info['classes']))
        
        for i, (class_name, class_idx) in enumerate(self.model_info['classes'].items()):
            class_table.setItem(i, 0, QTableWidgetItem(class_name))
            class_table.setItem(i, 1, QTableWidgetItem(str(class_idx)))
        
        stats_layout.addWidget(class_table)
        
        # Pestaña 3: Gráficos
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout(graphs_tab)
        
        # Crear figura para los gráficos
        fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(fig)
        graphs_layout.addWidget(self.canvas)
        
        # Botón para actualizar gráficos
        update_button = QPushButton("Actualizar Gráficos")
        update_button.clicked.connect(self.update_graphs)
        graphs_layout.addWidget(update_button)
        
        # Añadir pestañas al widget principal
        tabs.addTab(detection_tab, "Detección")
        tabs.addTab(stats_tab, "Estadísticas")
        tabs.addTab(graphs_tab, "Gráficos")
        
        layout.addWidget(tabs)
    
    def toggle_camera(self):
        """Alternar entre encender y apagar la cámara"""
        if self.timer.isActive():
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_button.setText("Iniciar Cámara")
        else:
            self.camera = cv2.VideoCapture(0)
            self.timer.start(30)  # 30ms = ~33fps
            self.camera_button.setText("Detener Cámara")
    
    def update_frame(self):
        """Actualizar el frame de la cámara y realizar predicción"""
        ret, frame = self.camera.read()
        if ret:
            # Preprocesar frame para predicción
            resized = cv2.resize(frame, tuple(self.model_info['image_size']))
            input_arr = tf.keras.preprocessing.image.img_to_array(resized)
            input_arr = np.expand_dims(input_arr, axis=0)
            
            # Realizar predicción
            predictions = self.model.predict(input_arr, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx]
            
            # Obtener nombre de la clase
            class_names = {v: k for k, v in self.model_info['classes'].items()}
            predicted_class = class_names[pred_idx]
            
            # Actualizar etiqueta de predicción
            self.prediction_label.setText(
                f"Predicción: {predicted_class}\nConfianza: {confidence:.2%}")
            
            # Convertir frame para mostrar
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                800, 600, Qt.AspectRatioMode.KeepAspectRatio))
    
    def capture_frame(self):
        """Capturar el frame actual"""
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captura_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                self.captured_frames.append(filename)
                print(f"Frame guardado como {filename}")
    
    def load_image(self):
        """Cargar una imagen desde archivo"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        
        if filename:
            # Cargar y mostrar imagen
            image = cv2.imread(filename)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                800, 600, Qt.AspectRatioMode.KeepAspectRatio))
            
            # Realizar predicción
            resized = cv2.resize(image, tuple(self.model_info['image_size']))
            input_arr = tf.keras.preprocessing.image.img_to_array(resized)
            input_arr = np.expand_dims(input_arr, axis=0)
            
            predictions = self.model.predict(input_arr, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx]
            
            class_names = {v: k for k, v in self.model_info['classes'].items()}
            predicted_class = class_names[pred_idx]
            
            self.prediction_label.setText(
                f"Predicción: {predicted_class}\nConfianza: {confidence:.2%}")
            
            # Guardar predicción en el historial
            self.predictions_history.append({
                'class': predicted_class,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
    
    def update_graphs(self):
        """Actualizar los gráficos de análisis"""
        if not self.predictions_history:
            return
        
        # Limpiar figura actual
        self.canvas.figure.clear()
        
        # Crear subplots
        gs = self.canvas.figure.add_gridspec(2, 2)
        ax1 = self.canvas.figure.add_subplot(gs[0, 0])
        ax2 = self.canvas.figure.add_subplot(gs[0, 1])
        ax3 = self.canvas.figure.add_subplot(gs[1, :])
        
        # Gráfico 1: Distribución de predicciones
        predictions = [p['class'] for p in self.predictions_history]
        unique, counts = np.unique(predictions, return_counts=True)
        ax1.pie(counts, labels=unique, autopct='%1.1f%%')
        ax1.set_title('Distribución de Predicciones')
        
        # Gráfico 2: Histograma de confianza
        confidences = [p['confidence'] for p in self.predictions_history]
        ax2.hist(confidences, bins=20)
        ax2.set_title('Distribución de Confianza')
        ax2.set_xlabel('Confianza')
        ax2.set_ylabel('Frecuencia')
        
        # Gráfico 3: Línea temporal de confianza
        timestamps = [p['timestamp'] for p in self.predictions_history]
        ax3.plot(timestamps, confidences)
        ax3.set_title('Confianza a lo largo del tiempo')
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Confianza')
        plt.xticks(rotation=45)
        
        # Ajustar layout y mostrar
        self.canvas.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TomatoDiseaseDetector()
    window.show()
    sys.exit(app.exec())