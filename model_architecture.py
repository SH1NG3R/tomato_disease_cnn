import tensorflow as tf
sns.set_style('darkgrid')
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
print ('modules loaded')
def create_model():
    """Recrear la arquitectura del modelo original"""
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = 11  # número de clases en tu modelo

    # Crear el modelo base
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False, 
        weights=None,  # No cargar pesos de ImageNet
        input_shape=img_shape, 
        pooling='max'
    )

    # Crear el modelo completo
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.016),
                            activity_regularizer=tf.keras.regularizers.l1(0.006),
                            bias_regularizer=tf.keras.regularizers.l1(0.006),
                            activation='relu'),
        tf.keras.layers.Dropout(rate=0.45),
        tf.keras.layers.Dense(class_count, activation='softmax')
    ])

    return model

# Guardar esta función en un archivo separado llamado model_architecture.py
