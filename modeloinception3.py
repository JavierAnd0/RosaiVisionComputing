import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f" RAM Disponible: {mem.available / (1024**3):.2f} GB")

check_memory()


train_dir = r"C:\Users\Javier connor\OneDrive\Escritorio\ProyectoROSAI\DatasetAumentadoTrain"
val_dir = r"C:\Users\Javier connor\OneDrive\Escritorio\ProyectoROSAI\DatasetValidation"


img_size = (299, 299) 
batch_size = 32
epochs = 20


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

num_classes = train_generator.num_classes
print(f" Número de clases detectadas: {num_classes}")

# 📌 Modelo base
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=1
)

model.save(r"C:/Users/Javier connor/OneDrive/Escritorio/ProyectoROSAI/modelo_inception.keras")
print(" Modelo InceptionV3 entrenado y guardado correctamente")

check_memory()