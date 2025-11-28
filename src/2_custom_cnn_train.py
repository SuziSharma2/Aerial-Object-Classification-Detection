import tensorflow as tf
from tensorflow.keras import layers, models
from utils import plot_training
from 1_preprocessing import train_gen, valid_gen

IMG_SIZE = 224

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../models/custom_cnn.h5", save_best_only=True, monitor="val_accuracy"
)

history = model.fit(train_gen, 
                    validation_data=valid_gen, 
                    epochs=15,
                    callbacks=[checkpoint])

plot_training(history, "Custom CNN")
