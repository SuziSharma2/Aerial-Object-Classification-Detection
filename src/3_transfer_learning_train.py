import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from utils import plot_training
from 1_preprocessing import train_gen, valid_gen

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../models/mobilenet_best.h5", save_best_only=True, monitor="val_accuracy"
)

history = model.fit(train_gen, 
                    validation_data=valid_gen, 
                    epochs=10,
                    callbacks=[checkpoint])

plot_training(history, "MobileNet Transfer Learning")
