import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from 1_preprocessing import test_gen

model_path = "../models/mobilenet_best.h5"
model = tf.keras.models.load_model(model_path)

pred = model.predict(test_gen)
pred_classes = (pred > 0.5).astype("int32")

print("Classification Report:")
print(classification_report(test_gen.classes, pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(test_gen.classes, pred_classes))
