from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, callbacks

DATASET_PATH = "dataset/"
IMG_SIZE = (128, 128)

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

train_data = train_gen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=32,
    class_mode="binary", subset="training",
)
val_data = val_gen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=32,
    class_mode="binary", subset="validation", shuffle=False,
)

base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[stop])

loss, acc = model.evaluate(val_data)
print(f"Validation accuracy: {acc:.2f}")
model.save("image_classifier.h5")