from tensorflow.keras import models

model = models.load_model("models/amka2_categorical_pretraining.h5")

model.summary()

# conv = model.layers[0]
# conv.summary()
# for layer in conv.layers:
#     if layer.name.startswith("block5_conv3"):
#         layer.trainable = True
# conv.summary()