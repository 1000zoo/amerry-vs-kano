from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os
tf.compat.v1.disable_eager_execution()

MODEL_PATH = "/Users/1000zoo/Desktop/amka6_binary_fine_tuning.h5"
IMG_PATH = "/Users/1000zoo/Desktop/test"

def category_preds(preds):
    category = ["Amerry", "Kano", "Others"]
    print(category[np.argmax(preds)], np.max(preds), "%")
    print(preds)
    return(category[np.argmax(preds)])

def binary_preds(preds):
    print(preds)
    print(preds.shape)
    print(preds[0])
    if preds[0][0] < 0.5:
        print("Amerry", 1-preds[0][0], "%")
        return "Amerry"
    else:
        print("Kano", preds[0][0], "%")
        return "Kano"

def gradCAM(model, x):
    conv = model.get_layer("vgg16")
    max_output = conv.outputs
    last_conv_layer = conv.get_layer("block5_conv3")

    grads = K.gradients(max_output, last_conv_layer.output)[0]
    print(grads)
    pooled_grads = K.mean(grads, axis=(0,1,2))

    iterate = K.function([conv.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap, conv_layer_output_value, pooled_grads_value

def main():

    for i, image in enumerate(os.listdir(IMG_PATH)):
        image_path = os.path.join(IMG_PATH, image)

        img = pp.image.load_img(image_path, target_size = (256,256))
        img_tensor = pp.image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = preprocess_input(img_tensor)

        model = load_model(MODEL_PATH)
        heatmap, conv_output, pooled_grads = gradCAM(model, img_tensor)

        img = cv2.imread(image_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap*0.4 + img
        
        preds = model.predict(img_tensor)
        p = binary_preds(preds)

        cv2.imwrite('grad_predict' + p + str(i) + '.jpg', superimposed_img)

if __name__ == "__main__":
    main()