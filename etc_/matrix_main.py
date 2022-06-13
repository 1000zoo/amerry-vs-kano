from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

TEST_DIR = "/Users/1000zoo/Desktop/ann-project/kamerry-data-set/just-test-data"
MODEL_DIR = "/Users/1000zoo/Desktop/ann-project/result-models/"

def print_matrix(model, model_name):
    input_shape = input_shape_of(model)
    test_data = test_generator(input_shape)
    y_test = test_data.classes
    y_pred = model.predict_generator(test_data)
    matrix = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
    print_result(model_name, matrix)

def print_result(model_name, matrix):
    print()
    print("="*30)
    print(model_name)
    print("="*30)
    print_format(matrix)
    print("accuracy:", accuracy(matrix))
    print()

def accuracy(matrix):
    l = len(matrix)
    m = 0
    for i in range(l):
        m += matrix[i][i]
    sum = np.sum(matrix)
    return m / sum

def print_format(matrix):
    print("\tAmerry\tKano\tOther\n")
    index_list = ["Amerry", "Kano", "Other"]
    for m1, l in zip(matrix, index_list):
        print(l, end="\t")
        for m2 in m1:
            print(m2, end="\t")
        print("\n")

def test_generator(input_shape, batch_size=20, class_mode="categorical"):
    data = ImageDataGenerator(rescale=1./255)
    if input_shape[0] == None:
        return data.flow_from_directory(
            TEST_DIR,
            batch_size = batch_size,
            class_mode = class_mode,
            shuffle = False # ***********
        )
    else:
        return data.flow_from_directory(
            TEST_DIR,
            target_size = input_shape,
            batch_size = batch_size,
            class_mode = class_mode,
            shuffle = False # ***********
        )

def input_shape_of(model):
    return (model.input.shape[1], model.input.shape[1])

def main():
    model_list = os.listdir(MODEL_DIR)

    for model in model_list:
        if model == ".DS_Store":
            continue
        m = load_model(os.path.join(MODEL_DIR+model))
        print_matrix(m, model)


if __name__ == "__main__":
    main()