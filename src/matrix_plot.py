from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

TEST_DIR = "/Users/1000zoo/Desktop/ann-project/kamerry-data-set/test-for-scikit-learn"
MODEL_DIR = "/Users/1000zoo/Desktop/ann-project/result-models/"

def plot_matrix(model, model_name):

    ## https://www.delftstack.com/howto/python/plot-confusion-matrix-in-python/
    # creates confusion matrix
    input_shape = input_shape_of(model)
    test_data = test_generator(input_shape)
    y_test = test_data.classes
    y_pred = model.predict_generator(test_data)
    mat_con = confusion_matrix(y_test, np.argmax(y_pred, axis=1))

    # Setting the attributes
    fig, px = plt.subplots(figsize=(4, 4))
    px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
    label_list = ["Amerry", "Kano", "Others"]
    for m in range(mat_con.shape[0]):
        for n in range(mat_con.shape[1]):
            px.text(x=m,y=n,s=mat_con[n, m], va='center', ha='center', size='xx-large')

    # Sets the labels
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title(model_name.split(".")[0].split("_")[0], fontsize=15)
    plt.savefig(model_name.split(".")[0].split("_")[0]+".jpg")

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
        plot_matrix(m, model)


if __name__ == "__main__":
    main()