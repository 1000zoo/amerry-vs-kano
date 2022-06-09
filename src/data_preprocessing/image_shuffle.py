import random
import os

def shuffle_images(path):
    image_list = os.listdir(path=path)
    dir_name = path.split("/")[-1]
    path += "/"

    new_name_list = []
    for num, fname in enumerate(image_list):
        new_name = dir_name
        extension = fname.split(".")[-1]

        if num < 10:
            new_name += "00"
        elif num < 100:
            new_name += "0"
        new_name += str(num) + "." + extension
        new_name_list.append(new_name)
    random.shuffle(new_name_list)

    for image, new in zip(image_list, new_name_list):
        old_name = os.path.join(path, image)
        new_name = os.path.join(path, new)
        os.rename(old_name, new_name)

shuffle_images("/Users/1000zoo/Desktop/project_train/amerry")
shuffle_images("/Users/1000zoo/Desktop/project_train/kano")
shuffle_images("/Users/1000zoo/Desktop/project_test/amerry")
shuffle_images("/Users/1000zoo/Desktop/project_test/kano")
shuffle_images("/Users/1000zoo/Desktop/project_val/amerry")
shuffle_images("/Users/1000zoo/Desktop/project_val/kano")

