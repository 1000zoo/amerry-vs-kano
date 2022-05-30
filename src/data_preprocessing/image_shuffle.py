import random
import os

def shuffle_images(path):
    image_list = os.listdir(path=path)
    dir_name = path.split("/")[-1]
    path += "/"

    image_list_len = len(image_list)
    new_name_list = []
    for num in range(image_list_len):
        new_name = dir_name
        if num < 10:
            new_name += "00"
        elif num < 100:
            new_name += "0"
        new_name += str(num) + ".jpg"
        new_name_list.append(new_name)
    random.shuffle(new_name_list)

    for image, new in zip(image_list, new_name_list):
        old_name = os.path.join(path, image)
        new_name = os.path.join(path, new)
        os.rename(old_name, new_name)



shuffle_images("/Users/1000zoo/Desktop/ann-project/kano")
shuffle_images("/Users/1000zoo/Desktop/ann-project/kano_test")
shuffle_images("/Users/1000zoo/Desktop/ann-project/kano_val")
shuffle_images("/Users/1000zoo/Desktop/ann-project/amerry")
shuffle_images("/Users/1000zoo/Desktop/ann-project/amerry_test")
shuffle_images("/Users/1000zoo/Desktop/ann-project/amerry_val")
