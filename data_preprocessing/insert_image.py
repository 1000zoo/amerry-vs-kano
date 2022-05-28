import random
import os

def insert_image(path):
    image_list = os.listdir(path)
    dir_name = path.split("/")[-1].split("_")[-1]
    path += "/"
    new_name_list = []

    for num in range(len(image_list)):
        rand = int(random.random()*1000)
        new_name = dir_name
        if rand < 10:
            new_name += "00"
        elif rand < 100:
            new_name += "0"
        new_name += str(rand) + "-" + str(num) + ".jpg"
        new_name_list.append(new_name)

    for image, new in zip(image_list, new_name_list):
        old_name = os.path.join(path, image)
        new_name = os.path.join(path, new)
        os.rename(old_name, new_name)

insert_image("/Users/1000zoo/Desktop/ann-project/new_kano")
insert_image("/Users/1000zoo/Desktop/ann-project/new_amerry")