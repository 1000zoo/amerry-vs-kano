from PIL import Image
import os

OPATH = "/Users/1000zoo/Desktop/new__/old-dachshund/old/"
NPATH = "/Users/1000zoo/Desktop/new__/old-dachshund/new/"
def wtoj(img_path):
    try:
        img_name = img_path.split("/")[-1]
        if img_name.split(".")[-1] == "webp":
            img_name = convert_extention(img_name)
            new_img_path = NPATH + img_name
            im = Image.open(img_path).convert("RGB")
            im.save(new_img_path, "jpeg")
    except IndexError:
        return

def convert_extention(file):
    name = ""
    file_split = file.split(".")[:-1]
    for s in file_split:
        name += s
    name += ".jpg"
    return name


def get_img_list(dir_list):
    img_list = []
    for dir in dir_list:
        img_list.append(OPATH + dir)
    return img_list

def main():
    dir_list = os.listdir(OPATH)
    img_list = get_img_list(dir_list)

    for img in img_list:
        wtoj(img)

if __name__ == "__main__":
    main()