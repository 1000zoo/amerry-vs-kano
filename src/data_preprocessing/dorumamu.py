
"""
image_numbering.py 에서
실수로 확장자 앞에 "." 을 추가하지 못한채로 코드를 실행하여
이를 되돌리기 위한 코드
"""


import os

dirpath = "/Users/1000zoo/Desktop/amerry"
classlen = len(dirpath.split("/")[-1]) + 3
dirpath += "/"

for fname in os.listdir(dirpath):
    new_fname = fname[:classlen] + "." + fname[classlen:]
    os.rename(dirpath+fname, dirpath+new_fname)