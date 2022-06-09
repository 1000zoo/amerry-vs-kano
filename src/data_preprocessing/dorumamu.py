# import os

# dirpath = "/Users/1000zoo/Desktop/"
# path = "/Users/1000zoo/Desktop/kano000jpeg"

# old_name = path
# fname = path.split("/")[-1]
# path = path.split("/")[:-1]
# print(fname)
# new_name = dirpath

# fname = fname[:7] + "." + fname[7:]
# print(fname)

# os.rename(old_name, new_name + fname)

import os

dirpath = "/Users/1000zoo/Desktop/amerry"
classlen = len(dirpath.split("/")[-1]) + 3
dirpath += "/"

for fname in os.listdir(dirpath):
    new_fname = fname[:classlen] + "." + fname[classlen:]
    os.rename(dirpath+fname, dirpath+new_fname)