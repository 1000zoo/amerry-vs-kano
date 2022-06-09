import random

from numpy import s_

li = [1,2,1,1,1,3,4,5]

random.shuffle(li)
print(li)
s = set(li)
l = list(s)
print(l)
