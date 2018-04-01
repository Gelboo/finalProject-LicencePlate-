# import os
# letters = ['A','B','C','D','E','F','G','H','I','J','K','L',
# 'M','N','O','P','Q','R','S','T','V','U','W','X','Y','Z']
#
# letters_low = []
# for f in letters:
#     letters_low.append(f.lower())
#
# numbers = ['0','1','2','3','4','5','6','7','8','9','10']
#
# folders = letters+letters_low+numbers
#
# print folders
#
# for f in folders:
#     if not os.path.exists(f):
#         os.makedirs(f)
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img.png')
# i,j = 0,0
# while i < 600:
#     while j < 600:
#         cv2.rectangle(img,(i,j),(i+100,j+200),(0,0,255),3)
#         j += 200
#     i += 100
#     j = 0
# cv2.imshow("image",img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# plt.imshow(img)
# plt.show()


from itertools import islice

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

for i in window(img,400 ):
    plt.imshow(i)
    plt.show()
