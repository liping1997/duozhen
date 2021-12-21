import cv2
import os
import torch
import numpy as np




def cutimg(img, num,overlap_factor=128):


    """a,b,c,d,分别存储A,B1,B2,B3的256*256块"""
    factor = int(np.sqrt(num))
    a=[]
    a1=[]
    b1=[]
    c1=[]
    d1=[]
    b=[]
    c=[]
    d=[]
    for i in range(factor):
        a1 = []
        b1 = []
        c1 = []
        d1 = []
        for ii in range(factor):
            img_temp1 = img[i * overlap_factor:(i + 2) * overlap_factor, ii * overlap_factor:(ii + 2) * overlap_factor]
            img_temp2 = img[i * overlap_factor:(i + 2) * overlap_factor, (ii+4) * overlap_factor:(ii + 6) * overlap_factor]
            img_temp3 = img[i * overlap_factor:(i + 2) * overlap_factor, (ii+8) * overlap_factor:(ii + 10) * overlap_factor]
            img_temp4 = img[i * overlap_factor:(i + 2) * overlap_factor, (ii+12) * overlap_factor:(ii + 14) * overlap_factor]
            a1.append(img_temp1)
            b1.append(img_temp2)
            c1.append(img_temp3)
            d1.append(img_temp4)
        a.append(a1)
        b.append(b1)
        c.append(c1)
        d.append(d1)
    return a,b,c,d

# img=cv2.imread('./data_test/173.jpg')
# a,b,c,d=cutimg(img,9)
# for i in range(9):
#     img=np.hstack((a[i],b[i],c[i],d[i]))
#     cv2.imwrite('./save/{}.jpg'.format(i),img)

# img=cv2.imread('./0.jpg')
# a,b,c,d=cutimg(img,25)
# cv2.imshow('1',a[2])
# cv2.waitKey(0)