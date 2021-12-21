import cv2
import os
import numpy as np
from crop import *
from merge import *
import math

"""
下面这段代码主要用于裁剪图片，生成用于测试所需要的图片,保存在datasets/aaaaa/test文件夹下      hstack水平拼接 vstack垂直拼接
"""

def generatetestpicture(test_file,num,save_file):         #将test_file文件夹下的图片分成num份，拼接之后保存在save_file中
    dir_num=len(os.listdir(test_file))
    for k in range(dir_num):
        img = cv2.imread('{}/{}.jpg'.format(test_file,k))
        a, b, c, d = cutimg(img,num)       #核心代码--切块
        for i in range(int(math.sqrt(num))):
            for j in range(int(math.sqrt(num))):
                img1 = np.hstack((a[i][j], b[i][j], c[i][j], d[i][j]))
                cv2.imwrite('{}/{}_{}_{}.jpg'.format(save_file,k,i,j), img1,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

