import cv2
import os
import numpy as np
from crop import *
from merge import *
import math



def generatetestpicture(test_folder,save_folder,num=9):         #将test_file文件夹下的图片分成num份，拼接之后保存在save_file中，save_file通常是datasets/test
    dir_num=len(os.listdir(test_folder))
    for k in range(dir_num):
        img = cv2.imread('{}/{}.jpg'.format(test_folder,k))
        a, b, c, d = cutimg(img,num)       #核心代码--切块
        for i in range(int(math.sqrt(num))):
            for j in range(int(math.sqrt(num))):
                img1 = np.hstack((a[i][j], b[i][j], c[i][j], d[i][j]))
                cv2.imwrite('{}/{}_{}_{}.jpg'.format(save_folder,k,i,j), img1,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

def finalpicture(test_folder,input_folder,num,save_folder):      #test_folder和上面一致，input_folder是results中images文件夹，save_folder一般是savewjj
    dat_list1 = []
    dat_list2 = []
    dat_list3 = []
    dir_num=len(os.listdir(test_folder))
    for k in range(dir_num):
        for i in range(num):
            for j in range(num):
                img1 = cv2.imread('{}/{}_{}_{}_{}.png'.format(input_folder,k,i, j, 'fake_B0'))
                img2 = cv2.imread('{}/{}_{}_{}_{}.png'.format(input_folder,k,i, j, 'fake_B1'))
                img3 = cv2.imread('{}/{}_{}_{}_{}.png'.format(input_folder,k,i, j, 'fake_B2'))
                dat_list1.append(img1)
                dat_list2.append(img2)
                dat_list3.append(img3)
    for i in range(dir_num):
        h1 = imgFusion(dat_list1[0+9*i:3+9*i])
        h2 = imgFusion(dat_list1[3+9*i:6+9*i])
        h3 = imgFusion(dat_list1[6+9*i:9+9*i])

        hh1=[h1,h2,h3]
        dat1=imgFusion(hh1,128,False)
        h4 = imgFusion(dat_list1[0 + 9 * i:3 + 9 * i])
        h5 = imgFusion(dat_list1[3 + 9 * i:6 + 9 * i])
        h6 = imgFusion(dat_list1[6 + 9 * i:9 + 9 * i])

        hh2 = [h4,h5,h6]
        dat2 = imgFusion(hh2, 128, False)
        h7 = imgFusion(dat_list1[0 + 9 * i:3 + 9 * i])
        h8 = imgFusion(dat_list1[3 + 9 * i:6 + 9 * i])
        h9 = imgFusion(dat_list1[6 + 9 * i:9 + 9 * i])

        hh3 = [h7,h8.h9]
        dat3 = imgFusion(hh3, 128, False)

        dat4=np.zeros((512,512,3))
        dat4=np.uint8(dat4)
        dat=np.hstack((dat4,dat1,dat2,dat3))
        img4=cv2.imread('./{}/{}.jpg'.format(test_folder,i))

        dat=np.vstack((img4,dat))


        cv2.imwrite('{}/{}.jpg'.format(save_folder,i), dat,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print("第{}张图片已经保存".format(i))