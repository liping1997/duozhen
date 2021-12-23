import numpy as np
import cv2
from crop import cutimg



def ddd(img):
    H,W,C=img.shape
    IMG=img.copy()
    for i in range(H):
        for j in range(W):
            for k in range(C):
                y = 1 - 1 / (1 + np.exp(-0.03 * (j-64)))
                IMG[i][j][k]=IMG[i][j][k]*y
    return IMG
def eee(img):
    H,W,C=img.shape
    IMG=img.copy()
    for i in range(H):
        for j in range(W):
            for k in range(C):
                y = 1 / (1 + np.exp(-0.03 * (j-64)))
                IMG[i][j][k]=IMG[i][j][k]*y
    return IMG
def fff(img):
    H,W,C=img.shape
    IMG=img.copy()
    for i in range(H):
        for j in range(W):
            for k in range(C):
                y = 1 - 1 / (1 + np.exp(-0.03 * (i-64)))
                IMG[i][j][k]=IMG[i][j][k]*y
    return IMG
def ggg(img):
    H,W,C=img.shape
    IMG=img.copy()
    for i in range(H):
        for j in range(W):
            for k in range(C):
                y = 1 / (1 + np.exp(-0.03 * (i-64)))
                IMG[i][j][k]=IMG[i][j][k]*y
    return IMG


# def imgFusion(img1, img2, img3, overlap, left_right=True):
#     """
#     图像加权融合
#     :param img3:
#     :param img5:
#     :param img4:
#     :param img1:
#     :param img2:
#     :param overlap: 重合长度
#     :param left_right: 是否是左右融合
#     :return:
#     """
#     # 这里先暂时考虑平行向融合
#
#     if left_right:  # 左右融合
#         col, row,num = img1.shape
#         img_new = np.zeros((row, 2 * col,3))
#         img_new=np.uint8(img_new)
#         img_new[:, :overlap] = img1[:,:overlap]
#         img_new[:, overlap:overlap * 2] = ddd(img1[:, overlap:overlap * 2] )+ eee(img2[:, :overlap])
#         img_new[:, overlap * 2:overlap * 3] = ddd(img2[:,overlap:overlap * 2] )+ eee(img3[:,:overlap])
#         img_new[:, overlap * 3:overlap * 4] =  img3[:,overlap:overlap*2]
#
#
#     else:  # 上下融合
#         row, col,num = img1.shape
#         img_new = np.zeros((2 * row, col,3))
#         img_new=np.uint8(img_new)
#         img_new[:overlap, :] = img1[:overlap,:]
#         img_new[overlap:overlap * 2, :] = fff(img1[overlap:overlap * 2, :]) + ggg(img2[:overlap, :])
#         img_new[overlap * 2:overlap * 3, :] = fff(img2[overlap:overlap * 2, :]) + ggg(img3[:overlap , :])
#         img_new[overlap * 3:overlap * 4, :] = img3[overlap:overlap*2 , :]
#
#
#     return img_new

def imgFusion(imglist, overlap=128, left_right=True):



    if left_right:  # 左右融合
        col, row,num = imglist[0].shape

        a=(len(imglist)+1)/2
        img_new = np.zeros((row, int(a * col),3))
        img_new=np.uint8(img_new)
        for i in range(len(imglist)+1):
            if i==0:
                img_new[:, :overlap] = imglist[i][:,:overlap]
            elif i==len(imglist):
                img_new[:,overlap*i:overlap*(i+1)]=imglist[i-1][:,overlap:overlap*2]
            else:
                img_new[:,overlap*i:overlap*(i+1)]=ddd(imglist[i-1][:,overlap:overlap*2])+eee(imglist[i][:,:overlap])



    else:  # 上下融合
        row, col,num = imglist[0].shape
        a=(len(imglist)+1)/2
        img_new = np.zeros((int(a * row), col,3))
        img_new=np.uint8(img_new)
        for i in range(len(imglist) + 1):
            if i == 0:
                img_new[ :overlap,:] = imglist[i][:overlap,:]
            elif i == len(imglist):
                img_new[overlap * i:overlap * (i + 1),:] = imglist[i - 1][overlap:overlap * 2,:]
            else:
                img_new[overlap * i:overlap * (i + 1),:] = fff(imglist[i - 1][overlap:overlap * 2,:]) + ggg(
                    imglist[i][:overlap,:])


    return img_new





# img=cv2.imread('./0.jpg')
# a,b,c,d=cutimg(img,25)
# h1=imgFusion(a[0],128)
# h2=imgFusion(a[1],128)
# h3=imgFusion(a[2],128)
# h4=imgFusion(a[3],128)
# h5=imgFusion(a[4],128)
# h=[h1,h2,h3,h4,h5]
# dat=imgFusion(h,128,False)
# cv2.imshow('dat',dat)
# cv2.waitKey(0)




