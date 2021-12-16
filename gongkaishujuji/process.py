import cv2
import os
import numpy as np
a=1

###############
#这块吧结构图边角为零的代码删除了
#############


###################这里是拼接
for i in os.listdir('./test/A'):
    path=os.path.join('./test/A',i)
    path1=path.replace('A.jpg','B.jpg')
    path1=path1.replace('A','B')
    img=cv2.imread(path)
    img1=cv2.imread(path1)
    img2=np.hstack((img,img1))
    cv2.imwrite('./train/test/{}.jpg'.format(a),img2,[int(cv2.IMWRITE_JPEG_QUALITY),100])
    a=a+1



#############这里是把造影图边角为零的图片删除
# a=0
# for i in range(8):
#     if i!=0:
#         for j in range(48):
#             img=cv2.imread('./ABC/{}C_{}.jpg'.format(i+1,j+1))
#             img1=cv2.imread('./ABC/{}A_{}.jpg'.format(i+1,j+1))
#             img2=img[0][0][0]
#             img3=img[0][255][0]
#             img4=img[255][0][0]
#             img5=img[255][255][0]
#             img6=img1[0][0][0]
#             img7=img1[0][255][0]
#             img8=img1[255][0][0]
#             img9=img1[255][255][0]
#             if img2<3 or img3<3 or img4<3 or img5<3 or img6<3 or img7<3 or img8<3 or img9<3:
#                 print(i)
#             else:
#                 a=a+1
#                 cv2.imwrite('./test/A/{}A.jpg'.format(a),img1,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#                 cv2.imwrite('./test/B/{}B.jpg'.format(a),img,[int(cv2.IMWRITE_JPEG_QUALITY),100])

#
# for i in range(1078):
#     img=cv2.imread('./train/A/{}A.jpg'.format(i)):


# for i in os.listdir('./test/B'):
#     path=os.path.join('./test/B',i)
#     print(path)
#     path1=path.replace('.j','B.j')
#     print(path1)
#     os.rename(path,path1)




