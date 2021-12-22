import cv2
import os


###########可以将train中有缺陷图片删除
# a=0
# for i in os.listdir('./dataset/train'):
#     path=os.path.join('./dataset/train',i)
#     img=cv2.imread(path)
#     img1=img[0,0,0]
#     img2 = img[0, 511, 0]
#     img3 = img[0, 1023, 0]
#     img4 = img[0, 1535, 0]
#     img5 = img[0, 2023, 0]
#     img6 = img[511, 0, 0]
#     img7 = img[511, 511, 0]
#     img8 = img[511, 1023, 0]
#     img9 = img[511, 1535, 0]
#     img10 = img[511, 2023, 0]
#
#     if img1>0:
#         if img2<1 or img3<1 or img4<1 or img5<1 or img6<1 or img7<1 or img8<1 or img9<1 or img10<1:
#             print(i)
#         else:
#             cv2.imwrite('./train/{}.jpg'.format(a),img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#             a=a+1
#     else:
#         if img2<1 and img6<1 and img7<1:
#             cv2.imwrite('./train/{}.jpg'.format(a), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#             a = a + 1
