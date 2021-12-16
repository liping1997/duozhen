import numpy as np
import cv2


class A:
    def __int__(self,a,b):
        self.a=a
        self.b=b
    def __call__(self,c):
        print(self.a+self.b+c)

a=A(2,3)
a(5)

