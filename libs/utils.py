import cv2
import numpy as np
import pyaudio

def img2fre(img):
    return np.fft.fft2(img)