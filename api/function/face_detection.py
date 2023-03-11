import cv2
import numpy as np

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    return faces