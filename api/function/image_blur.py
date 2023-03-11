import cv2
import numpy as np
from function.face_detection import detect_faces

def blur_faces(image):
    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]

        face_roi_blur = cv2.GaussianBlur(face_roi, (51, 51), 33)

        image[y:y+h, x:x+w] = face_roi_blur

    return image
