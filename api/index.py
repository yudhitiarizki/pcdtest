from flask import Flask, render_template, request, send_file
import cv2
import os
import uuid
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/up')
def up():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    # read image file
    image_file = request.files['image']
    ext = os.path.splitext(image_file.filename)[1]
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    # blur faces in the image
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        face_roi_blur = cv2.GaussianBlur(face_roi, (51, 51), 33)
        image[y:y+h, x:x+w] = face_roi_blur

    # generate a unique filename with the original file extension
    filename = str(uuid.uuid4()) + ext
    filepath = os.path.join('./static/images/', filename)

    # save the blurred image to a file
    cv2.imwrite(filepath, image)

    # return the name of the blurred image file
    return render_template('result.html', filename=filename)


@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join('./static/images/', filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
