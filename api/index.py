from flask import Flask, render_template, request, send_file
import cv2
import os
import uuid
from function.image_blur import blur_faces
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
    image_file = request.files['image']
    ext = os.path.splitext(image_file.filename)[1]
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Blur faces in the image
    image_blurred = blur_faces(image)

    # Generate a unique filename with the original file extension
    filename = str(uuid.uuid4()) + ext
    filepath = os.path.join('./static/images/', filename)

    # Save the blurred image to a file
    cv2.imwrite(filepath, image_blurred)

    # Return the name of the blurred image file
    return render_template('result.html', filename=filename)

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join('./static/images/', filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
