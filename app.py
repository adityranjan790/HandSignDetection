from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["A", "B", "C", "D", "F", "G", "H", "I", "J", "K", "L", "N", "O", "P", "Q", "R", "T", "U", "V", "W", "X", "Y"]


def detect_sign(image_data):
    # Decode the image data
    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255
        imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]
        aspectRatio = h / w
        if aspectRatio > 1:
            k = 300 / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, 300))
            wGap = math.ceil((300 - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = 300 / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (300, hCal))
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        return labels[index]
    return ""


@app.route('/detect_sign', methods=['POST'])
def get_letter():
    print("POST request received")  # Debug statement
    data = request.get_json()
    image_data = data.get("image")
    letter = detect_sign(image_data)  # Pass the image data to the detection function
    return jsonify({"sign": letter})  # Change 'letter' to 'sign' for consistency with client code


if __name__ == "__main__":
    app.run(debug=True)
