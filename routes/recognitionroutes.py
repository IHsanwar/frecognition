from flask import Blueprint, render_template, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64
import os

recognition_bp = Blueprint('recognition', __name__)

known_faces_dir = 'facedata'
known_faces_encodings = []
known_faces_names = []

# Load known faces
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        known_faces_encodings.append(face_encodings[0])
        known_faces_names.append(os.path.splitext(filename)[0])

@recognition_bp.route('/')
def index():
    return render_template('index.html')

@recognition_bp.route('/process_image', methods=['POST'])
def process_image():
    # Get the base64 image data from the request
    data = request.json['image']
    image_data = base64.b64decode(data.split(",")[1])  # Remove the header of the base64 string
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face recognition
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(distances) if distances.size > 0 else None
        if best_match_index is not None and distances[best_match_index] < 0.65:
            name = known_faces_names[best_match_index]
        else:
            name = "Unknown"
        names.append(name)
    
    return jsonify({"names": names})
