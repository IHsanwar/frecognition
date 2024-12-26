from flask import Blueprint, render_template, request, jsonify, current_app
import face_recognition
import numpy as np
import cv2
import base64
import os
import uuid  # For generating unique filename
from routes.absensiroute import *


recognition_bp = Blueprint('recognition', __name__)

UPLOAD_FOLDER = 'facedata'
# Define known faces storage
known_faces_dir = 'facedata'
known_faces_encodings = []
known_faces_names = []

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


for filename in os.listdir(known_faces_dir):
    # Check if the file has a valid image extension
    if os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS:
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            known_faces_encodings.append(face_encodings[0])
            known_faces_names.append(os.path.splitext(filename)[0])
    else:
        print(f"Skipping non-image file: {filename}")

@recognition_bp.route('/')
def index():
    return render_template('index.html')

@recognition_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file part'
    
    file = request.files['image']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
        return 'File uploaded successfully'
    else:
        return 'Invalid file type'
    
    
@recognition_bp.route('/process_image', methods=['POST'])
def process_image():
    data = request.json['image']
    image_data = base64.b64decode(data.split(",")[1])  # Remove the header of the base64 string
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get original frame dimensions
    frame_height, frame_width, _ = rgb_frame.shape

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    name = "Unknown"  # Initialize name to avoid UnboundLocalError

    for face_location, face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(distances) if distances.size > 0 else None
        if best_match_index is not None and distances[best_match_index] < 0.50:
            name = known_faces_names[best_match_index]

        top, right, bottom, left = face_location

        # Normalize coordinates (values between 0 and 1)
        results.append({
            "name": name,
            "location": {
                "top": top / frame_height,
                "right": right / frame_width,
                "bottom": bottom / frame_height,
                "left": left / frame_width
            }
        })

    # Insert the last recognized name or "Unknown"
    insertdb(name)
    return jsonify({"results": results})


@recognition_bp.route('/register-image', methods=['GET'])
def index_regis():
    # Render the HTML registration page
    return render_template('regisimage.html')  




@recognition_bp.route('/register-image', methods=['POST'])
def register_image():
    try:
        data = request.json['image']
        name = request.json['name']

        # Decode the image from base64
        image_data = base64.b64decode(data.split(",")[1])
        np_img = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # Use current_app to access the app config (UPLOAD_FOLDER)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        filename = f"{name}_{uuid.uuid4()}.jpg"
        filepath = os.path.join(upload_folder, filename)

        # Save the image to the specified upload folder
        cv2.imwrite(filepath, frame)
        
        # Perform face recognition and register the face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        if not face_locations:
            return jsonify({"error": "No face detected in the image"}), 400

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_encoding = face_encodings[0]

        known_faces_encodings.append(face_encoding)
        known_faces_names.append(name)

        return jsonify({"message": f"Face registered for {name}", "image_path": filepath}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500