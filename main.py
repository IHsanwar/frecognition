from flask import Flask
from routes.recognitionroutes import recognition_bp, UPLOAD_FOLDER # Import the blueprint only
import os
app = Flask(__name__)

# Configuration

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Register blueprint
app.register_blueprint(recognition_bp)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
