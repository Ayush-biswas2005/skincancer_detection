import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from src.predict import predictor  # Import our fully operational model predictor

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading.')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run real AI inference using our MobileNetV3 model
        prediction_result, confidence_score = predictor.predict_image(filepath)
        
        return render_template(
            'result.html', 
            prediction=prediction_result, 
            confidence=confidence_score, 
            image_filename=filename
        )
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)