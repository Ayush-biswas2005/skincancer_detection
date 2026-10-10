import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages' # Needed for error flashing

# Configuration for upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists locally
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if uploaded file has a valid image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main upload homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, validation, and error handling."""
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser submits an empty file without filename
    if file.filename == '':
        flash('No image selected for uploading.')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # --- MODEL INTEGRATION PLACEHOLDER (Waiting for Member 2) ---
        # When Member 2 provides skin_model.pth and preprocess logic, 
        # we will plug it in right here.
        prediction_result = "Pending Member 2 Model Integration"
        confidence_score = "N/A"
        # -----------------------------------------------------------
        
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