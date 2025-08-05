from flask import Flask, send_from_directory, request, jsonify, session, redirect, url_for
import os
import subprocess

app = Flask(__name__, static_folder='web', static_url_path='')
app.secret_key = '1234567'  # Replace with a secure key for production

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ===== Login Routes =====
@app.route('/login', methods=['GET'])
def show_login():
    return send_from_directory('web', 'login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Replace this with your actual authentication logic
    if username == 'admin' and password == 'admin123':
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('home'))
    else:
        return "Invalid credentials", 401

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('show_login'))

# ===== Protected Routes =====
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('show_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def home():
    return send_from_directory('web', 'surveillance.html')

@app.route('/about_us')
@login_required
def about_us():
    return send_from_directory('web', 'Aboutus.html')

@app.route('/contact_us')
@login_required
def contact_us():
    return send_from_directory('web', 'Contactus.html')

@app.route('/crowd_detection')
@login_required
def crowd_detection():
    return send_from_directory('web', 'CrowdDetection.html')

@app.route('/fall_detection')
@login_required
def fall_detection():
    return send_from_directory('web', 'FallDetection.html')

@app.route('/helmet_detection')
@login_required
def helmet_detection():
    return send_from_directory('web', 'Helmet.html')

@app.route('/accident_detection')
@login_required
def accident_detection():
    return send_from_directory('web', 'Accident.html')

# ===== Shared Upload Route =====
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return jsonify({'success': True, 'filename': filename}), 200

# ===== Fall Detection =====
@app.route('/upload_fall_video', methods=['POST'])
@login_required
def upload_fall_video():
    return upload_file()

@app.route('/detect_fall_video/<filename>')
@login_required
def detect_fall_video(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'message': 'File not found'}), 404

    subprocess.Popen(["python", "fall.py", video_path])
    return jsonify({'success': True, 'message': 'Fall detection started.'}), 200

@app.route('/start_fall_webcam_detection')
@login_required
def start_fall_webcam_detection():
    subprocess.Popen(["python", "fall.py", "webcam"], shell=True)
    return jsonify({'success': True, 'message': 'Fall detection webcam started.'}), 200

# ===== Crowd Detection =====
@app.route('/detect_video/<filename>')
@login_required
def detect_video(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'message': 'File not found'}), 404

    subprocess.Popen(["python", "run.py", video_path])
    return jsonify({'success': True, 'message': 'Crowd detection started.'}), 200

@app.route('/start_crowd_webcam_detection')
@login_required
def start_crowd_webcam_detection():
    subprocess.Popen(["python", "run.py", "webcam"], shell=True)
    return jsonify({'success': True, 'message': 'Crowd detection webcam started.'}), 200

# ===== Helmet Detection =====
@app.route('/detect_helmet/<filename>')
@login_required
def detect_helmet(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'message': 'File not found'}), 404

    subprocess.Popen(["python", "helmet.py", video_path])
    return jsonify({'success': True, 'message': 'Helmet detection started.'}), 200

@app.route('/start_helmet_webcam_detection')
@login_required
def start_helmet_webcam_detection():
    subprocess.Popen(["python", "helmet.py", "webcam"], shell=True)
    return jsonify({'success': True, 'message': 'Helmet webcam detection started.'}), 200

# ===== Accident Detection =====
@app.route('/detect_accident/<filename>')
@login_required
def detect_accident(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'message': 'File not found'}), 404

    subprocess.Popen(["python", "evaluate.py", video_path])
    return jsonify({'success': True, 'message': 'Accident detection started.'}), 200

@app.route('/start_accident_webcam_detection')
@login_required
def start_accident_webcam_detection():
    subprocess.Popen(["python", "evaluate.py", "webcam"], shell=True)
    return jsonify({'success': True, 'message': 'Accident webcam detection started.'}), 200

# ===== Static Files =====
@app.route('/web/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join('web', 'css'), filename)

@app.route('/web/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join('web', 'js'), filename)

@app.route('/web/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(os.path.join('web', 'images'), filename)

# ===== Run App =====
if __name__ == '__main__':
    app.run(debug=True)