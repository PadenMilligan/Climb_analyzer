from flask import Flask, render_template, request, send_from_directory, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from pose_processing import process_climber_videos, analyze_single_video
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type', 'single')
        
        if analysis_type == 'single':
            # Single video analysis with metrics
            file = request.files.get('video')
            if not file:
                return render_template('index.html', error="Please upload a video.")
            
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            # Create a timestamped output folder
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_subdir = os.path.join(app.config['RESULT_FOLDER'], timestamp)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Analyze video with metrics
            result_path, metrics_path = analyze_single_video(path, output_subdir)
            
            if result_path and metrics_path:
                result_filename = os.path.basename(result_path)
                metrics_filename = os.path.basename(metrics_path)
                
                # Load metrics for display
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                
                return render_template('index.html',
                                     video_url=f"/static/results/{timestamp}/{result_filename}",
                                     metrics_url=f"/static/results/{timestamp}/{metrics_filename}",
                                     metrics_data=metrics_data,
                                     success=True,
                                     analysis_type='single')
            else:
                return render_template('index.html', error="Failed to analyze video.")
        
        else:
            # Dual video comparison (original functionality)
            file1 = request.files.get('video1')
            file2 = request.files.get('video2')

            if not file1 or not file2:
                return render_template('index.html', error="Please upload two videos.")

            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file1.save(path1)
            file2.save(path2)

            # Create a timestamped output folder
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_subdir = os.path.join(app.config['RESULT_FOLDER'], timestamp)
            os.makedirs(output_subdir, exist_ok=True)

            # Process uploaded videos
            result_path = process_climber_videos(path1, path2, output_subdir)
            if result_path:
                result_filename = os.path.basename(result_path)
                return render_template('index.html',
                                     video_url=f"/static/results/{timestamp}/{result_filename}",
                                     success=True,
                                     analysis_type='dual')
            else:
                return render_template('index.html', error="Failed to process videos.")

    return render_template('index.html')

@app.route('/static/results/<timestamp>/<filename>')
def serve_video(timestamp, filename):
    full_path = os.path.abspath(os.path.join(app.config['RESULT_FOLDER'], timestamp, filename))
    print(f"[DEBUG] Serving file from: {full_path}")
    if not os.path.exists(full_path):
        print("[ERROR] File does not exist:", full_path)
        return "File not found", 404
    
    # Determine MIME type
    if filename.endswith('.mp4') or filename.endswith('.mov'):
        mimetype = 'video/mp4'
    elif filename.endswith('.json'):
        mimetype = 'application/json'
    else:
        mimetype = 'application/octet-stream'
    
    return send_file(full_path, mimetype=mimetype)


if __name__ == "__main__":
    # Use PORT from environment variable (for cloud deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
