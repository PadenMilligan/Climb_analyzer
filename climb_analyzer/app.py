from flask import Flask, render_template, request, send_from_directory, jsonify, send_file
import os
import time
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
                
                # Verify files exist and are ready
                if not os.path.exists(result_path):
                    print(f"[ERROR] Result video not found: {result_path}")
                    return render_template('index.html', error=f"Video file not found: {result_filename}")
                if not os.path.exists(metrics_path):
                    print(f"[ERROR] Metrics file not found: {metrics_path}")
                    return render_template('index.html', error=f"Metrics file not found: {metrics_filename}")
                
                # Verify video file is complete (not still being written)
                video_size = os.path.getsize(result_path)
                if video_size == 0:
                    print(f"[ERROR] Video file is empty: {result_path}")
                    return render_template('index.html', error=f"Video file is empty: {result_filename}")
                
                # Wait a moment to ensure file is fully flushed (especially on Windows)
                time.sleep(0.2)
                final_size = os.path.getsize(result_path)
                if final_size != video_size:
                    # File size changed, wait a bit more
                    time.sleep(0.3)
                    final_size = os.path.getsize(result_path)
                
                print(f"[DEBUG] Video file verified: {final_size / (1024*1024):.2f} MB")
                
                # Load metrics for display
                try:
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics_data = json.load(f)
                    print(f"[DEBUG] Loaded metrics data. Keys: {list(metrics_data.keys())}")
                    if 'average_scores' in metrics_data:
                        print(f"[DEBUG] Average scores: {metrics_data['average_scores']}")
                except Exception as e:
                    print(f"[ERROR] Failed to load metrics: {e}")
                    return render_template('index.html', error=f"Failed to load metrics data: {str(e)}")
                
                # Determine video MIME type based on extension
                video_ext = os.path.splitext(result_filename)[1].lower()
                video_type = 'video/mp4' if video_ext == '.mp4' else 'video/x-msvideo' if video_ext == '.avi' else 'video/mp4'
                
                video_url = f"/static/results/{timestamp}/{result_filename}"
                print(f"[DEBUG] Rendering template with:")
                print(f"  video_url: {video_url}")
                print(f"  video_type: {video_type}")
                print(f"  success: True")
                print(f"  analysis_type: single")
                print(f"  metrics_data present: {metrics_data is not None}")
                
                return render_template('index.html',
                                     video_url=video_url,
                                     video_type=video_type,
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
    
    # Check file size and verify it's readable
    try:
        file_size = os.path.getsize(full_path)
        print(f"[DEBUG] File size: {file_size / (1024*1024):.2f} MB")
        
        # For video files, verify they're not still being written
        if filename.endswith(('.mp4', '.avi', '.mov')):
            if file_size == 0:
                print("[ERROR] Video file is empty")
                return "Video file is empty", 404
            
            # Try to open the file to ensure it's not locked
            try:
                with open(full_path, 'rb') as test_file:
                    test_file.read(1)  # Read first byte to verify file is accessible
            except IOError as e:
                print(f"[ERROR] Cannot read video file (may still be writing): {e}")
                # Wait a moment and try again
                time.sleep(0.5)
                try:
                    with open(full_path, 'rb') as test_file:
                        test_file.read(1)
                except IOError:
                    return "Video file is not ready", 503  # Service Unavailable
    except OSError as e:
        print(f"[ERROR] Cannot access file: {e}")
        return "File access error", 500
    
    # Determine MIME type - MP4 files are assumed to be H.264 for web playback
    if filename.endswith('.mp4') or filename.endswith('.mov'):
        mimetype = 'video/mp4'  # H.264 MP4 is the standard web format
    elif filename.endswith('.avi'):
        mimetype = 'video/x-msvideo'
    elif filename.endswith('.json'):
        mimetype = 'application/json'
    else:
        mimetype = 'application/octet-stream'
    
    # Add headers for better video playback support (range requests for seeking)
    # Use conditional_response=True to support range requests properly
    response = send_file(full_path, mimetype=mimetype, conditional=True)
    if filename.endswith(('.mp4', '.avi', '.mov')):
        response.headers['Accept-Ranges'] = 'bytes'
        # Add cache headers for better mobile performance
        response.headers['Cache-Control'] = 'public, max-age=3600'
        # MP4 files are optimized as H.264 for web/mobile playback
        # Don't manually set Content-Length - let Flask handle it for range requests
    return response


if __name__ == "__main__":
    # Use PORT from environment variable (for cloud deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
