import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import json
import time
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from metrics_calculator import ClimbingMetricsCalculator

def create_video_writer(output_path, fps, frame_width, frame_height):
    """
    Create a VideoWriter with H.264 codec optimized for web playback.
    Prioritizes H.264 variants for maximum web browser compatibility.
    Returns: (VideoWriter, actual_output_path, codec_used) or None if all fail
    """
    # For web playback, H.264 is essential. Try multiple H.264 variants first
    # H.264 codecs (in order of preference for web compatibility)
    h264_codecs = [
        ('H264', 'H264'),  # H.264 (best for web, may have OpenH264 warning)
        ('avc1', 'avc1'),  # H.264 alternative (AVC1 is the web standard)
        ('X264', 'X264'),  # x264 encoder (if available)
    ]
    
    # Fallback codecs (less web-compatible, will be re-encoded to H.264 later)
    fallback_codecs = [
        ('mp4v', 'mp4v'),  # MPEG-4 Part 2 (fallback)
        ('XVID', 'XVID'),  # Xvid MPEG-4 (fallback)
    ]
    
    # Try H.264 codecs first
    for codec_name, fourcc_str in h264_codecs:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if out.isOpened():
            print(f"[DEBUG] Using H.264 codec: {codec_name} (web-optimized)")
            return (out, output_path, codec_name)
        else:
            out.release()
    
    # Try fallback codecs if H.264 fails
    for codec_name, fourcc_str in fallback_codecs:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if out.isOpened():
            print(f"[WARNING] Using fallback codec: {codec_name} (will be converted to H.264 for web)")
            return (out, output_path, codec_name)
        else:
            out.release()
    
    # If all codecs fail, try with .avi extension as fallback
    if output_path.endswith('.mp4'):
        avi_path = output_path.replace('.mp4', '.avi')
        print(f"[WARNING] MP4 codecs failed, trying AVI format: {avi_path}")
        all_codecs = h264_codecs + fallback_codecs
        for codec_name, fourcc_str in all_codecs:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(avi_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                print(f"[DEBUG] Using codec: {codec_name} with AVI format (will be converted to H.264 MP4)")
                return (out, avi_path, codec_name)
            else:
                out.release()
    
    return None

def create_graph_overlay(metrics_history, frame_width=None, frame_height=None):
    """
    Create a real-time radar chart overlay showing current metrics.
    Returns a numpy array image that can be overlaid on video frames.
    Graph size is adaptive based on video dimensions.
    """
    # Calculate graph size based on video dimensions (about 25% of width/height, square)
    if frame_width and frame_height:
        graph_size = int(min(frame_width, frame_height) * 0.25)
        # Ensure minimum and maximum sizes
        graph_size = max(250, min(graph_size, 400))
        graph_width = graph_size
        graph_height = graph_size
    else:
        graph_width = 350
        graph_height = 350
    
    if len(metrics_history) < 1:
        # Return a blank overlay if not enough data
        overlay = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
        overlay[:, :] = (20, 20, 20)  # Dark background
        return overlay
    
    # Get the most recent metrics (current values)
    current_metrics = metrics_history[-1]
    
    # Convert hip-to-wall distance to a score (0-100)
    # Closer to wall = higher score
    # Typical range: 0-200 pixels, with 0-50 being ideal (score 80-100)
    hip_to_wall_distance = current_metrics.get('hip_to_wall', 0)
    if hip_to_wall_distance <= 50:
        # Very close to wall - excellent
        hip_to_wall_score = 100 - (hip_to_wall_distance / 50) * 20  # 100 to 80
    elif hip_to_wall_distance <= 150:
        # Medium distance - good
        hip_to_wall_score = 80 - ((hip_to_wall_distance - 50) / 100) * 30  # 80 to 50
    else:
        # Far from wall - needs improvement
        hip_to_wall_score = max(0, 50 - ((hip_to_wall_distance - 150) / 100) * 50)  # 50 to 0
    
    # Prepare data for radar chart
    categories = ['Stability', 'Balance', 'Technique', 'Rhythm', 'Hip-to-Wall']
    values = [
        current_metrics['stability'],
        current_metrics['balance'],
        current_metrics['technique'],
        current_metrics['rhythm'],
        hip_to_wall_score
    ]
    
    # Number of categories
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add the first value at the end to close the polygon
    values += values[:1]
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(graph_width/100, graph_height/100), facecolor='black')
    fig.patch.set_alpha(0.85)  # Semi-transparent
    
    # Create polar subplot
    ax = fig.add_subplot(111, projection='polar', facecolor='black')
    
    # Plot the radar chart
    ax.plot(angles, values, 'o-', linewidth=3, color='#30cfd0', alpha=0.9, label='Current')
    ax.fill(angles, values, alpha=0.25, color='#30cfd0')
    
    # Set the category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=10, fontweight='bold')
    
    # Set the radial limits (0-100)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='white', fontsize=8)
    ax.grid(True, color='white', alpha=0.3, linestyle='--', linewidth=1)
    
    # Style the plot
    ax.spines['polar'].set_color('white')
    ax.spines['polar'].set_linewidth(2)
    
    # Add value labels at each point
    for angle, value, category in zip(angles[:-1], values[:-1], categories):
        # Position text slightly outside the radar
        ax.text(angle, 105, f'{int(value)}', 
                horizontalalignment='center', 
                verticalalignment='center',
                color='white', 
                fontsize=9, 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='white', linewidth=1))
    
    # Add title
    plt.title('Performance Metrics', color='white', fontsize=12, fontweight='bold', pad=20)
    
    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_img = np.asarray(buf)
    
    # Convert RGBA to BGR for OpenCV
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)  # Close to free memory
    
    return graph_img

def overlay_graph_on_frame(frame, graph_overlay, position='top_right', padding=20):
    """
    Overlay a graph image on a video frame.
    position: 'top_right', 'top_left', 'bottom_right', 'bottom_left'
    """
    frame_height, frame_width = frame.shape[:2]
    graph_height, graph_width = graph_overlay.shape[:2]
    
    # Calculate position
    if position == 'top_right':
        x = frame_width - graph_width - padding
        y = padding
    elif position == 'top_left':
        x = padding
        y = padding
    elif position == 'bottom_right':
        x = frame_width - graph_width - padding
        y = frame_height - graph_height - padding
    else:  # bottom_left
        x = padding
        y = frame_height - graph_height - padding
    
    # Ensure we don't go out of bounds
    if x < 0 or y < 0 or x + graph_width > frame_width or y + graph_height > frame_height:
        # Resize graph if needed
        if x + graph_width > frame_width:
            graph_width = frame_width - x - padding
        if y + graph_height > frame_height:
            graph_height = frame_height - y - padding
        if graph_width > 0 and graph_height > 0:
            graph_overlay = cv2.resize(graph_overlay, (graph_width, graph_height))
        else:
            return frame  # Can't fit, return original
    
    # Create ROI (Region of Interest)
    roi = frame[y:y+graph_height, x:x+graph_width]
    
    # Create a mask for non-black pixels (where graph content exists)
    mask = (graph_overlay[:, :, 0] > 30) | (graph_overlay[:, :, 1] > 30) | (graph_overlay[:, :, 2] > 30)
    
    # Blend: 75% graph, 25% original frame for better visibility
    roi_blended = cv2.addWeighted(roi, 0.25, graph_overlay, 0.75, 0)
    roi[mask] = roi_blended[mask]
    
    return frame

def process_climber_videos(path1, path2, output_folder):
    print(f"[DEBUG] Input paths:\n  path1: {path1}\n  path2: {path2}")
    print(f"[DEBUG] Output folder: {output_folder}")

    # Setup MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose1 = mp_pose.Pose()
    pose2 = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    fps = min(cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def detect_takeoff(cap, pose_instance):
        hip_y_positions = []
        frame_count = 0
        max_frames = int(fps * 5)

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_instance.process(image_rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                y = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                hip_y_positions.append(y)
            frame_count += 1

        diffs = np.diff(hip_y_positions)
        for i in range(3, len(diffs)):
            if all(d < -0.001 for d in diffs[i-3:i]):
                return i
        return 0

    takeoff1 = detect_takeoff(cv2.VideoCapture(path1), pose1)
    takeoff2 = detect_takeoff(cv2.VideoCapture(path2), pose2)
    print(f"[DEBUG] Takeoff frames: Climber 1 = {takeoff1}, Climber 2 = {takeoff2}")

    cap1.set(cv2.CAP_PROP_POS_FRAMES, takeoff1)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, takeoff2)

    video_output_path = os.path.abspath(os.path.join(output_folder, "synced_pose_output.mp4"))
    print(f"[DEBUG] Attempting to write video to: {video_output_path}")

    csv1_path = os.path.abspath(os.path.join(output_folder, "climber1_log.csv"))
    csv2_path = os.path.abspath(os.path.join(output_folder, "climber2_log.csv"))

    result = create_video_writer(video_output_path, fps, frame_width * 2, frame_height)
    if result is None:
        print(f"[ERROR] Failed to open VideoWriter. Check codec or output path.")
        return None
    
    out, video_output_path, codec_used = result

    csv1 = open(csv1_path, "w", newline="")
    csv2 = open(csv2_path, "w", newline="")
    writer1 = csv.writer(csv1)
    writer2 = csv.writer(csv2)
    writer1.writerow(["Frame", "Time (s)", "Hip_X", "Hip_Y", "Hip_Speed (px/s)"])
    writer2.writerow(["Frame", "Time (s)", "Hip_X", "Hip_Y", "Hip_Speed (px/s)"])

    def process(frame, prev_hip, pose_instance, writer, frame_idx):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_instance.process(image_rgb)
        speed_text = "Speed: N/A"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            cx = int((lh.x + rh.x) / 2 * frame.shape[1])
            cy = int((lh.y + rh.y) / 2 * frame.shape[0])
            curr_hip = np.array([cx, cy])

            if prev_hip is not None:
                displacement = np.linalg.norm(curr_hip - prev_hip)
                speed = displacement * fps
                speed_text = f"Speed: {speed:.2f} px/s"
            else:
                speed = 0.0

            prev_hip = curr_hip
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            writer.writerow([frame_idx, round(frame_idx / fps, 2), cx, cy, round(speed, 2)])

        cv2.putText(frame, speed_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, prev_hip

    prev_hip1 = None
    prev_hip2 = None
    frame_idx = 0
    wrote_frame = False

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        frame1, prev_hip1 = process(frame1, prev_hip1, pose1, writer1, frame_idx)
        frame2, prev_hip2 = process(frame2, prev_hip2, pose2, writer2, frame_idx)

        frame2 = cv2.resize(frame2, (frame_width, frame_height))
        combined = np.hstack((frame1, frame2))
        out.write(combined)
        wrote_frame = True
        frame_idx += 1

    cap1.release()
    cap2.release()
    out.release()
    csv1.close()
    csv2.close()
    cv2.destroyAllWindows()

    if wrote_frame and os.path.exists(video_output_path):
        print(f"[✅] Video saved successfully: {video_output_path}")
        return video_output_path
    else:
        print(f"[❌] No frames were written. Video file was not created.")
        return None


def analyze_single_video(video_path, output_folder):
    """
    Analyze a single climbing video with advanced metrics.
    Returns path to output video and metrics JSON file.
    """
    print(f"[1/7] 📹 Starting video analysis...")
    print(f"     Video: {video_path}")
    print(f"     Output: {output_folder}")
    
    # Setup MediaPipe Pose
    print(f"[2/7] 🤖 Initializing MediaPipe pose detection...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"     Video info: {frame_width}x{frame_height} @ {fps:.2f} fps, {total_frames} frames ({duration:.1f}s)")
    
    # Initialize metrics calculator
    metrics_calc = ClimbingMetricsCalculator(fps=fps)
    
    # Detect takeoff
    print(f"[3/7] 🚀 Detecting climb start (takeoff)...")
    def detect_takeoff(cap, pose_instance):
        hip_y_positions = []
        frame_count = 0
        max_frames = int(fps * 5)
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_instance.process(image_rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                y = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                hip_y_positions.append(y)
            frame_count += 1
        
        if len(hip_y_positions) < 4:
            return 0
        
        diffs = np.diff(hip_y_positions)
        for i in range(3, len(diffs)):
            if all(d < -0.001 for d in diffs[i-3:i]):
                return i
        return 0
    
    takeoff_frame = detect_takeoff(cv2.VideoCapture(video_path), pose)
    print(f"     Takeoff detected at frame {takeoff_frame} ({takeoff_frame/fps:.2f}s)")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, takeoff_frame)
    
    video_output_path = os.path.abspath(os.path.join(output_folder, "analyzed_output.mp4"))
    metrics_json_path = os.path.abspath(os.path.join(output_folder, "metrics.json"))
    
    print(f"[4/7] 🎬 Setting up video writer...")
    result = create_video_writer(video_output_path, fps, frame_width, frame_height)
    if result is None:
        print(f"[ERROR] Failed to open VideoWriter.")
        return None, None
    
    out, video_output_path, codec_used = result
    print(f"     Output video: {os.path.basename(video_output_path)}")
    print(f"     Codec: {codec_used}")
    
    # CSV for detailed metrics
    csv_path = os.path.abspath(os.path.join(output_folder, "metrics_log.csv"))
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame", "Time (s)", "Hip_X", "Hip_Y", "Hip_Speed (px/s)",
        "Hip_to_Wall_Distance", "Stability", "Balance", "Technique", "Rhythm", "Overall_Score"
    ])
    
    # Store metrics for JSON export
    all_metrics = []
    
    # For real-time graph overlay (keep recent history)
    metrics_history = []  # Rolling window for graph
    max_history_seconds = 10  # Show last 10 seconds on graph
    
    prev_hip = None
    frame_idx = 0
    frames_to_process = total_frames - takeoff_frame
    last_progress = -1
    
    print(f"[5/7] 📊 Processing frames and calculating metrics...")
    print(f"     Processing {frames_to_process} frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Progress reporting every 10%
        if frames_to_process > 0:
            progress = int((frame_idx / frames_to_process) * 100)
            if progress != last_progress and progress % 10 == 0:
                print(f"     Progress: {progress}% ({frame_idx}/{frames_to_process} frames)")
                last_progress = progress
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        time_sec = round(frame_idx / fps, 2)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Calculate hip speed using metrics calculator
            hip_speed, curr_hip = metrics_calc.calculate_hip_speed(
                results.pose_landmarks, frame_width, frame_height, prev_hip
            )
            
            # Get hip center for display (convert back to pixel coordinates)
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            cx = int((lh.x + rh.x) / 2 * frame_width)
            cy = int((lh.y + rh.y) / 2 * frame_height)
            
            # Use speed from metrics calculator
            speed = hip_speed
            
            # Calculate all metrics
            hip_to_wall = metrics_calc.calculate_hip_to_wall_distance(
                results.pose_landmarks, frame_width, frame_height
            )
            stability = metrics_calc.calculate_stability(curr_hip, speed)
            balance = metrics_calc.calculate_balance(results.pose_landmarks)
            technique = metrics_calc.calculate_technique(
                results.pose_landmarks, frame_width, frame_height
            )
            rhythm = metrics_calc.calculate_rhythm(speed, frame_idx)
            overall_score = metrics_calc.calculate_overall_score(
                stability, balance, technique, rhythm
            )
            
            # Store metrics
            metrics_data = {
                "frame": frame_idx,
                "time": time_sec,
                "hip_speed": round(hip_speed, 2),
                "hip_to_wall": round(hip_to_wall, 2),
                "stability": round(stability, 2),
                "balance": round(balance, 2),
                "technique": round(technique, 2),
                "rhythm": round(rhythm, 2),
                "overall_score": round(overall_score, 2)
            }
            all_metrics.append(metrics_data)
            
            # Add to history for graph overlay (rolling window)
            metrics_history.append(metrics_data)
            # Keep only last N seconds of data
            cutoff_time = time_sec - max_history_seconds
            metrics_history = [m for m in metrics_history if m['time'] >= cutoff_time]
            
            # Write to CSV
            csv_writer.writerow([
                frame_idx, time_sec, cx, cy, round(hip_speed, 2),
                round(hip_to_wall, 2), round(stability, 2),
                round(balance, 2), round(technique, 2),
                round(rhythm, 2), round(overall_score, 2)
            ])
            
            # Draw pose
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw metrics on frame (left side)
            y_offset = 30
            cv2.putText(frame, f"Stability: {stability:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(frame, f"Balance: {balance:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(frame, f"Technique: {technique:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(frame, f"Rhythm: {rhythm:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(frame, f"Overall: {overall_score:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Hip Speed: {hip_speed:.1f} px/s", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            cv2.putText(frame, f"Hip-to-Wall: {hip_to_wall:.1f}px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Create and overlay real-time radar chart (top-right corner)
            if len(metrics_history) >= 1:
                try:
                    graph_overlay = create_graph_overlay(metrics_history, frame_width, frame_height)
                    frame = overlay_graph_on_frame(frame, graph_overlay, position='top_right', padding=20)
                except Exception as e:
                    print(f"[WARNING] Failed to create graph overlay: {e}")
            
            prev_hip = curr_hip
        else:
            # No pose detected
            all_metrics.append({
                "frame": frame_idx,
                "time": time_sec,
                "hip_speed": 0,
                "hip_to_wall": 0,
                "stability": 0,
                "balance": 0,
                "technique": 0,
                "rhythm": 0,
                "overall_score": 0
            })
            prev_hip = None
        
        # Overlay radar chart even if no pose detected (use previous history)
        if len(metrics_history) >= 1:
            try:
                graph_overlay = create_graph_overlay(metrics_history, frame_width, frame_height)
                frame = overlay_graph_on_frame(frame, graph_overlay, position='top_right', padding=20)
            except Exception as e:
                print(f"[WARNING] Failed to create graph overlay: {e}")
        
        out.write(frame)
        frame_idx += 1
    
    print(f"     ✓ Processed {frame_idx} frames")
    
    # Close video writer and ensure file is flushed
    cap.release()
    out.release()
    csv_file.close()
    
    # Ensure video file is fully written to disk before returning
    print(f"[6/7] 📈 Finalizing video file...")
    if os.path.exists(video_output_path):
        # Wait for file system to sync and verify file is stable
        max_wait = 3  # Maximum seconds to wait
        wait_interval = 0.1
        waited = 0
        stable_count = 0
        last_size = 0
        
        while waited < max_wait:
            current_size = os.path.getsize(video_output_path)
            if current_size == last_size and current_size > 0:
                stable_count += 1
                if stable_count >= 3:  # File size stable for 0.3 seconds
                    break
            else:
                stable_count = 0
            last_size = current_size
            time.sleep(wait_interval)
            waited += wait_interval
        
        final_size = os.path.getsize(video_output_path)
        if final_size > 0:
            print(f"     ✓ Video file finalized ({final_size / (1024*1024):.2f} MB)")
        else:
            print(f"     ⚠️ Warning: Video file exists but size is 0")
    else:
        print(f"     ❌ Error: Video file not found after writing")
    
    print(f"[6/7] 📈 Calculating average metrics...")
    # Calculate average scores
    if all_metrics:
        def safe_mean(values):
            filtered = [v for v in values if v > 0]
            return round(np.mean(filtered), 2) if filtered else 0.0
        
        avg_metrics = {
            "avg_stability": safe_mean([m["stability"] for m in all_metrics]),
            "avg_balance": safe_mean([m["balance"] for m in all_metrics]),
            "avg_technique": safe_mean([m["technique"] for m in all_metrics]),
            "avg_rhythm": safe_mean([m["rhythm"] for m in all_metrics]),
            "avg_overall_score": safe_mean([m["overall_score"] for m in all_metrics]),
            "avg_hip_to_wall": safe_mean([m["hip_to_wall"] for m in all_metrics]),
            "avg_hip_speed": safe_mean([m["hip_speed"] for m in all_metrics])
        }
    else:
        avg_metrics = {
            "avg_stability": 0.0,
            "avg_balance": 0.0,
            "avg_technique": 0.0,
            "avg_rhythm": 0.0,
            "avg_overall_score": 0.0,
            "avg_hip_to_wall": 0.0,
            "avg_hip_speed": 0.0
        }
    
    # Save metrics to JSON
    metrics_output = {
        "video_info": {
            "fps": fps,
            "total_frames": frame_idx,
            "duration_seconds": round(frame_idx / fps, 2)
        },
        "average_scores": avg_metrics,
        "frame_by_frame": all_metrics
    }
    
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    cv2.destroyAllWindows()
    
    print(f"[7/7] 💾 Saving results...")
    if os.path.exists(video_output_path):
        file_size_mb = os.path.getsize(video_output_path) / (1024 * 1024)
        
        # Always optimize video for web playback - ensure H.264 encoding
        if video_output_path.endswith(('.mp4', '.avi')):
            print(f"     Optimizing video for web playback (H.264)...")
            optimized_path = optimize_video_for_web(video_output_path, codec_used)
            if optimized_path and optimized_path != video_output_path:
                # Replace original with optimized version
                try:
                    os.replace(optimized_path, video_output_path)
                    new_size = os.path.getsize(video_output_path) / (1024 * 1024)
                    print(f"     ✓ Video optimized to H.264 ({file_size_mb:.2f} MB → {new_size:.2f} MB)")
                except Exception as e:
                    print(f"     ⚠️ Could not replace with optimized version: {e}")
                    # Keep original
            elif optimized_path:
                print(f"     ✓ Video already H.264 optimized")
            else:
                if codec_used in ['H264', 'avc1', 'X264']:
                    print(f"     ✓ Video already encoded as H.264 ({codec_used})")
                else:
                    print(f"     ⚠️ Video optimization skipped (ffmpeg not available). Video may not play optimally in browsers.")
        
        print(f"[✅] Analysis complete!")
        print(f"     Video: {os.path.basename(video_output_path)} ({os.path.getsize(video_output_path) / (1024 * 1024):.2f} MB)")
        print(f"     Metrics: {os.path.basename(metrics_json_path)}")
        print(f"     Average scores - Stability: {avg_metrics['avg_stability']:.1f}, "
              f"Balance: {avg_metrics['avg_balance']:.1f}, "
              f"Technique: {avg_metrics['avg_technique']:.1f}, "
              f"Rhythm: {avg_metrics['avg_rhythm']:.1f}, "
              f"Overall: {avg_metrics['avg_overall_score']:.1f}")
        return video_output_path, metrics_json_path
    else:
        print(f"[❌] Analysis failed - video file not found.")
        return None, None

def optimize_video_for_web(video_path, original_codec=None):  # pylint: disable=unused-argument
    """
    Optimize video for web playback by ensuring H.264 encoding and faststart.
    Uses ffmpeg to:
    1. Convert to H.264 codec (if not already)
    2. Move moov atom to beginning (faststart) for streaming
    3. Optimize for web browsers
    
    Returns optimized path or original path if optimization fails.
    """
    try:
        # Check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              timeout=2)
        if result.returncode != 0:
            return video_path  # ffmpeg not available
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return video_path  # ffmpeg not found or timeout
    
    try:
        # Create temporary output path
        temp_path = video_path + '.optimized.mp4'
        
        # Build ffmpeg command optimized for mobile/web streaming
        # Aggressive compression for smooth mobile playback on longer videos
        cmd = [
            'ffmpeg', 
            '-i', video_path,
            '-c:v', 'libx264',           # H.264 codec (best for web/mobile)
            '-preset', 'fast',           # Faster encoding (still good quality)
            '-crf', '30',                # Higher CRF = lower bitrate (30 is good balance)
            '-maxrate', '1.5M',          # Maximum bitrate (1.5 Mbps for better mobile streaming)
            '-bufsize', '3M',            # Buffer size (2x maxrate for smooth streaming)
            '-vf', 'scale=-2:720',      # Scale to max 720p height (maintains aspect ratio, reduces file size)
            '-profile:v', 'baseline',   # Baseline profile (best mobile compatibility)
            '-level', '3.1',            # Level 3.1 (widely supported on mobile)
            '-pix_fmt', 'yuv420p',      # Pixel format (required for compatibility)
            '-movflags', '+faststart',  # Move metadata to beginning (critical for streaming)
            '-tune', 'fastdecode',      # Optimize for fast decoding (mobile)
            '-g', '30',                 # GOP size (keyframe every 30 frames for seeking)
            '-keyint_min', '30',        # Minimum keyframe interval
            '-sc_threshold', '0',       # Disable scene change detection for consistent GOP
            '-an',                      # Remove audio to reduce file size (climbing videos don't need audio)
            '-y',                       # Overwrite output
            temp_path
        ]
        
        # Only add audio encoding if source has audio
        # For now, we'll skip audio re-encoding to avoid issues
        # cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        
        print(f"     Converting to H.264 with mobile/web optimization (2 Mbps max bitrate)...")
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            if file_size > 0:
                print(f"     ✓ H.264 conversion successful")
                return temp_path
            else:
                print(f"     ⚠️ Optimized file is empty")
                return video_path
        else:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            print(f"     ⚠️ ffmpeg optimization failed: {error_msg[:200]}")
            return video_path
    except subprocess.TimeoutExpired:
        print(f"     ⚠️ Video optimization timed out")
        return video_path
    except Exception as e:
        print(f"     ⚠️ Video optimization error: {e}")
        return video_path
