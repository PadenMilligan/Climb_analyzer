import numpy as np
from collections import deque
import mediapipe as mp

class ClimbingMetricsCalculator:
    """Calculates advanced climbing metrics from pose landmarks"""
    
    def __init__(self, fps=30, window_size=10):
        self.fps = fps
        self.window_size = window_size
        self.hip_positions = deque(maxlen=window_size)
        self.hip_speeds = deque(maxlen=window_size)
        self.body_angles = deque(maxlen=window_size)
        self.limb_positions = deque(maxlen=window_size)
        
    def calculate_hip_to_wall_distance(self, landmarks, frame_width, frame_height):
        """
        Estimate hip distance to wall using body angle and pose geometry.
        Uses the relative position of shoulders and hips to estimate depth.
        """
        try:
            mp_pose = mp.solutions.pose
            left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Calculate hip center
            hip_x = (left_hip.x + right_hip.x) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            hip_z = (left_hip.z + right_hip.z) / 2 if hasattr(left_hip, 'z') else 0
            
            # Calculate shoulder center
            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_z = (left_shoulder.z + right_shoulder.z) / 2 if hasattr(left_shoulder, 'z') else 0
            
            # Estimate distance using Z coordinate (MediaPipe provides this)
            # If Z is not available, estimate from body angle
            if hasattr(left_hip, 'z') and left_hip.z != 0:
                # Use MediaPipe's Z coordinate (negative = closer to camera)
                distance = abs(left_hip.z) * frame_width  # Scale to pixels
            else:
                # Estimate from body angle (torso lean)
                torso_angle = np.arctan2(shoulder_y - hip_y, abs(shoulder_x - hip_x))
                # More vertical = closer to wall, more horizontal = further
                distance = abs(np.sin(torso_angle)) * frame_width * 0.3
            
            return max(0, distance)
        except:
            return 0.0
    
    def calculate_stability(self, current_hip_pos, current_speed):
        """
        Calculate stability based on variance in hip position and smoothness of movement.
        Lower variance and smoother movement = higher stability.
        """
        if len(self.hip_positions) < 3:
            return 50.0  # Default neutral score
        
        self.hip_positions.append(current_hip_pos)
        self.hip_speeds.append(current_speed)
        
        # Calculate position variance
        positions_array = np.array(self.hip_positions)
        position_variance = np.var(positions_array, axis=0).sum()
        
        # Calculate speed variance (jerkiness)
        speeds_array = np.array(self.hip_speeds)
        speed_variance = np.var(speeds_array) if len(speeds_array) > 1 else 0
        
        # Normalize and convert to stability score (0-100)
        # Lower variance = higher stability
        position_stability = max(0, 100 - (position_variance * 100))
        speed_stability = max(0, 100 - (speed_variance * 0.1))
        
        stability = (position_stability * 0.6 + speed_stability * 0.4)
        return min(100, max(0, stability))
    
    def calculate_balance(self, landmarks):
        """
        Calculate balance based on weight distribution between left and right limbs.
        More even distribution = better balance.
        """
        try:
            mp_pose = mp.solutions.pose
            left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Calculate center of mass distribution
            # Weight on left side
            left_weight = abs(left_hip.y - left_ankle.y) + abs(left_hip.y - left_wrist.y)
            # Weight on right side
            right_weight = abs(right_hip.y - right_ankle.y) + abs(right_hip.y - right_wrist.y)
            
            total_weight = left_weight + right_weight
            if total_weight == 0:
                return 50.0
            
            # Balance ratio (0.5 = perfect balance)
            balance_ratio = min(left_weight, right_weight) / total_weight
            
            # Convert to score (0-100), 0.5 ratio = 100 score
            balance_score = balance_ratio * 200
            return min(100, max(0, balance_score))
        except:
            return 50.0
    
    def calculate_technique(self, landmarks, frame_width, frame_height):
        """
        Calculate technique score based on body angle, limb positioning, and movement efficiency.
        """
        try:
            mp_pose = mp.solutions.pose
            left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Calculate centers
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Body angle (torso angle relative to vertical)
            # In MediaPipe: y increases downward, so for vertical body: shoulder_y < hip_y
            # Calculate angle from vertical (0 = perfectly vertical, π/2 = horizontal)
            dx = abs(shoulder_center_x - hip_center_x)
            dy = abs(shoulder_center_y - hip_center_y)
            
            # Avoid division by zero
            if dx < 0.01:
                dx = 0.01
            
            # Angle from vertical (in radians)
            # For vertical body: dy/dx should be large, angle close to π/2
            # We want to measure deviation from vertical (0 radians = vertical)
            body_angle_rad = np.arctan2(dx, dy)  # Angle from vertical
            
            # Convert to degrees for easier thresholding
            body_angle_deg = np.degrees(body_angle_rad)
            
            # Body angle score: More vertical (smaller angle) = better
            # 0-15 degrees = excellent (40 points), 15-30 = good (30 points), 30-45 = fair (20 points), >45 = poor (0-10 points)
            if body_angle_deg <= 15:
                body_angle_score = 40.0
            elif body_angle_deg <= 30:
                body_angle_score = 40.0 - ((body_angle_deg - 15) / 15) * 10  # 40 to 30
            elif body_angle_deg <= 45:
                body_angle_score = 30.0 - ((body_angle_deg - 30) / 15) * 10  # 30 to 20
            else:
                body_angle_score = max(0, 20.0 - ((body_angle_deg - 45) / 15) * 10)  # 20 to 0
            
            # Knee bend (good technique uses bent knees for power and flexibility)
            # Calculate leg extension: distance from hip to knee relative to hip to ankle
            left_leg_length = np.sqrt((left_ankle.x - left_hip.x)**2 + (left_ankle.y - left_hip.y)**2)
            right_leg_length = np.sqrt((right_ankle.x - right_hip.x)**2 + (right_ankle.y - right_hip.y)**2)
            avg_leg_length = (left_leg_length + right_leg_length) / 2
            
            left_knee_to_hip = np.sqrt((left_knee.x - left_hip.x)**2 + (left_knee.y - left_hip.y)**2)
            right_knee_to_hip = np.sqrt((right_knee.x - right_hip.x)**2 + (right_knee.y - right_hip.y)**2)
            avg_knee_to_hip = (left_knee_to_hip + right_knee_to_hip) / 2
            
            # Knee bend ratio: higher ratio = more bent (better for technique)
            # Normalize by leg length to get bend ratio
            if avg_leg_length > 0.01:
                knee_bend_ratio = avg_knee_to_hip / avg_leg_length
            else:
                knee_bend_ratio = 0.5  # Default moderate bend
            
            # Optimal knee bend is around 0.4-0.6 (40-60% of leg length from hip to knee)
            # Score based on how close to optimal range
            if 0.35 <= knee_bend_ratio <= 0.65:
                knee_bend_score = 35.0  # Excellent bend
            elif 0.25 <= knee_bend_ratio < 0.35 or 0.65 < knee_bend_ratio <= 0.75:
                knee_bend_score = 30.0 - abs(knee_bend_ratio - 0.5) * 50  # Good bend
            elif 0.15 <= knee_bend_ratio < 0.25 or 0.75 < knee_bend_ratio <= 0.85:
                knee_bend_score = 20.0 - abs(knee_bend_ratio - 0.5) * 30  # Fair bend
            else:
                knee_bend_score = max(0, 10.0 - abs(knee_bend_ratio - 0.5) * 20)  # Poor bend
            
            # Hip-to-wall distance (closer to wall = better technique)
            # Use the hip-to-wall distance calculation
            hip_to_wall = self.calculate_hip_to_wall_distance(landmarks, frame_width, frame_height)
            
            # Normalize by frame size (typical good distance is < 10% of frame width)
            normalized_distance = hip_to_wall / frame_width if frame_width > 0 else 0
            
            # Score: closer to wall (smaller distance) = better
            # 0-5% of frame width = excellent (25 points), 5-10% = good (20 points), 10-20% = fair (10 points), >20% = poor (0-5 points)
            if normalized_distance <= 0.05:
                hip_distance_score = 25.0
            elif normalized_distance <= 0.10:
                hip_distance_score = 25.0 - ((normalized_distance - 0.05) / 0.05) * 5  # 25 to 20
            elif normalized_distance <= 0.20:
                hip_distance_score = 20.0 - ((normalized_distance - 0.10) / 0.10) * 10  # 20 to 10
            else:
                hip_distance_score = max(0, 10.0 - ((normalized_distance - 0.20) / 0.20) * 10)  # 10 to 0
            
            technique_score = body_angle_score + knee_bend_score + hip_distance_score
            return min(100, max(0, technique_score))
        except Exception as e:
            # Return a default score instead of 50 to avoid penalizing on errors
            return 60.0
    
    def calculate_rhythm(self, current_speed, frame_idx):
        """
        Calculate rhythm based on movement patterns and timing.
        Consistent, rhythmic movement = better rhythm.
        """
        if len(self.hip_speeds) < 5:
            return 50.0
        
        self.hip_speeds.append(current_speed)
        speeds_array = np.array(list(self.hip_speeds))
        
        # Detect movement peaks (when climber makes a move)
        if len(speeds_array) > 5:
            # Find peaks in speed (movement events)
            peaks = []
            for i in range(1, len(speeds_array) - 1):
                if speeds_array[i] > speeds_array[i-1] and speeds_array[i] > speeds_array[i+1]:
                    if speeds_array[i] > np.mean(speeds_array) + np.std(speeds_array):
                        peaks.append(i)
            
            if len(peaks) >= 2:
                # Calculate time between peaks (rhythm consistency)
                peak_intervals = np.diff(peaks) / self.fps  # Convert to seconds
                if len(peak_intervals) > 0:
                    interval_variance = np.var(peak_intervals)
                    # Lower variance = more consistent rhythm
                    rhythm_score = max(0, 100 - (interval_variance * 100))
                    return min(100, max(0, rhythm_score))
        
        # Fallback: use speed consistency
        speed_consistency = 100 - (np.std(speeds_array) / (np.mean(speeds_array) + 1) * 50)
        return min(100, max(0, speed_consistency))
    
    def calculate_overall_score(self, stability, balance, technique, rhythm):
        """
        Calculate overall climbing performance score.
        """
        # Weighted average
        overall = (
            stability * 0.25 +
            balance * 0.25 +
            technique * 0.30 +
            rhythm * 0.20
        )
        return min(100, max(0, overall))
    
    def calculate_hip_speed(self, landmarks, frame_width, frame_height, prev_hip_pos=None):
        """
        Calculate the speed of hip movement.
        Returns: (hip_speed_px_per_sec, current_hip_center_position)
        """
        try:
            mp_pose = mp.solutions.pose
            left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate hip center
            hip_x = (left_hip.x + right_hip.x) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            
            # Convert to pixel coordinates
            hip_center = np.array([
                hip_x * frame_width,
                hip_y * frame_height
            ])
            
            # Calculate speed if we have previous position
            if prev_hip_pos is not None:
                displacement = np.linalg.norm(hip_center - prev_hip_pos)
                speed = displacement * self.fps
            else:
                speed = 0.0
            
            return speed, hip_center
        except:
            return 0.0, None

