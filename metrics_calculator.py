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
            
            # Body angle (torso angle relative to vertical)
            hip_center_y = (left_hip.y + right_hip.y) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            body_angle = abs(np.arctan2(shoulder_center_y - hip_center_y, 
                                       abs(left_shoulder.x - left_hip.x)))
            
            # Knee bend (good technique uses bent knees for power)
            left_knee_bend = abs(left_knee.y - left_hip.y)
            right_knee_bend = abs(right_knee.y - right_hip.y)
            avg_knee_bend = (left_knee_bend + right_knee_bend) / 2
            
            # Technique scoring:
            # - Body angle: More vertical (closer to wall) = better (0-40 points)
            body_angle_score = max(0, 40 - (body_angle * 100))
            
            # - Knee bend: Moderate bend is good (0-30 points)
            knee_bend_score = min(30, avg_knee_bend * 100) if avg_knee_bend < 0.3 else max(0, 30 - (avg_knee_bend - 0.3) * 50)
            
            # - Hip position: Lower hips = better technique (0-30 points)
            hip_height_score = max(0, 30 - (hip_center_y - 0.3) * 100) if hip_center_y > 0.3 else 30
            
            technique_score = body_angle_score + knee_bend_score + hip_height_score
            return min(100, max(0, technique_score))
        except:
            return 50.0
    
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

