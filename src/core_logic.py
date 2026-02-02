import numpy as np
import cv2

def calculate_novelties(landmarks):
    features = {'nose_angle': 0, 'mouth_angle': 0, 'yaw_ratio': 1.0, 'asym_index': 0.0}
    
    try:
        # Get Key Coordinates
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)
        nose_tip = np.array(landmarks['nose_tip'][2])
        mouth_left = np.array(landmarks['top_lip'][0])
        mouth_right = np.array(landmarks['top_lip'][6])

        # Nose Angle 
        a = np.linalg.norm(nose_tip - left_eye)
        b = np.linalg.norm(nose_tip - right_eye)
        c = np.linalg.norm(left_eye - right_eye)
        if a * b == 0: cos_angle = 0
        else: cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        
        features['nose_angle'] = np.degrees(np.arccos(max(-1.0, min(1.0, cos_angle))))

        # Mouth Angle
        d = np.linalg.norm(nose_tip - mouth_left)
        e = np.linalg.norm(nose_tip - mouth_right)
        f = np.linalg.norm(mouth_left - mouth_right)
        
        if d * e == 0: cos_mouth = 0
        else: cos_mouth = (d**2 + e**2 - f**2) / (2 * d * e)
        
        features['mouth_angle'] = np.degrees(np.arccos(max(-1.0, min(1.0, cos_mouth))))

        # Yaw Ratio
        features['yaw_ratio'] = a / (b + 1e-6)

        # Asymmetry Index
        features['asym_index'] = abs(a - b) / (c + 1e-6)

    except Exception as e:
        pass

    return features

def get_ear(eye_landmarks):
    """
    Calculates Eye Aspect Ratio (EAR) for the Blink Test.
    Input: List of 6 (x,y) points for one eye.
    """
    # Vertical distances
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)