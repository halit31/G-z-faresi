import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
import config

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Left eye landmarks for EAR calculation
        # [362, 385, 387, 263, 373, 380]
        # p1: 362 (inner corner)
        # p2: 385 (top-inner)
        # p3: 387 (top-outer)
        # p4: 263 (outer corner)
        # p5: 373 (bottom-outer)
        # p6: 380 (bottom-inner)
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.IRIS_CENTER_ID = 473

    def calculate_ear(self, landmarks: List) -> float:
        """
        Calculates Eye Aspect Ratio (EAR).
        Formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        def get_pt(idx: int) -> np.ndarray:
            lm = landmarks[self.LEFT_EYE_LANDMARKS[idx]]
            return np.array([lm.x, lm.y])

        p1 = get_pt(0)
        p2 = get_pt(1)
        p3 = get_pt(2)
        p4 = get_pt(3)
        p5 = get_pt(4)
        p6 = get_pt(5)

        dist_v1 = np.linalg.norm(p2 - p6)
        dist_v2 = np.linalg.norm(p3 - p5)
        dist_h = np.linalg.norm(p1 - p4)

        ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
        return ear

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], bool, float, bool]:
        """
        Processes a frame and returns tracking data.
        Returns: (iris_pos_normalized, is_blinking, ear_value, face_detected)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, False, 0.0, False

        landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. Get iris position
        iris_landmark = landmarks[self.IRIS_CENTER_ID]
        iris_pos = (iris_landmark.x, iris_landmark.y)
        
        # 2. Calculate EAR
        ear = self.calculate_ear(landmarks)
        
        # 3. Detect blink
        is_blinking = ear < config.BLINK_EAR_THRESHOLD
        
        return iris_pos, is_blinking, ear, True

    def close(self):
        self.face_mesh.close()
