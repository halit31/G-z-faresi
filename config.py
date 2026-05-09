import cv2

# Mouse control settings
SMOOTHING_ALPHA = 0.12
CALIB_MARGIN = 0.25  # Use center 50% of camera frame (0.25 margin on each side)

# Blink detection settings
BLINK_EAR_THRESHOLD = 0.21
BLINK_CONSEC_FRAMES = 3
CLICK_COOLDOWN = 0.5  # Seconds

# Camera settings
CAMERA_INDEX = 0

# UI Settings
IRIS_DOT_COLOR = (0, 255, 0)  # Green
CALIB_RECT_COLOR = (255, 0, 0)  # Blue
TEXT_COLOR = (0, 255, 255)  # Cyan
WARNING_COLOR = (0, 0, 255)  # Red
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
