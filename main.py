import cv2
import sys
import time
from eye_tracker import EyeTracker
from mouse_controller import MouseController
import config

def main():
    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {config.CAMERA_INDEX}")
        sys.exit(1)

    # Initialize components
    tracker = EyeTracker()
    controller = MouseController()
    
    prev_time = time.time()
    
    print("Application started. Press 'Q' to quit.")
    print("Move your mouse to the top-left corner to trigger FAILSAFE and exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process eye tracking
            iris_pos, is_blinking, ear, face_detected = tracker.process_frame(frame)

            # UI: Draw calibration rectangle
            margin = config.CALIB_MARGIN
            cv2.rectangle(
                frame,
                (int(w * margin), int(h * margin)),
                (int(w * (1 - margin)), int(h * (1 - margin))),
                config.CALIB_RECT_COLOR,
                2
            )

            if face_detected and iris_pos:
                # 1. Move mouse
                controller.move(iris_pos[0], iris_pos[1])
                
                # 2. Handle click
                controller.handle_blink(is_blinking)

                # 3. Draw iris dot
                ix, iy = int(iris_pos[0] * w), int(iris_pos[1] * h)
                cv2.circle(frame, (ix, iy), 5, config.IRIS_DOT_COLOR, -1)
            else:
                # No face warning
                cv2.putText(
                    frame, "NO FACE DETECTED", (50, 100),
                    config.FONT, config.FONT_SCALE * 1.5, config.WARNING_COLOR, config.FONT_THICKNESS
                )

            # UI: Display EAR and FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(
                frame, f"EAR: {ear:.2f}", (10, 30),
                config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS
            )
            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 60),
                config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS
            )

            # Show preview
            cv2.imshow("Eye Mouse Controller", frame)

            # Exit logic
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Cleanup
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
