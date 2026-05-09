import pyautogui
import time
import numpy as np
from typing import Tuple
import config

# Disable pyautogui's built-in delay to make it faster
pyautogui.PAUSE = 0

class MouseController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_pos = np.array([self.screen_width / 2, self.screen_height / 2])
        self.last_click_time = 0
        self.blink_frames = 0
        
    def map_coordinates(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """
        Maps normalized iris coordinates (0-1) to screen resolution, 
        considering the calibration margin.
        """
        # Apply margin: map center 50% to full screen
        # iris_x is 0.0 to 1.0. 
        # margin is 0.25. 
        # range is [0.25, 0.75]. 
        # normalized_x = (iris_x - margin) / (1 - 2*margin)
        margin = config.CALIB_MARGIN
        range_size = 1.0 - 2.0 * margin
        
        norm_x = (iris_x - margin) / range_size
        norm_y = (iris_y - margin) / range_size
        
        # Clamp values to [0, 1]
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)
        
        target_x = norm_x * self.screen_width
        target_y = norm_y * self.screen_height
        
        return target_x, target_y

    def move(self, iris_x: float, iris_y: float):
        """
        Moves the mouse with EMA smoothing.
        pos = pos + alpha * (target - pos)
        """
        target_x, target_y = self.map_coordinates(iris_x, iris_y)
        target_pos = np.array([target_x, target_y])
        
        # EMA Smoothing
        self.current_pos = self.current_pos + config.SMOOTHING_ALPHA * (target_pos - self.current_pos)
        
        # Perform move
        pyautogui.moveTo(int(self.current_pos[0]), int(self.current_pos[1]))

    def handle_blink(self, is_blinking: bool):
        """
        Handles blink detection and triggers click with cooldown.
        """
        if is_blinking:
            self.blink_frames += 1
        else:
            if self.blink_frames >= config.BLINK_CONSEC_FRAMES:
                self.click()
            self.blink_frames = 0

    def click(self):
        """
        Triggers left click if cooldown has passed.
        """
        current_time = time.time()
        if current_time - self.last_click_time > config.CLICK_COOLDOWN:
            pyautogui.click()
            self.last_click_time = current_time
