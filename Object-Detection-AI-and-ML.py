"""
HSV Color-Based Object Detector
=========================================
This program uses Python, OpenCV, and NumPy to perform real-time color thresholding
and object tracking using the HSV color space. 

Key Concepts Implemented:
1. BGR to HSV Color Space Conversion.
2. GUI Control Panel (Trackbars) for dynamic color masking.
3. Morphological Filtering (Opening & Closing) to eliminate noise.
4. Contour analysis to identify object centers (centroids) and draw bounding boxes.
5. Camera fallback: generates a synthetic moving target if no camera is available.

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np

# Global variables for trackbar window names
TRACKBAR_WINDOW = "Control Panel (HSV Tuner)"
VIDEO_WINDOW = "Object Detection & Tracking"

def empty_callback(x):
    """Placeholder callback function required by OpenCV trackbars."""
    pass

def setup_trackbars():
    """
    Creates an interactive control window with trackbars to fine-tune
    HSV boundaries. The default startup values are pre-set to target a
    bright blue object.
    """
    cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRACKBAR_WINDOW, 400, 350)
    
    # Hue values range from 0 to 179 in OpenCV
    cv2.createTrackbar("Lower Hue", TRACKBAR_WINDOW, 90, 179, empty_callback)
    cv2.createTrackbar("Upper Hue", TRACKBAR_WINDOW, 130, 179, empty_callback)
    
    # Saturation and Value range from 0 to 255
    cv2.createTrackbar("Lower Sat", TRACKBAR_WINDOW, 100, 255, empty_callback)
    cv2.createTrackbar("Upper Sat", TRACKBAR_WINDOW, 255, 255, empty_callback)
    cv2.createTrackbar("Lower Val", TRACKBAR_WINDOW, 80, 255, empty_callback)
    cv2.createTrackbar("Upper Val", TRACKBAR_WINDOW, 255, 255, empty_callback)
    
    # Noise reduction control slider (Kernel size)
    cv2.createTrackbar("Noise Filter", TRACKBAR_WINDOW, 5, 15, empty_callback)

def get_hsv_limits():
    """Reads current values from the GUI trackbars and returns lower/upper bounds."""
    l_h = cv2.getTrackbarPos("Lower Hue", TRACKBAR_WINDOW)
    u_h = cv2.getTrackbarPos("Upper Hue", TRACKBAR_WINDOW)
    l_s = cv2.getTrackbarPos("Lower Sat", TRACKBAR_WINDOW)
    u_s = cv2.getTrackbarPos("Upper Sat", TRACKBAR_WINDOW)
    l_v = cv2.getTrackbarPos("Lower Val", TRACKBAR_WINDOW)
    u_v = cv2.getTrackbarPos("Upper Val", TRACKBAR_WINDOW)
    
    # Format boundaries as NumPy arrays for opencv processing
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    return lower_bound, upper_bound

def apply_morphological_ops(mask, kernel_size):
    """
    Applies mathematical morphology to eliminate salt-and-pepper noise.
    - Opening (Erosion followed by Dilation) removes tiny white pixel noise blocks.
    - Closing (Dilation followed by Erosion) bridges internal structural gaps.
    """
    if kernel_size < 1:
        kernel_size = 1
    # Kernel size must be odd for morphological structuring elements
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform opening to clear background noise
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Perform closing to fill small holes inside target shapes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

def generate_fallback_frame(frame_count):
    """
    Generates a synthetic frame in case no physical camera is detected.
    Produces a moving, colored circle against a dark background to demonstrate
    real-time color tracking capabilities.
    """
    height, width = 480, 640
    # Create black canvas
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate oscillating trajectory coordinates for visual tracking
    x = int(width / 2 + 150 * np.sin(frame_count * 0.05))
    y = int(height / 2 + 100 * np.cos(frame_count * 0.03))
    
    # Draw a blue sphere (Pre-matched by default startup trackbar values)
    cv2.circle(frame, (x, y), 40, (235, 120, 10), -1)
    
    # Inject slight background noise
    noise = np.random.randint(0, 15, size=(height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    # Add a title text indicating demo mode
    cv2.putText(frame, "Demo Mode: No Web Cam Detected", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def main():
    print("[INFO] Initializing HSV Tracker...")
    print("[INFO] Press 'q' or 'ESC' to exit the application.")
    
    # Initialize trackbars
    setup_trackbars()
    
    # Attempt to load camera stream (0 is usually the integrated webcam)
    cap = cv2.VideoCapture(0)
    is_camera_active = cap.isOpened()
    
    if not is_camera_active:
        print("[WARNING] No active camera detected. Falling back to synthetic simulation mode.")
        
    frame_counter = 0
    
    while True:
        if is_camera_active:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera failed to yield frames. Switching to fallback.")
                is_camera_active = False
                continue
            # Mirror the frame horizontally to make tracking intuitive (like a mirror)
            frame = cv2.flip(frame, 1)
        else:
            # Generate fake moving scene for self-contained execution
            frame = generate_fallback_frame(frame_counter)
            frame_counter += 1
            
        # 1. Convert BGR image coordinates to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Get active slider bounds
        lower_hsv, upper_hsv = get_hsv_limits()
        
        # 3. Apply color mask (Threshold the image based on bounds)
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
        
        # 4. Clean mask from minor disturbances
        noise_size = cv2.getTrackbarPos("Noise Filter", TRACKBAR_WINDOW)
        mask = apply_morphological_ops(mask, noise_size)
        
        # 5. Extract contours from binary mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize tracking metrics
        object_detected = False
        
        # 6. Filter and evaluate active contours
        if len(contours) > 0:
            # Pick the largest contour by area to ignore secondary small artifacts
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Minimum area threshold (adjust to avoid tracking tiny background elements)
            if area > 400:
                object_detected = True
                
                # Retrieve bounding rectangle parameters
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Draw the bounding rectangle around tracked objects
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Determine spatial centroid (center of mass) of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    
                    # Highlight center with a dot and coordinate label
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"({center_x}, {center_y})", (center_x + 10, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Label bounding box with area magnitude
                cv2.putText(frame, f"Target Area: {int(area)} px", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Visual status message overlay
        status_text = "Target Locked" if object_detected else "Searching Target..."
        status_color = (0, 255, 0) if object_detected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_text}", (15, 60 if not is_camera_active else 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 7. Render dynamic windows for monitoring
        # We can construct a combined display stack using mask (binary) and source
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display_panel = np.hstack((frame, rgb_mask))
        
        cv2.imshow(VIDEO_WINDOW, display_panel)
        
        # Intercept frame termination signals
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ASCII for the ESC key
            break
            
    # Clean up systems safely
    if is_camera_active:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application cleanly shut down.")

if __name__ == "__main__":
    main()
