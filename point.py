import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize the MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()
# Set a frame reduction factor to create a movement border
frame_reduction = 100

# Add screen padding to avoid triggering fail-safe
screen_padding = 10
safe_screen_width = screen_width - (2 * screen_padding)
safe_screen_height = screen_height - (2 * screen_padding)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Get webcam dimensions
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

# Variables for smoothing
smoothening = 7
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Variables for click detection
click_delay = 0.3
last_click_time = time.time()
pinch_threshold = 0.04
is_pinched = False

print("Starting hand gesture mouse control. Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image_rgb)
    
    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the position of the index finger tip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert the normalized coordinates to screen coordinates
            index_x = int(np.interp(index_tip.x, [0, 1], [0, wCam]))
            index_y = int(np.interp(index_tip.y, [0, 1], [0, hCam]))
            
            # Smooth the values for better control
            curr_x = prev_x + (index_x - prev_x) / smoothening
            curr_y = prev_y + (index_y - prev_y) / smoothening
            
            # Convert to screen coordinates with frame reduction
            screen_x = np.interp(curr_x, [frame_reduction, wCam-frame_reduction], [screen_padding, safe_screen_width])
            screen_y = np.interp(curr_y, [frame_reduction, hCam-frame_reduction], [screen_padding, safe_screen_height])
            
            # Ensure coordinates stay within safe bounds
            screen_x = max(screen_padding, min(screen_x, safe_screen_width))
            screen_y = max(screen_padding, min(screen_y, safe_screen_height))
            
            # Move the mouse
            pyautogui.moveTo(screen_x, screen_y)
            
            # Update previous positions
            prev_x, prev_y = curr_x, curr_y
            
            # Check for pinch gesture (index finger and thumb)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                              (thumb_tip.y - index_tip.y)**2)
            
            # Draw a line between index finger and thumb
            cv2.line(image, 
                    (int(thumb_tip.x * wCam), int(thumb_tip.y * hCam)),
                    (int(index_tip.x * wCam), int(index_tip.y * hCam)),
                    (0, 255, 0), 2)
            
            # If distance between thumb and index finger is small enough, consider it a click
            current_time = time.time()
            if distance < pinch_threshold and not is_pinched and current_time - last_click_time > click_delay:
                print("Click!")
                pyautogui.click()
                last_click_time = current_time
                is_pinched = True
            elif distance >= pinch_threshold:
                is_pinched = False
    
    # Display the frame rate on the image
    cv2.putText(image, f"Hand Gesture Mouse Control", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow('Hand Gesture Mouse Control', image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Gesture control terminated.")