import cv2
import mediapipe as mp
import time

# Initalize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Open the default camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera inaccessible.")
    exit()

# Variables for FPS Calc
prev_time = 0

# Setting up the MediaPipe Hands object
with mp_hands.Hands(
        model_complexity = 1, 
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5
    ) as hands:

    
    while cam.isOpened():                   # Loop runs continuously as long as the video capture object is open
        captured, frame_bgr = cam.read()    # Attempts to capture next frame
        if not captured:                    # Failed to capture next frame
            print("Camera inaccessible.")
            break


        # MediaPipe expects the input image to be in the RGB color format
        # OpenCV displays video frames in BGR
        # A copy of the frame in each format is made

        frame_bgr = cv2.flip(frame_bgr, 1) # Flips frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        res = hands.process(frame_rgb)  # The RGB format is used for processing

        # Draw hand landmarks on feed
        if res.multi_hand_landmarks:                        # Check if hand landmarks have been found
            for hand_landmarks in res.multi_hand_landmarks: # Iterate over each hand landmark found
                mp_drawing.draw_landmarks(                  # Draws landmarks and connections
                    frame_bgr, hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,

                    # Draws landmarks
                    mp_drawing.DrawingSpec(                 
                        color = (0, 255, 0), 
                        thickness = 2, 
                        circle_radius = 2
                    ), 

                    # Draws connections between landmarks
                    mp_drawing.DrawingSpec(                 
                        color = (255, 0, 0), 
                        thickness = 2
                    )
                )
        
        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display FPS
        cv2.putText(
            frame_bgr,                  # Frame to display FPS on
            f"FPS: {int(fps)}",         # Text to display
            (10, 30),                   # Position
            cv2.FONT_HERSHEY_COMPLEX,   # Font
            1,                          # Font scale
            (0, 255, 0),                # Color
            2,                          # Thickness
            cv2.LINE_AA                 # Anti-aliased lines (??)
        )

        cv2.imshow("Hand Detection Test", frame_bgr)    # The BGR format is used for displaying the frame

        # Program exit (Press 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcame and close window
cam.release()
cv2.destroyAllWindows()
