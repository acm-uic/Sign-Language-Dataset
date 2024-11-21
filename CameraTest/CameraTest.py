import cv2
import mediapipe as mp
import time
import numpy as np

# Initalize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Open the default camera
vidCam = cv2.VideoCapture(0)
if not vidCam.isOpened():
    print("Camera inaccessible.")
    exit()

# Variables for FPS Calc
prevTime = 0

#Variable for Picture Counter

counter=0

# Setting up the MediaPipe Hands object
with mp_hands.Hands(
        model_complexity = 1, 
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5
    ) as hands:

    
    while vidCam.isOpened():                # Loop runs continuously as long as the video capture object is open
        captured, frameBGR = vidCam.read()  # Attempts to capture next frame
        if not captured:                    # Failed to capture next frame
            print("Camera inaccessible.")
            break


        # MediaPipe expects the input image to be in the RGB color format
        # OpenCV displays video frames in BGR
        # A copy of the frame in each format is made

        frameBGR = cv2.flip(frameBGR, 1)    # Flips frame
        frameRGB = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)

        res = hands.process(frameRGB)   # The RGB format is used for processing

        wireImage = np.zeros_like(frameBGR) # Black screen to show just wireframe

        # Draw hand landmarks on feed
        if res.multi_hand_landmarks:                        # Check if hand landmarks have been found
            for handLandmarks in res.multi_hand_landmarks:  # Iterate over each hand landmark found

                # Draws landmarks and connections on image
                mp_drawing.draw_landmarks(
                    frameBGR, handLandmarks, 
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

                # Draws landmarks and connections on black screen
                mp_drawing.draw_landmarks(
                    wireImage, handLandmarks,
                    mp_hands.HAND_CONNECTIONS,

                    # Draws landmarks
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), 
                        thickness=2, 
                        circle_radius=2
                    ),

                    # Draws connections between landmarks
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 0), 
                        thickness=2
                    )
                )
        
        # FPS Calculation
        currTime = time.time()
        framesPerSecond = 1 / (currTime - prevTime) if prevTime != 0 else 0
        prevTime = currTime

        # Picture Counter
        counter=int(currTime%3)+1
        
        

        # Display FPS
        cv2.putText(
            frameBGR,                       # Frame to display FPS on
            f"FPS: {int(framesPerSecond)}", # Text to display
            (10, 30),                       # Position
            cv2.FONT_HERSHEY_COMPLEX,       # Font
            1,                              # Font scale
            (0, 255, 0),                    # Color
            2,                              # Thickness
            cv2.LINE_AA                     # Anti-aliased lines (??)
        )
        # Display Time
        cv2.putText(
            frameBGR,                       # Frame to display FPS on
            f"Time 1-3: {int(counter)}", # Text to display
            (380, 30),                       # Position
            cv2.FONT_HERSHEY_COMPLEX,       # Font
            1,                              # Font scale
            (0, 0, 50),                    # Color
            2,                              # Thickness
            cv2.LINE_AA                     # Anti-aliased lines (??)
        )

        cv2.imshow("Hand Detection Test (Press any key to exit)", frameBGR) # Displaying the camera
        cv2.imshow("Mapped Hand Wireframe (Press any key to exit)", wireImage) # Displaying the wireframe

        # Program exit (press any key)
        if cv2.waitKey(1) != -1:
            break

# Release webcam and close window
vidCam.release()
cv2.destroyAllWindows()
