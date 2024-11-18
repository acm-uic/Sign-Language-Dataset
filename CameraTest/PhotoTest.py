import mediapipe as mp
import cv2
from tkinter.filedialog import askopenfilename

# Get file to read from user
filename = askopenfilename()
if not filename:
     print("No file selected. Exiting.")
     exit()

# Initalize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Turn file into cv2 object
imgBGR = cv2.imread(filename)
if imgBGR is None:
     print("Error loading image. Exiting.")
     exit()


# Setting up the MediaPipe Hands object
with mp_hands.Hands(
        model_complexity = 1, 
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5
    ) as hands:

    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)    # Convert to RGB
    res = hands.process(imgRGB)                         # Process image

    # Draw hand landmarks on image
    if res.multi_hand_landmarks:                            # Check if hand landmarks have been found
            for hand_landmarks in res.multi_hand_landmarks: # Iterate over each hand landmark found
                mp_drawing.draw_landmarks(                  # Draws landmarks and connections
                    imgBGR, hand_landmarks, 
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
    
    cv2.imshow("Hand Detection Test (Press any key to exit)", imgBGR)   # Displaying the frame

    # Program exit (press any key)
    while True:
         if cv2.waitKey(1) != -1:
            break

# Close window
cv2.destroyAllWindows()
