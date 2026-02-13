#importing libraries
import cv2
import numpy as np
import mediapipe as mp

#Mediapipe Hand Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
bg_img = cv2.imread("butterfly.webp")
bg_img = cv2.resize(bg_img, (640, 480))

# Create webcam and canvas
cap = cv2.VideoCapture(0)
c=0

#previous finger positions
prev_x, prev_y = None, None 
eraser_prev_x, eraser_prev_y = None, None
left_hand_landmarks = None
right_hand_landmarks = None
colour = (0, 0, 0)
mode = "Colour:"

def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]      # Finger tips
    mcps = [6, 10, 14, 18]       # Finger MCP joints

    for tip, mcp in zip(tips, mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            return False  # finger is open

    return True  # all fingers closed

def colours(colour):
    if(colour == (255, 0, 0)):
        return "Blue"
    if(colour == (0, 255, 255)):
        return "Yellow"
    if(colour == (0, 255, 0)):
        return "Green"
    if(colour == (0, 0, 255)):
        return "Red"
    if(colour == (255, 255, 0)):
        return "Cyan"
    if(colour == (255, 0, 255)):
        return "Pink"
    return ""


def distance(a, b, c, d):
    return ((a-c)**2 + (b-d)**2)**0.5

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if c==0:
            c=c+1
            h, w, _ = bg_img.shape
            canvas = np.zeros_like(frame) 

        if not ret:
            break
        display = bg_img.copy()
        
        # Showing the Colours on the screen
        cv2.rectangle(display, (w//7 - 25, 475), (w//7 + 25, 425), (255, 0, 0), thickness=-1)
        cv2.rectangle(display, ((2*w)//7 - 25, 475), ((2*w)//7 + 25, 425), (0, 255, 0), thickness=-1)
        cv2.rectangle(display, ((3*w)//7 - 25, 475), ((3*w)//7 + 25, 425), (0, 0, 255), thickness=-1)
        cv2.rectangle(display, ((4*w)//7 - 25, 475), ((4*w)//7 + 25, 425), (255, 255, 0), thickness=-1)
        cv2.rectangle(display, ((5*w)//7 - 25, 475), ((5*w)//7 + 25, 425), (255, 0, 255), thickness=-1)
        cv2.rectangle(display, ((6*w)//7 - 25, 475), ((6*w)//7 + 25, 425), (0, 255, 255), thickness=-1)

        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        #contains the information about detected hands, landmarks and positions
        result = hands.process(rgb)

        rgb.flags.writeable = True

        left_hand_landmarks = None
        right_hand_landmarks = None
        mode = "Colour:"
        if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):

                label = hand_info.classification[0].label

                if label == "Left":
                    left_hand_landmarks = hand_landmarks
                if label == "Right":
                    right_hand_landmarks = hand_landmarks

        if left_hand_landmarks and right_hand_landmarks:
            index_x = int(right_hand_landmarks.landmark[8].x * frame.shape[1])
            index_y = int(right_hand_landmarks.landmark[8].y * frame.shape[0])
            print(f"left fist = {is_fist(left_hand_landmarks)}")
            if not (is_fist(left_hand_landmarks)):
                eraser_prev_x, eraser_prev_y = None, None
                if(index_x>=((6*w)//7 - 25) and index_x<=((6*w)//7 + 25) and index_y>=425 and index_y<=475):
                    colour = (0, 255, 255)
                elif(index_x>=((5*w)//7 - 25) and index_x<=((5*w)//7 + 25) and index_y>=425 and index_y<=475):
                    colour = (255, 0, 255)
                elif(index_x>=((4*w)//7 - 25) and index_x<=((4*w)//7 + 25) and index_y>=425 and index_y<=475):
                    colour = (255, 255, 0)
                elif(index_x>=((3*w)//7 - 25) and index_x<=((3*w)//7 + 25) and index_y>=425 and index_y<=475):
                    colour = (0, 0, 255)
                elif(index_x>=((2*w)//7 - 25) and index_x<=((2*w)//7 + 25) and index_y>=425 and index_y<=475):
                    colour = (0, 255, 0)
                elif(index_x>=(w//7 - 25) and index_x<=(w//7 + 25) and index_y>=425 and index_y<=475):
                    colour = (255, 0, 0)
                prev_x, prev_y = None, None 
            else:
                mode = "Eraser"
                eraser_thickness = 10
                colour = (0,0,0)
                if eraser_prev_x is None and eraser_prev_y is None:
                    eraser_prev_x, eraser_prev_y = index_x, index_y

                cv2.line(canvas, (eraser_prev_x, eraser_prev_y), (index_x, index_y), (0, 0, 0),eraser_thickness)

                eraser_prev_x, eraser_prev_y = index_x, index_y
                prev_x, prev_y = None, None  
                
                        
        elif right_hand_landmarks:
            eraser_prev_x, eraser_prev_y = None, None
            # Index finger tip is landmark 8
            index_x = int(right_hand_landmarks.landmark[8].x * frame.shape[1])
            index_y = int(right_hand_landmarks.landmark[8].y * frame.shape[0])

            # Middle finger tip is landmark 12
            middle_x = int(right_hand_landmarks.landmark[12].x * frame.shape[1])
            middle_y = int(right_hand_landmarks.landmark[12].y * frame.shape[0])

            if prev_x is None and prev_y is None:
                prev_x, prev_y = index_x, index_y

            # print(distance(index_x, index_y, middle_x, middle_y))   
            if(distance(index_x, index_y, middle_x, middle_y) > 50):
                thickness = 5
            else:
                thickness = 15

            cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), colour, thickness)

            prev_x, prev_y = index_x, index_y
            # mp_drawing.draw_landmarks(combined, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = None, None  

        combined = cv2.addWeighted(display, 0.7, canvas, 0.3, 0)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    combined,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        cv2.putText(combined, f'{mode} {colours(colour)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Air Drawing", combined)
        cv2.imshow("Raw Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):  
            canvas = np.zeros_like(bg_img)

cap.release()
cv2.destroyAllWindows()
