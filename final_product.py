import pickle
import pyttsx3
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open the camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary
labels_dict = {1: 'A', 2: 'B', 3: 'C' , 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H' ,9: 'I', 10:
    'J',  11: 'K', 12: 'L', 13: 'M' , 14: 'N', 15: 'O',  16: 'P', 17: 'R', 18: 'S' , 19: 'T', 20: 'U',   
    21: 'V', 22: 'W', 23: 'X' , 24: 'Y', 25: 'Z', 26: 'Mahal Kita' , 27: 'Hello' , 28: 'Ano',
    29: 'Pasensia kana' , 30: 'paki', 31: 'isa', 32: 'dalawa', 33: 'tatlo' , 34: 'apat', 35: 'lima',
    36: 'anim', 37: 'pito', 38: 'walo' , 39: 'siyam', 40: 'sampu', 41: 'oo', 42: 'hindi', 43: 'salamat' ,
    44: 'anong gina gawa mo', 45: 'banyo',  46: 'paalam', 47: 'ayoko sa iyo', 48: 'mahal na mahal kita' ,
    49: 'ka in', 50: 'ang cute', 51: 'aglit', 52: 'kita kits', 53: 'magkano' , 54: 'ambulansiya', 55: 'eroplano'}

while True:
    try:
        # Read a frame from the camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if frame is not None:
            # Get frame dimensions
            H, W, _ = frame.shape

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame using MediaPipe Hands
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                if x_ and y_:  # Check if x_ and y_ are not empty
                    for i in range(len(x_)):
                        data_aux.append(x_[i] - min(x_))
                        data_aux.append(y_[i] - min(y_))

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    # Predict the gesture
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    print(predicted_character)

                    # Initialize the text-to-speech engine
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)

                    # Speak out the predicted character
                    engine.say(predicted_character)
                    engine.runAndWait()

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Show the frame
            cv2.imshow('frame', frame)
            cv2.waitKey(50)
        else:
            print("Error: Failed to read frame from video capture device.")

    except Exception as e:
        print("An error occurred:", e)

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
