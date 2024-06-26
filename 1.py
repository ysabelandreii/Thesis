import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 600

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

for j in range(16,21):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    done = False
    while not done:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Display instructions
        cv2.putText(frame, 'Ready? Press "S" !', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('s'):
            done = True

    counter = 400
    while counter <= dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        file_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(file_path, frame)
        print('Saved:', file_path)

        counter += 1

cap.release()
cv2.destroyAllWindows()
