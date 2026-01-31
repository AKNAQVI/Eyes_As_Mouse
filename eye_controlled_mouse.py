import cv2
import mediapipe 
import pyautogui
import numpy as np

face_mesh = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam = cv2.VideoCapture(0)
while True:
    _,image = cam.read()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh.process(rgb_image)
    all_face_landmarks = processed_image.multi_face_landmarks
    print(all_face_landmarks)
    cv2.imshow("Eye controlled mouse", image)
    key = cv2.waitKey(100)
    if key == 27:  # ESC key to exit
        break
cam.release()
cv2.destroyAllWindows()
