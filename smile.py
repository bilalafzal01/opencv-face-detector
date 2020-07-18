import cv2
from random import randrange

trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


def detect(grey, frame):
    faces = trained_face_data.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roiGray = grey[y:y + h, x:x + w]
        roiColor = frame[y:y + h, x:x + w]
        smiles = trained_smile_data.detectMultiScale(roiGray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roiColor, (sx, sy), (sx + sw, sy + sh), (0, 255, 0),
                          2)
    return frame


webcam = cv2.VideoCapture(0)

while True:
    succesful_frame_read, frame = webcam.read()

    greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = detect(greyscale_frame, frame)

    cv2.imshow('Bilal Face Detector', canvas)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
print("Smile ended!")