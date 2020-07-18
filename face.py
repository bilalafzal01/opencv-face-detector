import cv2
from random import randrange

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# loading pretrained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# loading an image to detect faces in
# img = cv2.imread('eminem-dre.jpg')

# capture from webcam
webcam = cv2.VideoCapture(0)
# key = cv2.waitKey(1)

# iterate over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # converting to greyscaled image
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get face coordinates
    face_coordinates = trained_face_data.detectMultiScale(greyscale_img)

    # iterate over each face coordinats tuple
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)

    cv2.imshow('Bilal Face Detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()

print("End of program!")