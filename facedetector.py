import cv2 

from random import randrange

# Load pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#To capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    #Reads current frame
    successful_frame_read, frame = webcam.read()

    #Image converter to grayscale. It is important for algorithm that images be black and white becouse its easier to detect fewer pixels
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Locates face coordinates in image, detectMultiScale with this function you can detect any size faces, no matter if they further away or closer
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    #Draw rectangles around the face using this function we use original image
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,randrange(256), 0), 4)

    cv2.imshow('Frontal Face Detector App', frame)
    key = cv2.waitKey(1)

    #Stops if you press Q key, 81,113 is ascii location of key Q
    if key==81 or key==113:
        break


#Release the VideoCapture object
webcam.release()
