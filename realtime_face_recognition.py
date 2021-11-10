import cv2

# Load trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video fom webcam
webcam = cv2.VideoCapture(0)

# Iterate over frames
while True:
    
    successful_frame_read, frame = webcam.read()
    
    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    # Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
       cv2.rectangle(frame, (x ,y), (x+w, y+h) ,(0 , 250, 0), 2)
 
    cv2.imshow('real-time face detector', frame)
    key = cv2.waitKey(1)
    
    # Stop if Q is pressed
    if key==81 or key==113:
        break 
