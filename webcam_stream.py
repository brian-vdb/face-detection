import os
import time
import cv2

# Define the width and height
WIDTH = 640
HEIGHT = 480

# Initialize the YuNet model
yunet_model_path = os.path.join('models', 'face_detection_yunet_2023mar.onnx')
yunet = cv2.FaceDetectorYN.create(
    model=      yunet_model_path,
    config=     "",
    input_size= (WIDTH, HEIGHT)
)

# Initialize the webcam stream
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the stream properties
codec = 0x47504A4D  # MJPG
cap.set(cv2.CAP_PROP_FOURCC, codec)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60.0)

# Initialize FPS Counter
prevFrame = time.time()
curFrame = time.time()

while True:
    # Get a frame
    ret, frame = cap.read()
    if ret:
        # Manage FPS counter
        curFrame = time.time()
        fps = 1 / (curFrame - prevFrame)
        prevFrame = curFrame
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (8, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Perform detection on the frame with YuNet
        _, faces = yunet.detect(frame)

        # Draw rectangles
        if faces is not None:
            for face in faces:
                # bouding box
                box = list(map(int, face[:4]))
                for i, val in enumerate(box):
                    if val < 0:
                        box[i] = 0

                # Define bounding box rectangle
                color = (0, 255, 0)
                cv2.rectangle(frame, box, color, 1)

        # Display the Result
        cv2.imshow("Display", frame)

    # Check for exit key 'q'
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Clean the application
cv2.destroyAllWindows()
