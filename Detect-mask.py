from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

# faceNet captures the face from the frame
# maskNet checks the if mask /No-mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Take the dimensions of the frame and then construct a blob from it
    #blob is processed image
    # cv2.dnn.blobFromImage pre-processes the image. It does the mean subtraction for the RGB values
    # on the image & scales it with 1. Mean values are obtained from the total dataset.
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
    # Pass the blob through network and obtain face detection
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # detections are -[~,~,prediciton,startX,startY,endX,endY] for the detected faces in the frame

    # initialize the list of faces, their corresponding locations,
    # and the list of predictions from our face network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]

    # filter out the weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x,y) co-ordinates of the bounding boxes for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding boxes fall within the diimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI convert it from BGR to RGB channel
            # ordering, resizing it to 224*224 and process it

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Add the face and the bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a prediction if atleast one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)


# load out face detector model from disk
prototxtpath = r'face_detector\deploy.prototxt'
weightspath = r'face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtpath, weightspath)

# load the dace mask detecctor model from the disk
maskNet = load_model("Face_mask_detector.model")

# Initialize  the video stream
print("Starting video stream...")
vs = VideoStream(src=0).start()  # Source =0 gives the primary camera

# Loop over the frames from the video frame
while True:
    # Grab the frame from the threaded video stream and resize it to have
    # a maximum width of 800 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # Detect faces in the frame and determine if they are wearing a mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face location and their corresponding location
    for (box, pred) in zip(locs, preds):
        # Unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutmask) = pred

        # Determine the class label and color to draw the bounding boxes and text
        label = 'Mask' if mask > withoutmask else "No mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probabilit in the label
        label = "{}:{:.2f}%".format(label, max(mask, withoutmask) * 100)

        # display the label and the bounding box reactangle on the output frame
        cv2.putText(frame, label,(startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the q was pressed , break from the loop
    if key == ord("q"):
        break

# do clean up
cv2.destroyAllWindows()
vs.stop()
