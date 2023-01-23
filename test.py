import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier


# Create a VideoCapture object to access the default camera (0)
cap = cv2.VideoCapture(0)

# Create an instance of HandDetector class with maxHands=1
detector = HandDetector(maxHands=1)

# Create an instance of Classifier class, loading in a pre-trained Keras model and a file of labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Initialize variables for image cropping and resizing
space = 20
imgSize = 300
counter = 0

# List of labels for the hand gestures
labels = ["A", "B", "C", "F", "I"]

# Enter a while loop that captures video frames continuously
while True:
    img: object

    # Capture a frame from the video
    success, img = cap.read()

    # Make a copy of the original frame
    imgOutput = img.copy()

    # Find any hands in the frame using the detector object
    hands, img = detector.findHands(img)

    # If hands are detected
    if hands:

        # Get the bounding box of the hand
        hand = hands[0]
        x, y, width, height = hand["bbox"]

        # 0 -> 255
        # Create a white image with the same size as the desired resized image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the image around the hand
        imgCrop = img[y - space : y + height + space, x - space : x + width + space]
        imgCropShape = imgCrop.shape

        # Get the aspect ratio of the hand
        aspectRatio = height / width

        # Resize the image based on the aspect ratio
        if aspectRatio > 1:
            k = imgSize / height
            widthCalc = math.ceil(k * width)
            imgResize = cv2.resize(imgCrop, (widthCalc, imgSize))
            imgResizeShape = imgResize.shape

            # Push the resized image to the center of the white image
            widthPush = math.ceil((imgSize - widthCalc) / 2)
            imgWhite[:, widthPush : widthCalc + widthPush] = imgResize

            # Get the prediction and index of the hand gesture from the classifier
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        else:
            k = imgSize / width
            heightCalc = math.ceil(k * height)
            imgResize = cv2.resize(imgCrop, (imgSize, heightCalc))
            imgResizeShape = imgResize.shape

            # Push the resized image to the center of the white image
            heightPush = math.ceil((imgSize - heightCalc) / 2)
            imgWhite[heightPush : heightCalc + heightPush, :] = imgResize

            # Get the prediction and index of the hand gesture from the classifier
            prediction, index = classifier.getPrediction(imgWhite)

        # Draw a rectangle on the output frame around the bounding box of the hand
        cv2.rectangle(
            imgOutput,
            (x - space + 90, y - space - 50),
            (x - space + 90, y - space - 50 + 50),
            (255, 0, 255),
            cv2.FILLED,
        )

        # Draw the label of the predicted hand gesture on the output frame
        cv2.putText(
            imgOutput,
            labels[index],
            (x, y - 26),
            cv2.FONT_HERSHEY_PLAIN,
            1.7,
            (255, 0, 255),
            2,
        )

        cv2.rectangle(
            imgOutput,
            (x - space, y - space),
            (x + width + space, y + height + space),
            (255, 0, 255),
            4,
        )

        # Show the output frame, the cropped hand image, and the resized image on the screen
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

