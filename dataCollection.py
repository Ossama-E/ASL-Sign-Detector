import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

space = 20
imgSize = 300
counter = 0

folder = "Data/C"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, width, height = hand["bbox"]

        # 0 -> 255
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - space: y + height + space, x - space: x + width + space]

        imgCropShape = imgCrop.shape

        aspectRatio = height / width

        if aspectRatio > 1:
            k = imgSize / height
            widthCalc = math.ceil(k * width)
            imgResize = cv2.resize(imgCrop, (widthCalc, imgSize))
            imgResizeShape = imgResize.shape

            widthPush = math.ceil((imgSize - widthCalc) / 2)

            imgWhite[:, widthPush: widthCalc + widthPush] = imgResize

        else:
            k = imgSize / width
            heightCalc = math.ceil(k * height)
            imgResize = cv2.resize(imgCrop, (imgSize, heightCalc))
            imgResizeShape = imgResize.shape

            heightPush = math.ceil((imgSize - heightCalc) / 2)

            imgWhite[heightPush: heightCalc + heightPush, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg' , imgWhite)
        print(counter)