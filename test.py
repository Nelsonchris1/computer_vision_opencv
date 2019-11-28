import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

image1 = image.copy()
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image1, (9, 9), 0)

# detecting coins in an image using various threshold methods

# Using simple threshold
(T, threshINV) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
final = cv2.bitwise_and(image1,image1, mask=threshINV)
cv2.imshow("Threshold ", final)
cv2.waitKey(0)

(cnts,_) = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("There are {} coins in this simplpe threshold image".format(len(cnts)))

coins1 = image.copy()
cv2.drawContours(coins1, cnts, -1, (0, 0, 255), 2)
cv2.imshow("Using simple threshold", coins1)
cv2.waitKey(0)

# Using adaptive threshold
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Adaptive gaussian thresh", thresh)
cv2.waitKey(0)

(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("There are {} coins in this adaptive threshold".format(len(cnts)))


coin2 = image.copy()
cv2.drawContours(coin2, cnts, -1, (255, 0, 0), 2)
cv2.imshow("Using adaptive threshold", coin2)
cv2.waitKey(0)

#Using otsu and calvard threshold
import mahotas
T = mahotas.thresholding.otsu(blurred)
Thresh = image1.copy()
Thresh[Thresh > T] = 255
Thresh[Thresh < T] = 0
thresh = cv2.bitwise_not(Thresh)
cv2.imshow("Otsu Threshold", thresh)
cv2.waitKey(0)

(cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("There are {} coins in this adaptive threshold".format(len(cnts)))

coin3 = image.copy()
cv2.drawContours(coin3, cnts, -1, (255, 255, 255), 2)
cv2.imshow("Otsu contour", coin3)
cv2.waitKey(0)