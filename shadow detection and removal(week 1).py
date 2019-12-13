import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

# image1 = image.copy()
# image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(image1, (9, 9), 0)


#### detecting coins in an image using various threshold methods

# #Using simple threshold
# (T, threshINV) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
# final = cv2.bitwise_and(image1,image1, mask=threshINV)
# cv2.imshow("Threshold ", final)
# cv2.waitKey(0)
#
# (cnts,_) = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("There are {} coins in this simplpe threshold image".format(len(cnts)))
#
# coins1 = image.copy()
# cv2.drawContours(coins1, cnts, -1, (0, 0, 255), 2)
# cv2.imshow("Using simple threshold", coins1)
# cv2.waitKey(0)

#### Using adaptive threshold
# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
# cv2.imshow("Adaptive gaussian thresh", thresh)
# cv2.waitKey(0)
#
# (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("There are {} coins in this adaptive threshold".format(len(cnts)))


# coin2 = image.copy()
# cv2.drawContours(coin2, cnts, -1, (255, 0, 0), 2)
# cv2.imshow("Using adaptive threshold", coin2)
# cv2.waitKey(0)

# #Using otsu  threshold to find contour and draw out contour
# import mahotas
# T = mahotas.thresholding.otsu(blurred)
# Thresh = image1.copy()
# Thresh[Thresh > T] = 255
# Thresh[Thresh < T] = 0
# thresh = cv2.bitwise_not(Thresh)
# cv2.imshow("Otsu Threshold", thresh)
# cv2.waitKey(0)

# (cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("There are {} coins in this otsu threshold".format(len(cnts)))

# coin3 = image.copy()
# cv2.drawContours(coin3, cnts, -1, (255, 255, 255), 2)
# text = "There are  {} coins in this image".format(len(cnts))
# cv2.putText(coin3, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# cv2.imshow("Otsu contour", coin3)
# cv2.waitKey(0)

#### Using calvard  to find contour and draw out the contour
# T = mahotas.thresholding.rc(blurred)
# Thresh = image1.copy()
# Thresh[Thresh > T] = 255
# Thresh[Thresh < T] = 0
# thresh = cv2.bitwise_not(Thresh)
# cv2.imshow("calvards Threshold", thresh)
# cv2.waitKey(0)

# (cnts,_) = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("There are {} coins in this calvards threshold image".format(len(cnts)))
# coin3 = image.copy()
# cv2.drawContours(coin3, cnts, -1, (255, 255, 255), 2)
# text = "There are {} coins".format(len(cnts))
# cv2.putText(coin3, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# cv2.imshow("calvard contour", coin3)
# cv2.waitKey(0)

#Detecting shadow and removing shadows from an image
img = cv2.imread("shadow7.jpeg")
img1 = img.copy()

cv2.imshow("Original", img)
cv2.waitKey(0)
(B, G, R) = cv2.split(img)

result_chan =[]
result_norm_chan = []

for chan in (B, G, R):
    dilated_img = cv2.dilate(chan, np.ones((5, 5), np.uint8))
    blurred_img = cv2.GaussianBlur(dilated_img, (9, 9), 0)
    absdiff_img = 255 - cv2.absdiff(chan, blurred_img)
    norm_img = cv2.normalize(absdiff_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    result_chan.append(absdiff_img)
    result_norm_chan.append(norm_img)

result = cv2.merge(result_chan)
result_norm = cv2.merge(result_norm_chan)

cv2.imshow("Normalized", result_norm)
cv2.waitKey(0)


####Detecting and removing shadow from a black and white image
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Black and white", img1)
# cv2.waitKey(0)

# dilated_img = cv2.dilate(img1, np.ones((7, 7), np.uint8))
# blurred_img = cv2.medianBlur(img1, 9)
# #T, thresh = cv2.threshold(img1, 150, 255,cv2.THRESH_BINARY_INV)
# absdiff_img = 255 - cv2.absdiff(img1, blurred_img)

# cv2.imshow("Result", dilated_img)
# cv2.waitKey(0)


