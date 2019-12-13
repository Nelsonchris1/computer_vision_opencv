import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to image")
# args = vars(ap.parse_args())
# image = cv2.imread(args["image"])
# cv2.imshow("Original", image)
# cv2.waitKey(0)
#
#### image filtering
#
# image_blur = cv2.blur(image, (3, 3))
# image_median = cv2.medianBlur(image, 3)
# image_gaussian = cv2.GaussianBlur(image, (3, 3), 0)
# image_filter = cv2.bilateralFilter(image, 3, 31, 31)
#
#
# blurred = np.hstack([image_blur, image_median, image_gaussian, image_filter])
#
# cv2.imshow("Average, Meidan, Gaussian, Bilateral Filter", blurred)
# cv2.waitKey(0)
#
#### invert an image
#
# invert = cv2.bitwise_not(image)
# cv2.imshow("Original", image)
# cv2.imshow("Inverted image", invert)
# cv2.waitKey(0)
#
#
# video_capture = cv2.VideoCapture(0)
#
# frames_per_second = 20
#
#### Inverting a live vdieo
# def invert(frame):
#     return cv2.bitwise_not(frame)
#
#
#### Create a fuction for a vintage filter
# def verify_alpha_channel(frame):
#     try:
#         frame.shape[3]
#     except IndexError:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
#     return frame
#
# def apply_vintage(frame, intensity=0.9):
#     frame = verify_alpha_channel(frame)
#     h, w, c = frame.shape
#     blue = 20
#     green = 66
#     red = 112
#     serpia_bgra = (blue, green, red, 1)
#     overlay = np.full((h, w, 4), serpia_bgra, dtype= "uint8")
#     cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
#     return frame
#
# def HSV(frame):
#     return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#### Applying the vintage filter to an image
# cv2.imshow("Original", image)
# cv2.imshow("Vinatge", apply_vintage(image))
# cv2.waitKey(0)
#
# #Applying vintage filter to live video
#
# while True:
#     _, original_frame = video_capture.read()
#     Invert = apply_vintage(original_frame.copy())
#     cv2.imshow("Invert", Invert)
#     cv2.imshow("original_frame", original_frame)
#     if cv2.waitKey(2) & 0xFF == ord("q"):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()
#
#
#### A function to smoothen and add a vintage filter to either a video or image
#
# def smooth_vintage(frame):
#     blurred = cv2.GaussianBlur(frame, (5, 5), 0)
#
#     return apply_vintage(blurred)
#
# cv2.imshow("Original", image)
# cv2.imshow("Vinatge", apply_vintage(image))
# cv2.imshow("Vintage plus smooth", smooth_vintage(image))
# cv2.waitKey(0)
#
#### smoothning a live video and applying vintage filter
#
# while True:
#     _, original_frame = video_capture.read()
#     Invert = smooth_vintage(original_frame.copy())
#     cv2.imshow("Invert", Invert)
#     cv2.imshow("original_frame", original_frame)
#     if cv2.waitKey(2) & 0xFF == ord("q"):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()





#### Diplaying number of ball in an image

# image1 = cv2.imread("4 balls.jpg")
# image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(image2, (11, 11), 0)
#
#
# edged = cv2.Canny(blurred, 30, 150)
# (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("I count {} balls in this image".format(len(cnts)))
# ball = image1.copy()
# cv2.drawContours(ball, cnts, -1, (0, 255, 0), 2)
# cv2.imshow("Original", image1)
# cv2.imshow("Balls contuored", ball)
# cv2.waitKey(0)