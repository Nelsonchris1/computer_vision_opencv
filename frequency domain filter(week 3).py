import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image_filer.jpg")
cv2.imshow("Original", image)

image = cv2.resize(image, (600, 600), cv2.INTER_AREA)
#cv2.imshow("Original", image)


# Pick one color top perform fourier transform to
"""
An image in its original state is a spacial domain
To convert this image to the frequency domain, we perform fourier transform on the image  
"""
# image1 = image.copy()
# image1 = image1[:, :, 2]
# image2 = image[:, :, : 2]
# print(image2.shape)
# # cv2.imshow("Blue channel", image1)
# # cv2.waitKey(0)
#
#
#
# r = 10
# ham = np.hamming(600)[:, None]
# ham2dim = np.sqrt(np.dot(ham, ham.T) ** r)
#
#
# f_trans =  cv2.dft(np.float32(image2), cv2.DFT_COMPLEX_OUTPUT)
# f_shift = np.fft.fftshift(f_trans)
# f_complex = f_shift[:, :, 0] * 1j + f_shift[:, :, 1]
# f_filtered = ham2dim * f_complex
# f_filtered = np.fft.fftshift(f_filtered)
# inv_img = np.fft.ifft2(f_filtered)
#
# filtered_img = np.abs(inv_img)
# f_abs = np.abs(f_complex) + 1
# f_log = 20 * np.log(f_abs)
# f_img = 255 * f_log / np.max(f_log)
# magnitude = f_img.astype("uint8")
# cv2.imshow("FIltered_image", magnitude)
#
# filtered_img -= filtered_img.min()
# filtered_img = filtered_img * 255 / filtered_img.max()
# filtered_img = filtered_img.astype("uint8")
#
# #stack = np.hstack([magnitude, filtered_img])
#
# cv2.imshow("FIltered_image", filtered_img)
# cv2.waitKey(0)


"""
Fourier transform returns a 2d complex array.
complex array include Real and imaginary part
since np.float32 already worked on the real part,
we use 1j * f_shift for the imaginary part
"""
# f_complex = f_shift[:, :, 0] + (1j * f_shift[:, :, 1])
#
# # Next we take the absolute value so the image can be diplayed
# f_abs = np.abs(f_complex) + 1
#
# # since f_complex has a wide range even after getting the abs value, we perform the log
# f_uni = 20 * np.log(f_abs)
#
# # multiply by 255 to make sure the pixel scale is between 0 - 255
# f_output = 255 * f_uni / np.max(f_uni)
# f_output = f_output.astype(np.uint8)
#
# # Here we display the Frequency domain image
# cv2.imshow("Fourier transform", f_output)
# cv2.waitKey(0)

