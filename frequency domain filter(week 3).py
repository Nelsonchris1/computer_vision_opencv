import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image_filer.jpg")

print(image.shape)




# Pick one color top perform fourier transform to
"""
An image in its original state is a spacial domain image
To convert this image to the frequency domain, we perform fourier transform on the image  
"""

def frequency_filter(image, dim = 450, r = 20, return_frames = False):


    image = cv2.resize(image, (dim, dim), cv2.INTER_AREA)
    image1 = image.copy()
    image2 = image1[:, :, : 2]



    r = r

    #We use hamming window to perform filtering

    ham = np.hamming(dim)[:, None]
    ham2dim = np.sqrt(np.dot(ham, ham.T) ** r)

    f_trans = cv2.dft(np.float32(image2), cv2.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f_trans)

    """
    Fourier transform returns a 2d complex array.
    complex array include Real and imaginary part
    since np.float32 already worked on the real part,
    we use 1j * f_shift for the imaginary part
    """
    f_complex = f_shift[:, :, 0] * 1j + f_shift[:, :, 1]

    # apply hamming window mask to f_complex
    f_filtered = ham2dim * f_complex

    f_filtered = np.fft.fftshift(f_filtered)

    # perfome inverse fourier transform to apply it to original image
    inv_img = np.fft.ifft2(f_filtered)

    # Next we take the absolute value so the image can be diplayed
    filtered_img = np.abs(inv_img)
    f_abs = np.abs(f_complex) + 1
    f_log = 20 * np.log(f_abs)

    # multiply by 255 to make sure the pixel scale is between 0 - 255
    f_img = 255 * f_log / np.max(f_log)

    magnitude = f_img.astype("uint8")

    filtered_img -= filtered_img.min()
    filtered_img = filtered_img * 255 / filtered_img.max()
    filtered_img = filtered_img.astype("uint8")

    if return_frames == True:

        # Here we display the Frequency domain image/magnitude spectrum and the filtered image
        stack = np.hstack([magnitude, filtered_img])
        cv2.imshow("FIltered_image", stack)
        cv2.waitKey(0)




frequency_filter(image, 450, 5, True)

beach = cv2.imread("beach.png")

frequency_filter(image=beach, dim=300, r= 0 ,return_frames = True)



image = cv2.imread("image_filer.jpg", 0)

def low_pass_filter(image, dim = 450, n = 50, return_frames = False):
    image = cv2.resize(image, (dim, dim), cv2.INTER_AREA)
    image1 = image.copy()




    crows, ccols = dim // 2, dim // 2
    mask = np.zeros((dim, dim), np.uint8)
    mask[crows - n: crows + n, ccols - n:ccols + n] = 1

    dft = cv2.dft(np.float32(image1), cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    fshift = dft_shift * mask
    f_shift = np.fft.fftshift(fshift)
    f_inv = cv2.idft(f_shift)
    filtered_img = np.abs(f_inv)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img * 255 / filtered_img.max()
    filtered_img = filtered_img.astype("uint8")

    if return_frames == True:

        cv2.imshow("Filtered_image", filtered_img)
        cv2.waitKey(0)


low_pass_filter(image, 450, 200, True)
low_pass_filter(image, 450, 100, True)

