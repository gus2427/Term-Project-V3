import cv2
import numpy as np

def convert_to_nbit_grayscale(img, n_bits):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    step=int(256/2**n_bits)

    value_list=np.arange(step,256+1,step)

    # Create an array to store the mapped values
    bitMatrix = np.zeros_like(gray, dtype=int)

    # Iterate through the matrix and apply the mapping
    for i in range(len(value_list)):
        if i == 0:
            bitMatrix[gray < value_list[i]] = i
        else:
            bitMatrix[(gray >= value_list[i - 1]) & (gray < value_list[i])] = i

    scale=255/(2**n_bits-1)
    scaled_gray=(np.round(bitMatrix*scale)).astype(np.uint8)

    return scaled_gray, bitMatrix