import cv2
import numpy as np

def convert_to_nbit_RGB(img, n_bits):
    b,g,r=img[:,:,0],img[:,:,1],img[:,:,2]

    step=int(256/2**n_bits)

    dic = {
        0:b,
        1:g,
        2:r
    }

    value_list=np.arange(step,256+1,step)
    bitMatrix = np.zeros_like(img, dtype=int)

    for i,channel in dic.items():
        bitMatrix[:,:,i] = np.digitize(channel, value_list)
        bitMatrix[bitMatrix < 0] = 0

    scale=255/(2**n_bits-1)

    scaled_RGB=(np.round(bitMatrix*scale)).astype(np.uint8)

    return scaled_RGB, bitMatrix