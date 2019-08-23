import cv2
import numpy as np


def center_crop(images, patch_size):
    height, width = images.shape[1:3]
    cropped_images = []
    for image in images:
        bottom = height // 2 - (patch_size // 2)
        top = height // 2 + (patch_size - patch_size // 2)
        left = width // 2 - (patch_size // 2)
        right = width // 2 + (patch_size - patch_size // 2)
        cropped_images.append(image[bottom:top, left:right])

    return np.stack(cropped_images)
