import cv2
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

def removeBiggestElementsOnBinaryImage(image):
    duplicatedImage = image.copy()
    # Apply erosion
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=3)
    # Apply dilation
    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=15)
    imshow(dilation)
    # Negate images
    dilation = cv2.bitwise_not(dilation)
    # Apply AND operation
    andOperation = cv2.bitwise_and(dilation, duplicatedImage)
    return andOperation