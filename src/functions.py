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

def dessiner_rectangle_centre(image, ligne, epaisseur, couleur):
    x1, y1, x2, y2 = ligne

    # Vecteur directeur de la ligne
    vecteur_ligne = np.array([x2 - x1, y2 - y1])
    longueur_ligne = np.linalg.norm(vecteur_ligne)

    # Vecteur perpendiculaire à la ligne
    vecteur_perpendiculaire = np.array([-vecteur_ligne[1], vecteur_ligne[0]])
    vecteur_perpendiculaire = vecteur_perpendiculaire / np.linalg.norm(vecteur_perpendiculaire) * epaisseur / 2

    # Calculer les coins du rectangle
    coin1 = np.array([x1, y1]) + vecteur_perpendiculaire
    coin2 = np.array([x1, y1]) - vecteur_perpendiculaire
    coin3 = np.array([x2, y2]) - vecteur_perpendiculaire
    coin4 = np.array([x2, y2]) + vecteur_perpendiculaire

    # Dessiner le rectangle
    points = np.array([coin1, coin2, coin3, coin4], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(image, [points], color=couleur)
