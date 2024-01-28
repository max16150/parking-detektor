import cv2
import numpy as np

imageCount = 1

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

def imshow(img, name='IMG'):
    global imageCount
    randomNameString = name + '_' + str(imageCount)
    cv2.imshow(randomNameString, ResizeWithAspectRatio(img, width=720))
    imageCount += 1

def removeBiggestElementsOnBinaryImage(image):
    duplicatedImage = image.copy()
    # Apply erosion
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=3)
    # Apply dilation
    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=15)
    #imshow(dilation)
    # Negate images
    dilation = cv2.bitwise_not(dilation)
    # Apply AND operation
    andOperation = cv2.bitwise_and(dilation, duplicatedImage)
    return andOperation

# Charger l'image
image = cv2.imread(r"C:\Users\Maxime\Documents\Github\parking-detektor\res\images\7.png")

# Convertir l'image en HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Définir la plage de couleur pour le blanc
# Ces valeurs peuvent nécessiter des ajustements
lower_white = np.array([0, 0, 150])
upper_white = np.array([255, 20, 255])

# Créer un masque qui ne garde que le blanc
mask = cv2.inRange(hsv, lower_white, upper_white)

# Appliquer le masque pour obtenir seulement les parties blanches
binary = cv2.bitwise_and(image, image, mask=mask)

# Appliquer une expansion pour combler les trous
x = 2
kernel = np.ones((x, x), np.uint8)
binary = cv2.dilate(binary, kernel, iterations=0)


imshow(binary)

processed = removeBiggestElementsOnBinaryImage(binary)

#imshow(processed)


cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

# Détecter les contours
edges = cv2.Canny(binary, 50, 150, apertureSize=7)


# Appliquer la transformation de Hough
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=200, maxLineGap=50)

# Dessiner les lignes détectées
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(binary, (x1, y1), (x2, y2), (255, 0, 0), 2)



# Afficher l'image
#imshow(binary)

cv2.waitKey(0)
cv2.destroyAllWindows()



