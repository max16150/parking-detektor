import cv2
import numpy as np
import functions as f

initialized = False
SCREEN_IMAGE_SELECTOR_CONTROL = 'selector'

SCREEN_HSV_CONTROL = 'hsv_controler'
SCREEN_HSV = 'hsv'
globalHsvResult = None

SCREEN_OPENING_CONTROL = 'opening_controler'
SCREEN_OPENING = 'opening'
globalOpeningResult = None


images = [
    "1.jpg",
    "2.png",
    "3.png",
    "4.png",
    "5.png",
    "6.png",
    "7.png",
]

# Charger l'image
image = cv2.imread(r"C:\Users\Maxime\Documents\Github\parking-detektor\res\images\5.png")

# Redimensionner l'image
image = f.ResizeWithAspectRatio(image, width=600)

# Convertir l'image en HSV
imageHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def image_selectors(*args):
    global image
    global imageHsv
    if not initialized: return
    imageNumber = cv2.getTrackbarPos('Image', SCREEN_IMAGE_SELECTOR_CONTROL)
    image = cv2.imread(r"C:\Users\Maxime\Documents\Github\parking-detektor\res\images\\" + images[imageNumber])
    image = f.ResizeWithAspectRatio(image, width=600)
    imageHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    update_hsv()

def image_selectors_control():
    cv2.namedWindow(SCREEN_IMAGE_SELECTOR_CONTROL)
    cv2.createTrackbar('Image', SCREEN_IMAGE_SELECTOR_CONTROL, 0, len(images)-1, image_selectors)

def init_hsv_control():
    cv2.namedWindow(SCREEN_HSV_CONTROL)
    cv2.namedWindow(SCREEN_HSV)
    cv2.createTrackbar('Hue Bas', SCREEN_HSV_CONTROL, 0, 255, update_hsv)
    cv2.createTrackbar('Saturation Bas', SCREEN_HSV_CONTROL, 0, 255, update_hsv)
    cv2.createTrackbar('Value Bas', SCREEN_HSV_CONTROL, 185, 255, update_hsv)
    cv2.createTrackbar('Hue Haut', SCREEN_HSV_CONTROL, 255, 255, update_hsv)
    cv2.createTrackbar('Saturation Haut', SCREEN_HSV_CONTROL, 75, 255, update_hsv)
    cv2.createTrackbar('Value Haut', SCREEN_HSV_CONTROL, 255, 255, update_hsv)

def update_hsv(*args):
    global globalHsvResult
    if not initialized: return
    lower_hue = cv2.getTrackbarPos('Hue Bas', SCREEN_HSV_CONTROL)
    lower_saturation = cv2.getTrackbarPos('Saturation Bas', SCREEN_HSV_CONTROL)
    lower_value = cv2.getTrackbarPos('Value Bas', SCREEN_HSV_CONTROL)
    upper_hue = cv2.getTrackbarPos('Hue Haut', SCREEN_HSV_CONTROL)
    upper_saturation = cv2.getTrackbarPos('Saturation Haut', SCREEN_HSV_CONTROL)
    upper_value = cv2.getTrackbarPos('Value Haut', SCREEN_HSV_CONTROL)

    lower = np.array([lower_hue, lower_saturation, lower_value])
    upper = np.array([upper_hue, upper_saturation, upper_value])

    mask = cv2.inRange(imageHsv, lower, upper)
    binary = cv2.bitwise_and(image, image, mask=mask)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    Hori = np.concatenate((image, binary), axis=1) 
    cv2.imshow(SCREEN_HSV, Hori)
    globalHsvResult = binary
    update_opening()

def init_opening_control():
    cv2.namedWindow(SCREEN_OPENING_CONTROL)
    cv2.namedWindow(SCREEN_OPENING)
    cv2.createTrackbar('Erosion iterations', SCREEN_OPENING_CONTROL, 3, 20, update_opening)
    cv2.createTrackbar('Dilatation iterations', SCREEN_OPENING_CONTROL, 10, 20, update_opening)

def update_opening(*args):
    global globalHsvResult
    global globalOpeningResult
    if not initialized: return
    duplicatedHsvBinaryImage = globalHsvResult.copy()
    # Apply erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosionIterations = cv2.getTrackbarPos('Erosion iterations', SCREEN_OPENING_CONTROL)
    erosion = cv2.erode(duplicatedHsvBinaryImage, kernel, iterations=erosionIterations)
    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilationIterations = cv2.getTrackbarPos('Dilatation iterations', SCREEN_OPENING_CONTROL)
    dilation = cv2.dilate(erosion, kernel, iterations=dilationIterations)
    # Negate images
    dilationInverted = cv2.bitwise_not(dilation)
    # Apply AND operation
    andOperation = cv2.bitwise_and(dilationInverted, duplicatedHsvBinaryImage)
    Hori = np.concatenate((dilation, andOperation), axis=1)
    cv2.imshow(SCREEN_OPENING, Hori)
    globalOpeningResult = andOperation

def first_run():
    image_selectors_control()
    init_hsv_control()
    init_opening_control()
    global initialized
    initialized = True
    image_selectors()


first_run()

# Boucle d'affichage
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()