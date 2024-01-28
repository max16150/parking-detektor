import cv2
import numpy as np
import functions as f

initialized = False
SCREEN_1 = 'SCREEN_1'

# Fonction de rappel pour les trackbars
def update_image(*args):
    if not initialized: return
    low_threshold = cv2.getTrackbarPos('Seuil Bas', SCREEN_1)
    high_threshold = cv2.getTrackbarPos('Seuil Haut', SCREEN_1)
    threshold = cv2.getTrackbarPos('Seuil Hough', SCREEN_1)
    min_line_length = cv2.getTrackbarPos('Longueur Min Ligne', SCREEN_1)
    max_line_gap = cv2.getTrackbarPos('Ecart Max Ligne', SCREEN_1)

    edges = cv2.Canny(image_gray, low_threshold, high_threshold)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    image_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    first_image = f.ResizeWithAspectRatio(image.copy(), width=600)
    second_image = f.ResizeWithAspectRatio(image_with_lines.copy(), width=600)
    Hori = np.concatenate((first_image, second_image), axis=1) 
    cv2.imshow(SCREEN_1, Hori)

# Charger l'image et la convertir en gris
image = cv2.imread(r"C:\Users\Maxime\Documents\Github\parking-detektor\res\images\7.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Créer une fenêtre et des trackbars
cv2.namedWindow(SCREEN_1)
cv2.createTrackbar('Seuil Bas', SCREEN_1, 50, 255, update_image)
cv2.createTrackbar('Seuil Haut', SCREEN_1, 150, 255, update_image)
cv2.createTrackbar('Seuil Hough', SCREEN_1, 50, 255, update_image)
cv2.createTrackbar('Longueur Min Ligne', SCREEN_1, 100, 300, update_image)
cv2.createTrackbar('Ecart Max Ligne', SCREEN_1, 10, 100, update_image)

# Initialiser l'image avec les paramètres par défaut
initialized = True
update_image()

# Boucle d'affichage
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
