import cv2
import numpy as np

# Charger l'image
image = cv2.imread(r"C:\Users\Maxime\Documents\Github\parking-detektor\res\images\7.png")

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre pour réduire le bruit
filtered = cv2.GaussianBlur(gray, (5, 5), 0)

# Détecter les contours
edges = cv2.Canny(filtered, 50, 150)

# Trouver les lignes avec la transformation de Hough
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

# Assombrir l'image
image = cv2.convertScaleAbs(image, alpha=0.4, beta=0)

# Dessiner les lignes sur l'image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


# Redimensionner l'image
image = cv2.resize(image, (1080, 720), interpolation=cv2.INTER_CUBIC)

# Afficher l'image
cv2.imshow('Parking Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
