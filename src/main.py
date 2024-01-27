import cv2

# Charger l'image
image = cv2.imread(r"C:\Users\Maxime\Documents\Github\parking-detektor\res\images\1.jpg")

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un flou pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Initial', blurred)

# Utiliser Canny pour détecter les contours
edges = cv2.Canny(blurred, 50, 150)

# Afficher l'image avec contours
cv2.imshow('Contours', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()