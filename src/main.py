import cv2
import os

# Obtenir la liste de tous les fichiers d'images dans le dossier res/images
folder_path = r"res\images"
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Parcourir chaque fichier d'image et appliquer le traitement
for image_file in image_files:
    # Charger l'image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Utiliser Canny pour détecter les contours
    edges = cv2.Canny(blurred, 50, 150)

    # Calculer le ratio de redimensionnement
    ratio = 600.0 / edges.shape[1]  # 600 est la nouvelle longueur souhaitée

    # Calculer les nouvelles dimensions avec le ratio de redimensionnement
    dim = (600, int(edges.shape[0] * ratio))

    # Redimensionner l'image
    resized_edges = cv2.resize(edges, dim, interpolation=cv2.INTER_AREA)

    # Afficher l'image avec contours
    cv2.imshow('Contours', resized_edges)
    cv2.waitKey(0)

# Fermer toutes les fenêtres après le traitement de toutes les images
cv2.destroyAllWindows()
