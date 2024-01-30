import cv2
import numpy as np
import functions as f

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString

ITERATION = ' 2'

initialized = False
SCREEN_IMAGE_SELECTOR_CONTROL = 'Selecteur d\'image' + str(ITERATION)

SCREEN_HSV_CONTROL = 'Controlleur HSV' + str(ITERATION)
SCREEN_HSV = 'HSV' + str(ITERATION)
globalHsvResult = None

SCREEN_OPENING_CONTROL = 'Controlleur Ouverture' + str(ITERATION)
SCREEN_OPENING = 'Ouverture' + str(ITERATION)
globalOpeningResult = None

SCREEN_LINE_DETECTION_CONTROL = 'Controlleur Detection de lignes' + str(ITERATION)
SCREEN_LINE_DETECTION = 'Detection de lignes' + str(ITERATION)
globalLineDetectionResult = None

SCREEN_DENOISE_AND_SORT_LINES_CONTROL = 'Controlleur debruiter et trier les lignes' + str(ITERATION)
SCREEN_DENOISE_AND_SORT_LINES = 'Debruiter et trier les lignes' + str(ITERATION)
globalDenoiseAndSortLinesResult = None

SCREEN_FIND_INTERSECTIONS_CONTROL = 'Controlleur intersections' + str(ITERATION)
SCREEN_FIND_INTERSECTIONS = 'Intersections' + str(ITERATION)
globalFindIntersectionsResult = None

SCREEN_ISOLATE_PLACES_CONTROL = 'Controlleur isoler places' + str(ITERATION)
SCREEN_ISOLATE_PLACES = 'Isoler places' + str(ITERATION)
globalIsolatePlacesResult = None

SCREEN_FIND_POLYGONS_CONTROL = 'Controlleur trouver polygones' + str(ITERATION)
SCREEN_FIND_POLYGONS = 'Trouver polygones' + str(ITERATION)
globalFindPolygonsResult = None

SCREEN_FIND_CARS_CONTROL = 'Controlleur trouver voitures' + str(ITERATION)
SCREEN_FIND_CARS = 'Trouver voitures' + str(ITERATION)
globalFindCarsResult = None

SCREEN_DEFINE_FREE_PLACES_CONTROL = 'Controlleur definir places libres' + str(ITERATION)
SCREEN_DEFINE_FREE_PLACES = 'Definir places libres' + str(ITERATION)
globalDefineFreePlacesResult = None

images = [
    "1.jpg",
    "2.png",
    "3.png",
    "4.png",
    "41.png",
    "6.png",
    "7.png",
    "8.png",
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
    find_cars()

def init_image_selectors_control():
    cv2.namedWindow(SCREEN_IMAGE_SELECTOR_CONTROL)
    cv2.createTrackbar('Image', SCREEN_IMAGE_SELECTOR_CONTROL, 3, len(images)-1, image_selectors)

def init_hsv_control():
    cv2.namedWindow(SCREEN_HSV_CONTROL)
    cv2.namedWindow(SCREEN_HSV)
    cv2.createTrackbar('Hue Bas', SCREEN_HSV_CONTROL, 0, 255, update_hsv)
    cv2.createTrackbar('Saturation Bas', SCREEN_HSV_CONTROL, 0, 255, update_hsv)
    cv2.createTrackbar('Value Bas', SCREEN_HSV_CONTROL, 180, 255, update_hsv)
    cv2.createTrackbar('Hue Haut', SCREEN_HSV_CONTROL, 200, 255, update_hsv)
    cv2.createTrackbar('Saturation Haut', SCREEN_HSV_CONTROL, 25, 255, update_hsv)
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
    _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)

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
    line_detection()

def init_line_detection_control():
    cv2.namedWindow(SCREEN_LINE_DETECTION_CONTROL)
    cv2.namedWindow(SCREEN_LINE_DETECTION)
    cv2.createTrackbar('Ouverture Canny', SCREEN_LINE_DETECTION_CONTROL, 0, 2, line_detection)
    cv2.createTrackbar('Seuil Hough', SCREEN_LINE_DETECTION_CONTROL, 57, 255, line_detection)
    cv2.createTrackbar('Longueur Min Ligne', SCREEN_LINE_DETECTION_CONTROL, 104, 300, line_detection)
    cv2.createTrackbar('Ecart Max Ligne', SCREEN_LINE_DETECTION_CONTROL, 61, 100, line_detection)

def line_detection(*args):
    global globalOpeningResult
    global globalLineDetectionResult
    if not initialized: return

    openingResult = globalOpeningResult.copy()
    apertureSize =  [3, 5, 7][cv2.getTrackbarPos('Ouverture Canny', SCREEN_LINE_DETECTION_CONTROL)]
    threshold = cv2.getTrackbarPos('Seuil Hough', SCREEN_LINE_DETECTION_CONTROL)
    min_line_length = cv2.getTrackbarPos('Longueur Min Ligne', SCREEN_LINE_DETECTION_CONTROL)
    max_line_gap = cv2.getTrackbarPos('Ecart Max Ligne', SCREEN_LINE_DETECTION_CONTROL)

    edges = cv2.Canny(openingResult, 100, 200, apertureSize=apertureSize)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    lines = lines if lines is not None else []
    globalLineDetectionResult = lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(openingResult, (x1, y1), (x2, y2), (0, 255, 0), 2)

    Hori = np.concatenate((globalOpeningResult, openingResult), axis=1) 
    cv2.imshow(SCREEN_LINE_DETECTION, Hori)
    denoise_and_sort_lines()

def init_denoise_and_sort_lines():
    cv2.namedWindow(SCREEN_DENOISE_AND_SORT_LINES_CONTROL)
    cv2.namedWindow(SCREEN_DENOISE_AND_SORT_LINES)
    cv2.createTrackbar('Min batch size', SCREEN_DENOISE_AND_SORT_LINES_CONTROL, 1, 6, denoise_and_sort_lines)
    cv2.createTrackbar('Angle tolerence', SCREEN_DENOISE_AND_SORT_LINES_CONTROL, 100, 100, denoise_and_sort_lines)

def denoise_and_sort_lines(*args):
    global globalLineDetectionResult
    global globalDenoiseAndSortLinesResult
    global image
    if not initialized: return
    lines = globalLineDetectionResult.copy()
    imageCopy = image.copy()
    distinctOrientations = cv2.getTrackbarPos('Min batch size', SCREEN_DENOISE_AND_SORT_LINES_CONTROL)
    angleTolerence = cv2.getTrackbarPos('Angle tolerence', SCREEN_DENOISE_AND_SORT_LINES_CONTROL)

    # Calculer les angles des lignes
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        angles.append([angle])

    # Appliquer DBSCAN
    dbscan = DBSCAN(eps=float(angleTolerence/100+0.001), min_samples=distinctOrientations+1)
    clusters = [] if len(angles) == 0 else dbscan.fit_predict(angles)
    cluster_colors = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255]
    ]

    filteredLines = []

    # Processus de post-traitement pour chaque cluster
    for cluster_id in set(clusters):
        cluster_color = cluster_colors[cluster_id % len(cluster_colors)]
        if cluster_id != -1:  # -1 est le label pour les bruits
            cluster_lines = [lines[i] for i, label in enumerate(clusters) if label == cluster_id]
            filteredLines.append(cluster_lines)
            # Afficher les lignes avec la même couleur
            for value in cluster_lines:
                x1, y1, x2, y2 = value[0]
                cv2.line(imageCopy, (x1, y1), (x2, y2), cluster_color, 2)
    
    globalDenoiseAndSortLinesResult = filteredLines
    cv2.imshow(SCREEN_DENOISE_AND_SORT_LINES, imageCopy)
    #find_intersections()
    isolate_places()

def init_find_intersections_control():
    cv2.namedWindow(SCREEN_FIND_INTERSECTIONS_CONTROL)
    cv2.namedWindow(SCREEN_FIND_INTERSECTIONS)

def find_intersections(*args):
    global globalDenoiseAndSortLinesResult
    global globalFindIntersectionsResult
    if not initialized: return
    clusters = globalDenoiseAndSortLinesResult.copy()
    imageCopy = image.copy()

    if(len(clusters) < 2):
        globalFindIntersectionsResult = []
        cv2.imshow(SCREEN_FIND_INTERSECTIONS, imageCopy)
        return

    intersections = []
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            for line1 in clusters[i]:
                for line2 in clusters[j]:
                    x1, y1, x2, y2 = line1[0]
                    x3, y3, x4, y4 = line2[0]

                    ligne1 = LineString([(x1, y1), (x2, y2)])
                    ligne2 = LineString([(x3, y3), (x4, y4)])

                    # Trouver l'intersection
                    point_intersection = ligne1.intersection(ligne2)

                    # Vérifier si il y a une intersection et la stocker
                    if not point_intersection.is_empty:
                        intersections.append(point_intersection)

    # Afficher les intersections
    for point in intersections:
        cv2.circle(imageCopy, (int(point.x), int(point.y)), 4, (0, 0, 255), -1)

    cv2.imshow(SCREEN_FIND_INTERSECTIONS, imageCopy)
    globalFindIntersectionsResult = intersections

def init_isolate_places_control():
    cv2.namedWindow(SCREEN_ISOLATE_PLACES_CONTROL)
    cv2.namedWindow(SCREEN_ISOLATE_PLACES)
    cv2.createTrackbar('Largeur segments', SCREEN_ISOLATE_PLACES_CONTROL, 73, 100, isolate_places)
    cv2.createTrackbar('Erosion', SCREEN_ISOLATE_PLACES_CONTROL, 20, 20, isolate_places)
    cv2.createTrackbar('Dilatation', SCREEN_ISOLATE_PLACES_CONTROL, 20, 20, isolate_places)

def isolate_places(*args):
    global globalDenoiseAndSortLinesResult
    global globalIsolatePlacesResult
    if not initialized: return

    blackImage = np.zeros(image.shape, np.uint8)

    largeurSegments = cv2.getTrackbarPos('Largeur segments', SCREEN_ISOLATE_PLACES_CONTROL)
    erosionIterations = cv2.getTrackbarPos('Erosion', SCREEN_ISOLATE_PLACES_CONTROL)
    dilationIterations = cv2.getTrackbarPos('Dilatation', SCREEN_ISOLATE_PLACES_CONTROL)

    
    clusters = globalDenoiseAndSortLinesResult.copy()
    for cluster in clusters:
        for line in cluster:
            f.dessiner_rectangle_centre(blackImage, line[0], largeurSegments, (255, 255, 255))
    
    for cluster in clusters:
        for line in cluster:
            x1, y1, x2, y2 = line[0]
            cv2.line(blackImage, (x1, y1), (x2, y2), (0, 0, 0), 5)

    # Apply erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(blackImage, kernel, iterations=erosionIterations)

    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(erosion, kernel, iterations=dilationIterations)

    # Negate images
    dilation = cv2.bitwise_not(dilation)
    globalIsolatePlacesResult = dilation

    # Apply AND operation to original image
    dilation = cv2.bitwise_and(dilation, image)
    
    Hori = np.concatenate((blackImage, erosion, dilation), axis=1)
    cv2.imshow(SCREEN_ISOLATE_PLACES, Hori)
    find_polygons()


def init_find_polygons_control():
    cv2.namedWindow(SCREEN_FIND_POLYGONS_CONTROL)
    cv2.namedWindow(SCREEN_FIND_POLYGONS)
    cv2.createTrackbar('eps', SCREEN_FIND_POLYGONS_CONTROL, 500, 1000, find_polygons)
    cv2.createTrackbar('min_samples', SCREEN_FIND_POLYGONS_CONTROL, 4, 20, find_polygons)

def find_polygons(*args):
    global globalIsolatePlacesResult
    global globalFindPolygonsResult
    if not initialized: return

    polygons = []
    eps = cv2.getTrackbarPos('eps', SCREEN_FIND_POLYGONS_CONTROL)
    min_samples = cv2.getTrackbarPos('min_samples', SCREEN_FIND_POLYGONS_CONTROL)
    outputImage = image.copy()

    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(globalIsolatePlacesResult, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuillage binaire si nécessaire
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Trouver les contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculer la surface de chaque polygone
    surfaces = np.array([cv2.contourArea(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) for contour in contours]).reshape(-1, 1)

    # Appliquer DBSCAN pour identifier les outliers
    dbscan = DBSCAN(eps=float(eps+0.001), min_samples=min_samples+1)  # Ajustez 'eps' et 'min_samples' au besoin
    clusters = dbscan.fit_predict(surfaces)

    # Filtrer les polygones (garder ceux qui ne sont pas des outliers)
    sortedPolygons = [contours[i] for i in range(len(contours)) if clusters[i] != -1]

    # Dessiner les polygones filtrés sur l'image
    for polygone in sortedPolygons:
        cv2.drawContours(outputImage, [polygone], 0, (255, 0, 0), 2)
    
    globalFindPolygonsResult = sortedPolygons
    
    cv2.imshow(SCREEN_FIND_POLYGONS, outputImage)
    define_free_places()

def init_find_cars_control():
    cv2.namedWindow(SCREEN_FIND_CARS_CONTROL)
    cv2.namedWindow(SCREEN_FIND_CARS)
    #cv2.createTrackbar('Hue Bas', SCREEN_FIND_CARS_CONTROL, 0, 255, find_cars)
    #cv2.createTrackbar('Saturation Bas', SCREEN_FIND_CARS_CONTROL, 0, 255, find_cars)
    cv2.createTrackbar('Value Bas', SCREEN_FIND_CARS_CONTROL, 114, 255, find_cars)
    cv2.createTrackbar('Hue Haut', SCREEN_FIND_CARS_CONTROL, 71, 255, find_cars)
    cv2.createTrackbar('Saturation Haut', SCREEN_FIND_CARS_CONTROL, 85, 255, find_cars)
    cv2.createTrackbar('Value Haut', SCREEN_FIND_CARS_CONTROL, 196, 255, find_cars)
    cv2.createTrackbar('Erosion', SCREEN_FIND_CARS_CONTROL, 3, 20, find_cars)
    cv2.createTrackbar('Dilatation', SCREEN_FIND_CARS_CONTROL, 7, 20, find_cars)

def find_cars(*args):
    global globalFindPolygonsResult
    global globalFindCarsResult
    if not initialized: return
    
    lower_hue = 0 # cv2.getTrackbarPos('Hue Bas', SCREEN_FIND_CARS_CONTROL)
    lower_saturation = 0 # cv2.getTrackbarPos('Saturation Bas', SCREEN_FIND_CARS_CONTROL)
    lower_value = cv2.getTrackbarPos('Value Bas', SCREEN_FIND_CARS_CONTROL)
    upper_hue = cv2.getTrackbarPos('Hue Haut', SCREEN_FIND_CARS_CONTROL)
    upper_saturation = cv2.getTrackbarPos('Saturation Haut', SCREEN_FIND_CARS_CONTROL)
    upper_value = cv2.getTrackbarPos('Value Haut', SCREEN_FIND_CARS_CONTROL)

    srcHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([lower_hue, lower_saturation, lower_value])
    upper = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(srcHsv, lower, upper)
    binary = cv2.bitwise_and(image, image, mask=mask)
    _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)

    
    # Negate binary image
    binary = cv2.bitwise_not(binary)

    # Apply erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosionIterations = cv2.getTrackbarPos('Erosion', SCREEN_FIND_CARS_CONTROL)
    erosion = cv2.erode(binary, kernel, iterations=erosionIterations)

    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilationIterations = cv2.getTrackbarPos('Dilatation', SCREEN_FIND_CARS_CONTROL)
    dilation = cv2.dilate(erosion, kernel, iterations=dilationIterations)

    Hori = np.concatenate((binary, erosion, dilation), axis=1)
    cv2.imshow(SCREEN_FIND_CARS, Hori)

    globalFindCarsResult = dilation
    define_free_places()

def init_define_free_places_control():
    cv2.namedWindow(SCREEN_DEFINE_FREE_PLACES_CONTROL)
    cv2.namedWindow(SCREEN_DEFINE_FREE_PLACES)

def define_free_places(*args):
    global globalFindCarsResult
    global globalFindPolygonsResult
    global globalDefineFreePlacesResult
    global image
    if not initialized: return

    if globalFindCarsResult is None or globalFindPolygonsResult is None: 
        cv2.imshow(SCREEN_DEFINE_FREE_PLACES, image)
        return

    # Pour chaque polygone de place dans globalFindPolygonsResult, trouver le nombre de pixels blancs dans globalFindCarsResult
    # Si le nombre de pixels blancs est inférieur à un certain seuil, alors la place est libre
    # Sinon, la place est occupée

    places = globalFindPolygonsResult.copy()
    cars = globalFindCarsResult.copy()
    outputImage = image.copy()

    freePlaces = 0
    occupiedPlaces = 0

    for place in places:
        # Trouver le nombre de pixels blancs dans la place binaire
        mask = np.zeros(cars.shape[:2], np.uint8)
        cv2.fillPoly(mask, [place], 255)
        maskedCars = cv2.bitwise_and(cars, cars, mask=mask)
        maskedCars = cv2.cvtColor(maskedCars, cv2.COLOR_BGR2GRAY)

        whitePixels = cv2.countNonZero(maskedCars)

        if whitePixels < 100:
            # La place est libre
            freePlaces += 1
            cv2.drawContours(outputImage, [place], 0, (0, 255, 0), 2)
        else:
            # La place est occupée
            occupiedPlaces += 1
            cv2.drawContours(outputImage, [place], 0, (0, 0, 255), 2)

    # Ecrire le nombre de places libres sur l'image
    cv2.putText(outputImage, 'Places libres : ' + str(freePlaces), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Ecrire le nombre de places occupées sur l'image
    cv2.putText(outputImage, 'Places occupees : ' + str(occupiedPlaces), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.imshow(SCREEN_DEFINE_FREE_PLACES, outputImage)


def first_run():
    init_image_selectors_control()
    init_hsv_control()
    init_opening_control()
    init_line_detection_control()
    init_denoise_and_sort_lines()
    #init_find_intersections_control()
    init_isolate_places_control()
    init_find_polygons_control()
    init_find_cars_control()
    init_define_free_places_control()
    global initialized
    initialized = True
    image_selectors()
    find_cars()


first_run()

# Boucle d'affichage
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()