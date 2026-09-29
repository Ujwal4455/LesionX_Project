import numpy as np
import cv2


def extract_morphology_features(image_path):
    if not image_path:
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0.0, "perimeter": 0.0, "eccentricity": 0.0, "solidity": 0.0}

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        eccentricity = 0.0
    else:
        mu20 = moments["m20"] / moments["m00"]
        mu02 = moments["m02"] / moments["m00"]
        mu11 = moments["m11"] / moments["m00"]
        numerator = (mu20 - mu02) ** 2 + 4 * mu11 ** 2
        denominator = (mu20 + mu02) ** 2
        eccentricity = float(np.sqrt(numerator / max(denominator, 1e-8)))

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1e-8)

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "eccentricity": float(eccentricity),
        "solidity": float(solidity),
    }
