import numpy as np
import cv2
import sys

import img_lib

img_lib.printar()

print("python version", sys.version)
print("opencv version", cv2.__version__)

# (B,G,R) → 0-255
rgb_white = (250, 250, 250)
rgb_blue = (250, 0, 0)
rgb_green = (0, 255, 0)
rgb_red = (0, 0, 255)
rgb_black = (0, 0, 0)

# ---------- Running Routines ----------
"""
img = cv2.imread('images/aqui.jpg')
# cv2.imshow("Original Image", img)

# Images Routines
img = img[:, ::-1, :]  # Horizontal Mirror
img = img[::-1, :, :]  # Vertical Mirror
img = img[:, :, ::-1]  # RGB Mirror [BGR]→[RGB]

# ----- RGB FOREPLAY -----
# Convert to Grey
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey Image", img_grey)

# Convert to HSV → Hue; Saturation ; Valor/Luminosity [0-180][0-255][0-255]
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Splits the image in H,S,V Channels AND Merges it back together
h, s, v = cv2.split(img_hsv)
cv2.imshow("Hue", h)
cv2.imshow("Saturation", s)
cv2.imshow("Luminosity", v)
img_hsv_merged = cv2.merge((h, s, v))     # Merges image from BGR
cv2.imshow("Merged HSV Image", img_hsv_merged)

# Splits the image in R,G,B Channels AND Merges it back together
b, g, r = cv2.split(img)    # Splits image in BGR
# OR
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
#
cv2.imshow("Blue", b)       # Grey representation of Blue Component
cv2.imshow("Green", g)      # Grey representation of Green Component
cv2.imshow("Red", r)        # Grey representation of Red Component
img_bgr_merged = cv2.merge((b, g, r))     # Merges image from BGR
cv2.imshow("Merged RGB Image", img_bgr_merged)

# Shows only the Selected BGR Channel 
# Keep Blue Pixels
img_b = img.copy()
img_b[:, :, 1:3] = 0
cv2.imshow("Blue Image", img_b)
# Keep Green Pixels
img_g = img.copy()
img_g[:, :, (0, 2)] = 0
cv2.imshow("Green Image", img_g)
# Keep Red Pixels
img_r = img.copy()
img_r[:, :, 0:2] = 0
cv2.imshow("Red Image", img_r)

"""
# Video Camera Routine
cap = cv2.VideoCapture(0)
_, fr = cap.read()
print("----------% FRAME %----------")
print("Dimensões (px): " + str(fr.shape))
print("Tamanho(bytes): " + str(fr.size))
print("Tamanho   (kB): " + str(fr.size / 1024))
print("Tipo de Dados : " + str(fr.dtype))


# TRACKBAR CREATION
img_hue_map = cv2.imread("images/hsv.png")


def hue_map(x):
    hue_map_copy = img_hue_map.copy()
    cv2.rectangle(hue_map_copy, (int(hue_min*4.2666), 0), (int(hue_max*4.2666), 100), rgb_black, 2)
    cv2.imshow("HSV Filter Calibration", hue_map_copy)


# Grey TrackBar
"""
cv2.namedWindow("Grey Filter Calibration")
cv2.createTrackbar("Min", "Grey Filter Calibration", 0, 255, hue_map)
cv2.createTrackbar("Max", "Grey Filter Calibration", 0, 255, hue_map)
"""
# HSV TrackBar
cv2.namedWindow("HSV Filter Calibration")
cv2.createTrackbar("HUE Min", "HSV Filter Calibration", 0, 180, hue_map)
cv2.createTrackbar("HUE Max", "HSV Filter Calibration", 0, 180, hue_map)

# Video
while cv2.waitKey(1) - 27:
    _, fr = cap.read()
    cv2.imshow('Original frame', fr)
    # ----- RGB  Video FOREPLAY -----
    fr_hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV frame", fr_hsv)

    hue_min = cv2.getTrackbarPos("HUE Min", "HSV Filter Calibration")
    hue_max = cv2.getTrackbarPos("HUE Max", "HSV Filter Calibration")

    hsv_min = np.array([hue_min, 0, 0])
    hsv_max = np.array([hue_max, 255, 255])
    fr_hsv_binary = cv2.inRange(fr_hsv, hsv_min, hsv_max)
    # cv2.imshow("HSV Filtered Frame", fr_hsv_filtered)

    # ----- Erosion -----
    kernel = np.ones((5, 5), np.uint8)
    fr_binary_eroded = cv2.erode(fr_hsv_binary, kernel)
    cv2.imshow("HSV Filtered & Eroded Frame", fr_binary_eroded)

    # -----Contours -----
    # TEST HUE [84-94]
    contours, _ = cv2.findContours(fr_binary_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if cv2.waitKey(2)-ord('c') == 0:
        print("Nº of Contours", len(contours))

    for contour in contours:
        area = cv2.contourArea(contour)

        cv2.drawContours(fr, contour, -1, rgb_black, 2)
        if area > 5000:
            cv2.drawContours(fr, contour, -1, rgb_green, 3)
        cv2.imshow('Contours frame', fr)
    # TARGET
    # moment = cv2.moments(contour)

    # ----- Grey Video FOREPLAY -----
    """
    fr_grey = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grey frame', fr_grey)
    
    grey_min = cv2.getTrackbarPos("Min", "Grey Filter Calibration")
    grey_max = cv2.getTrackbarPos("Max", "Grey Filter Calibration")
    fr_grey_filtered = cv2.inRange(fr_grey, grey_min, grey_max)  # Between Min & Max turns White Else turns Black
    cv2.imshow("Filtered Frame", fr_grey_filtered)
    
    fr_grey_limit = fr_grey.copy()
    fr_grey_limit[fr_grey_limit < grey_min] = 0    # Below Min turns Black
    fr_grey_limit[fr_grey_limit > grey_max] = 255  # Above Max turns White
    cv2.imshow("Filtered Grey Limit Frame", fr_grey_limit)
    """

# Closing Routine
# cv2.waitKey(0)         # 0 → Keep Windows Open Untill Key is Pressed AND Returns ASCII from pressed key
cv2.waitKey(1)  # Keep Windows Open (1→1ms)
cv2.destroyAllWindows()
