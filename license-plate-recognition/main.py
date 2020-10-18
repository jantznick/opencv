import numpy as np
import argparse
import cv2
import imutils
# import re

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="sample-images/4.PNG",
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
img = cv2.resize(img, (480,620) )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale

gray = cv2.bilateralFilter(gray, 13, 50, 50)
edged = cv2.Canny(gray, 10, 150) #Perform Edge detection

contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.018 * peri, True)
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

img_contours = np.zeros(img.shape)
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

img_contour = np.zeros(img.shape)
if screenCnt.any():
	cv2.drawContours(img_contour, [screenCnt], -1, (0,255,0), 3)

cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.imshow('contours', img_contours)
# cv2.waitKey(0)
if screenCnt.any():
	cv2.imshow('contour', img_contour)
# cv2.waitKey(0)
# cv2.imshow('edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()