import numpy as np
import cv2

def angle_cos(p0, p1, p2):
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
	squares = []
	biggest = None
	sizes = []
	for gray in cv2.split(img):
		for thrs in range(0, 255, 26):
			if thrs == 0:
				bin = cv2.Canny(gray, 0, 15, apertureSize=5)
			else:
				_retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
			contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
				if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
					sizes.append(cv2.contourArea(cnt))
			avg = sum(sizes) / len(sizes)
			median = np.median(sizes)
			for cnt in contours:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
				if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
					cnt = cnt.reshape(-1, 2)
					max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
					if max_cos < 0.1:
						squares.append(cnt)
						if isinstance(biggest, (list, tuple, np.ndarray)):
							if cv2.contourArea(cnt) > cv2.contourArea(biggest):
								biggest = cnt
						else:
							biggest = cnt
	return squares, biggest

img = cv2.imread('test.jpg')
img = cv2.flip(img, 0)
img = cv2.resize(img, (1100,940) )
squares, biggest = find_squares(img)
cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3 )

# cropped = image[startY:endY, startX:endX]

# mask = np.zeros(img.shape,np.uint8)
# new_image = cv2.drawContours(mask,[biggest],0,255,-1,)
# new_image = cv2.bitwise_and(cropped,cropped,mask=mask)

cv2.imshow('squares', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Done')