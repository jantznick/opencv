import cv2
import pytesseract
from pytesseract import Output

# Read the image
img = cv2.imread('images/two.jpeg', 0)

# Resize image
(h, w) = img.shape[:2]
img = cv2.resize(img, (int(w * 0.25), int(h * 0.25)))
print('image size:')
print(f"{h},{w}")
# Simple thresholding
# ret, img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
extracted_text = pytesseract.image_to_string(img)
n_boxes = len(d['level'])
for i in range(n_boxes):
	(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	text = d['text'][i]
	if text != '':
		print(f"text: {text}")
		print(f"x: {x}, y: {y}, w: {w}, h: {h}")
	img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(extracted_text)

cv2.imshow('gray', img)
cv2.waitKey(0)