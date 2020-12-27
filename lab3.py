import cv2

image = cv2.imread('img.jpg')

cropped = image[10:500, 500:2000]


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


viewImage(cropped, "cropped")

scale_percent = 20  # Процент от изначального размера
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

viewImage(resized, "scale_percent")

(h, w, d) = image.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
viewImage(rotated, "rotated")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshold_image = cv2.threshold(image, 127, 255, 0)
viewImage(gray_image, "gray_image")
viewImage(threshold_image, "threshold_image")

blurred = cv2.GaussianBlur(image, (51, 51), 0)
viewImage(blurred, "blurred")

output = image.copy()
cv2.rectangle(output, (2600, 800), (4100, 2400), (0, 255, 255), 10)
viewImage(output, "output")

cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
viewImage(output, "line")

output2 = image.copy()
cv2.putText(output2, "We <3 Dogs", (1500, 3600),cv2.FONT_HERSHEY_SIMPLEX, 15, (30, 105, 210), 40)
viewImage(output2, "output2")

image_path = 'img.jpg'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image2 = cv2.imread(image_path)
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(10, 10)
)
faces_detected = "Лиц обнаружено: " + format(len(faces))
print(faces_detected)
# Рисуем квадраты вокруг лиц
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
viewImage(image2, faces_detected)
