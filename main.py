# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np

'''
# for image display

print("Package Imported")
img = cv2.imread("Resources/swans.png")
cv2.imshow("Output",img)
cv2.waitKey(0)
'''

''' 
# for video display

cap = cv2.VideoCapture("Resources/ava_max_salt.mp4")
while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

''' 
# for webcam video display

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width id->3
cap.set(4, 480)  # height id->4
cap.set(10, 100)  # brightness id->10
while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

'''
img = cv2.imread("Resources/swans.png")
kernel = np.ones((5, 5), np.uint8)  # size->(5,5).type of object->un signed integer of 8 bit i.e. values from 0 to 255

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)  # ksize or kernel size should be positive and odd

# Find edges in image
imgCanny = cv2.Canny(img, 150, 200)

# Increase thickness of edge
imgDilate = cv2.dilate(imgCanny, kernel, iterations=1)

# Decrease thickness of edge
imgErode = cv2.erode(imgDilate, kernel, iterations=1)

cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny image", imgCanny)
cv2.imshow("Dilate image", imgDilate)
cv2.imshow("Erode image", imgErode)
cv2.waitKey(0)
'''

'''
# Resizing and cropping

img = cv2.imread("Resources/cheetah.png")
print(img.shape)
img_resize = cv2.resize(img, (854, 480))  # width first then height
print(img_resize.shape)
imgCrop = img[0:200, 200:500]  # height first then width
cv2.imshow("Image", img)
cv2.imshow("Image resize", img_resize)
cv2.imshow("Image crop", imgCrop)
cv2.waitKey(0)
'''

'''
# Shapes and texts

img = np.zeros((512, 512, 3), np.uint8)
# print(img.shape)
# print(img)
# img[:] = 255, 0, 0
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 3)
cv2.rectangle(img, (0, 0), (300, 350), (0, 255, 0), 2)
cv2.circle(img, (300, 300), 70, (255, 0, 0), 5)
cv2.putText(img, "OpenCV", (380, 370), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
'''

'''
# Warp perspective
img = cv2.imread("Resources/book.jpg")
width, height = 550, 365
pts1 = np.float32([[96, 100], [385, 66], [480, 218], [128, 275]])
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("Image", img)
winName = "Output"
cv2.namedWindow(winName)  # Create a named window
cv2.moveWindow(winName, 800, 0)  # Move it to (40,30)
cv2.imshow(winName, imgOutput)
# cv2.waitKey(0)
cv2.waitKey()
cv2.destroyAllWindows()  # The function destroyAllWindows destroys all of the opened HighGUI windows.
'''

'''
# Joining images
img = cv2.imread("Resources/book.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# imgHor = np.hstack((img, img))
# imgVer = np.vstack((img, img))
# cv2.imshow("Horizontal", imgHor)
# cv2.imshow("Vertical", imgVer)

# Stack image function
def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


imgStack = stack_images(0.5, ([img, imgGray, img], [imgGray, img, imgGray]))
cv2.imshow("Horizontal", imgStack)
cv2.waitKey(0)
'''

'''
# Color detection
path = 'Resources/book.jpg'
winName = "TrackBars"
cv2.namedWindow(winName)
cv2.resizeWindow(winName, 640, 320)


def empty(arg):
    pass


def stack_images(scale, img_array):  # Stack image function
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


cv2.createTrackbar("Hue Min", winName, 20, 179, empty)
cv2.createTrackbar("Hue Max", winName, 169, 179, empty)
cv2.createTrackbar("Sat Min", winName, 0, 255, empty)
cv2.createTrackbar("Sat Max", winName, 255, 255, empty)
cv2.createTrackbar("Value Min", winName, 30, 255, empty)
cv2.createTrackbar("Value Max", winName, 224, 255, empty)
while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", winName)
    h_max = cv2.getTrackbarPos("Hue Max", winName)
    s_min = cv2.getTrackbarPos("Sat Min", winName)
    s_max = cv2.getTrackbarPos("Sat Max", winName)
    v_min = cv2.getTrackbarPos("Value Min", winName)
    v_max = cv2.getTrackbarPos("Value Max", winName)
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Original", img)
    # cv2.imshow("HSV", imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", imgResult)

    imgStack = stack_images(0.6, ([img, imgHSV], [mask, imgResult]))
    cv2.imshow("Stacked Images", imgStack)
    cv2.waitKey(1)
'''

'''
# Shape detection/Contours
path = 'Resources/shapes2.jpg'
img = cv2.imread(path)


# Stack image function
def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def get_contours(img_resize):
    contours, hierarchy = cv2.findContours(img_resize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, contour, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(contour, True)
            print(perimeter)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            print(len(approx))
            obj_corner = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if obj_corner == 3:
                object_type = "Triangle"
            elif obj_corner == 4:
                aspect_ratio = w / float(h)
                if 0.95 < aspect_ratio < 1.05:
                    object_type = "Square"
                else:
                    object_type = "Rectangle"
            elif obj_corner > 4:
                object_type = "Circle"
            else:
                object_type = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, object_type, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 255), 2)


# cv2.imshow("Original", img)
imgResize = cv2.resize(img, (800, 800))  # width first then height
imgContour = imgResize.copy()
# cv2.imshow("Resize", img_resize)
imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grayscale", imgGray)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
# cv2.imshow("Blur", imgBlur)
imgCanny = cv2.Canny(imgBlur, 50, 50)
get_contours(imgCanny)
imgBlank = np.zeros_like(imgResize)
imgStack = stack_images(0.8, ([imgResize, imgGray, imgBlur], [imgCanny, imgContour, imgBlank]))
cv2.imshow("Stack", imgStack)
cv2.waitKey(0)
'''

# Face detection
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread('Resources/person1.jpg')
imgResize = cv2.resize(img, (600, 330))  # width first then height
imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(imgResize, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("Result", imgResize)
cv2.waitKey(0)
