import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

def update_image(value):
    global img, ori
    img = copy.deepcopy(ori)
    img[:,:] += (value / 100.0) - 0.5
    cv2.imshow("Picture", img)
    update_histogram()
img1 = cv2.imread("itsSurabaya2.jpg")
     
img = cv2.imread("itsSurabaya2.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)
img[:, :] /= 255
ori = copy.deepcopy(img)

def update_histogram():
    global img, data
    img_clipped = np.clip(img, 0, 1)
    data = np.histogram(img_clipped, bins=np.linspace(0, 1, 256))
    plt.clf()
    plt.bar(data[1][:-1], data[0], width=0.005)
    plt.xlabel("Pixel")
    plt.xlim(0, 1)
    plt.ylim(0, max(data[0]) + 100)
    plt.pause(0.01)
    
plt.ion()
update_histogram()
plt.show()

cv2.namedWindow("Picture")
cv2.createTrackbar("Trackbar", "Picture", 50, 100, update_image)

cv2.imshow("Gambar Awal",img1)
cv2.imshow("Picture", img)
cv2.imshow("Original (GrayScale)", ori)

cv2.waitKey(0)
cv2.destroyAllWindows()