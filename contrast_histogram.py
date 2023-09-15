import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

img = cv2.imread("itsSurabaya2.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)
img[:, :] /= 255
ori = copy.deepcopy(img)

baris=img.shape[0]
kolom=img.shape[1]
contrast=0.1

def update_image(value):
    global img, ori
    img = copy.deepcopy(ori)
    
    contrast = value / 100.0
    
    for i in range(baris):
        for j in range(kolom):
            if (img[i,j]<0.5) :
                img[i,j]-=contrast-0.5
            else :
                img[i,j]+=contrast-0.5
                
    cv2.imshow("Picture", img)
    update_histogram()

img1 = cv2.imread("itsSurabaya2.jpg")

cv2.namedWindow("Picture")
cv2.createTrackbar("Trackbar", "Picture", 50, 100, update_image)

cv2.imshow("Gambar Awal",img1)
cv2.imshow("Picture", img)
cv2.imshow("Original (GrayScale)", ori)

img[:,:] *= 255
img[:,:] = np.floor(img[:,:])
img = img.astype(np.uint8)

def update_histogram():
    global img
    img_histogram = (img * 255).astype(np.uint8)
    
    Histogram = np.zeros((256, 1), np.int32)
    for i in range(baris):
        for j in range(kolom):
            r = img_histogram[i, j]
            Histogram[r] += 1

    plt.clf()
    plt.bar(np.arange(0, 256, 1), Histogram[:, 0])
    plt.xlabel("Pixel")
    plt.pause(0.01)
    
plt.ion()
update_histogram()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
