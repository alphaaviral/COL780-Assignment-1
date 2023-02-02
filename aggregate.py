import numpy as np
import cv2

def calculateIntegral(image):
    width = image.shape[1]
    height = image.shape[0]
    res = np.zeros(image.shape)
    for i in range(height):
        for j in range(width):
            if j>0 and i>0:
                res[i][j]  = res[i][j-1] + res[i-1][j] - res[i-1][j-1] + image[i][j]
            elif j==0 and i==0:
                res[i][j] = image[i][j]
            elif j==0:
                res[i][j] = res[i-1][j] + image[i][j]
            elif i==0:
                res[i][j] = res[i][j-1] + image[i][j]
    return res

def makeWhite(image, size, startx, starty):
    for m in range(startx, startx+size):
        for n in range(starty, starty+size):
            image[m][n] = 255
    return image

def makeBlack(image, size, startx, starty):
    for m in range(startx, startx+size):
        for n in range(starty, starty+size):
            image[m][n] = 0
    return image

frameNo=1
boxSize = 5
threshold = 0.4

while(1):
    path = 'output\kaggle\IBM with FDR, 20, 0.00001, 0.05\out'+str(frameNo)+'.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        break
    integral = calculateIntegral(img)
    i=0
    j=0
    res = np.zeros(img.shape)
    while i*boxSize<img.shape[0]:
        j=0
        while j*boxSize<img.shape[1]:
            if i>0 and j>0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1]+integral[(i*boxSize)-1][(j*boxSize)-1]-integral[(i*boxSize)+boxSize-1][(j*boxSize)-1]-integral[(i*boxSize)-1][(j*boxSize)+boxSize-1] 
            elif i==0 and j==0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1]
            elif i==0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1] - integral[(i*boxSize)+boxSize-1][(j*boxSize)-1]
            elif j==0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1] - integral[(i*boxSize)-1][(j*boxSize)+boxSize-1]
            
            if sum/(255*boxSize*boxSize) >= threshold:
                    img = makeWhite(img, boxSize, i*boxSize, j*boxSize)
            else:
                img = makeBlack(img, boxSize, i*boxSize, j*boxSize)
            j = j+1
        i = i+1    
    cv2.imwrite('output/out'+str(frameNo)+'.png',img)
    frameNo = frameNo + 1