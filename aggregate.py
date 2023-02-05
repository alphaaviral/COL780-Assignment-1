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

def makeWhite(image, xsize, ysize, startx, starty):
    for m in range(startx, startx+xsize):
        for n in range(starty, starty+ysize):
            image[m][n] = 255
    return image

def makeBlack(image, xsize, ysize, startx, starty):
    for m in range(startx, startx+xsize):
        for n in range(starty, starty+ysize):
            image[m][n] = 0
    return image

frameNo=1
boxSize = 5
threshold = 0.4

while(1):
    path = 'output\kaggle\BW\IBM\with FDR 0.1, cinter 3, 20, 0.005\out'+str(frameNo)+'.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        break
    integral = calculateIntegral(img)
    i=0
    j=0
    res = np.zeros(img.shape)
    while (i+1)*boxSize<img.shape[0]:
        j=0
        while (j+1)*boxSize<img.shape[1]:
            if i>0 and j>0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1]+integral[(i*boxSize)-1][(j*boxSize)-1]-integral[(i*boxSize)+boxSize-1][(j*boxSize)-1]-integral[(i*boxSize)-1][(j*boxSize)+boxSize-1] 
            elif i==0 and j==0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1]
            elif i==0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1] - integral[(i*boxSize)+boxSize-1][(j*boxSize)-1]
            elif j==0:
                sum = integral[(i*boxSize)+boxSize-1][(j*boxSize)+boxSize-1] - integral[(i*boxSize)-1][(j*boxSize)+boxSize-1]
            
            if sum/(255*boxSize*boxSize) >= threshold:
                    img = makeWhite(img, boxSize,boxSize, i*boxSize, j*boxSize)
            else:
                img = makeBlack(img, boxSize,boxSize, i*boxSize, j*boxSize)
            j = j+1
        sum = integral[(i*boxSize)+boxSize-1][img.shape[1]-1]+integral[(i*boxSize)-1][(j*boxSize)-1]-integral[(i*boxSize)+boxSize-1][(j*boxSize)-1]-integral[(i*boxSize)-1][img.shape[1]-1] 
        if sum/(255*boxSize*(img.shape[1]-(j*boxSize))) >= threshold:
            img = makeWhite(img, boxSize,img.shape[1]-(j*boxSize),  i*boxSize, j*boxSize)
        else:
            img = makeBlack(img, boxSize,img.shape[1]-(j*boxSize), i*boxSize, j*boxSize)

        i = i+1    
    
    j=0
    while (j+1)*boxSize<img.shape[1]:
        if i>0 and j>0:
            sum = integral[img.shape[0]-1][(j*boxSize)+boxSize-1]+integral[(i*boxSize)-1][(j*boxSize)-1]-integral[img.shape[0]-1][(j*boxSize)-1]-integral[(i*boxSize)-1][(j*boxSize)+boxSize-1] 
        elif j==0:
            sum = integral[img.shape[0]-1][(j*boxSize)+boxSize-1] - integral[(i*boxSize)-1][(j*boxSize)+boxSize-1]
        
        if sum/(255*(img.shape[0]-(i*boxSize))*boxSize) >= threshold:
                img = makeWhite(img, (img.shape[0]-(i*boxSize)),boxSize, i*boxSize, j*boxSize)
        else:
            img = makeBlack(img, (img.shape[0]-(i*boxSize)),boxSize, i*boxSize, j*boxSize)
        j = j+1
    sum = integral[img.shape[0]-1][img.shape[1]-1]+integral[(i*boxSize)-1][(j*boxSize)-1]-integral[img.shape[0]-1][(j*boxSize)-1]-integral[(i*boxSize)-1][img.shape[1]-1] 
    if sum/(255*(img.shape[0]-(i*boxSize))*(img.shape[1]-(j*boxSize))) >= threshold:
        img = makeWhite(img, (img.shape[0]-(i*boxSize)),img.shape[1]-(j*boxSize)-1,  i*boxSize, j*boxSize)
    else:
        img = makeBlack(img, (img.shape[0]-(i*boxSize)),img.shape[1]-(j*boxSize)-1, i*boxSize, j*boxSize)

    cv2.imwrite('output/temp/out'+str(frameNo)+'.png',img)
    frameNo = frameNo + 1