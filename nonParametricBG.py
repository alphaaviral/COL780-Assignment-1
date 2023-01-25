import cv2
import numpy as np
import statistics
import random


threshold = 0.001

class PixelData:
    def __init__(self, row, column):
        self.sigma = 0.1
        self.pastValues = []
        self.row = row
        self.column = column

    def kernelGetProbability(self,currentValue):
        sum=0
        for i in range(0,len(self.pastValues)):

            sum = sum + (1.0*(np.exp((np.power(currentValue-self.pastValues[i],2.0))/(-1*2*(np.power(self.sigma,2.0)))))/(np.power(2*np.pi,0.5)*self.sigma*len(self.pastValues)))
            if sum>threshold:
                return threshold+0.1
            # sum = sum+((np.power(np.power(2*np.pi,0.5)*self.sigma,-1.0)*np.exp(-1*np.power((currentValue-self.pastValues[i]),2.0)/(2*np.power(self.sigma,2.0))))/len(self.pastValues))
        # sum=1.0*sum/(np.power(2*np.pi,0.5)*self.sigma*len(self.pastValues))
        return sum
    
    def updateSigma(self):
        diff = []
        for i in range(0,len(self.pastValues)-1):
            diff.append(abs(self.pastValues[i+1] - self.pastValues[i]))
        
        med = statistics.median(diff)
        if med==0:
            self.sigma = 0.1
        else:
            self.sigma = med/(0.68*np.power(2,0.5))

pixelDataList = []

cap = cv2.VideoCapture('sample2.mp4')
columns = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for row in range(rows):
    for column in range(columns):
        pixelDataList.append(PixelData(row, column))

for frameNo in range(45):
    if(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for pixel in pixelDataList:
                pixel.pastValues.append(int(gray[pixel.row][pixel.column]))
            cv2.imshow('video', gray)
            cv2.waitKey(50)
        else:
            break

for pixel in pixelDataList:
    pixel.updateSigma()

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:
        res = np.zeros((rows,columns))
        for pixel in pixelDataList:
            if pixel.kernelGetProbability(int(gray[pixel.row][pixel.column]))<threshold :
                res[pixel.row][pixel.column] = 255
        cv2.imshow('video', gray)
        cv2.imshow('result', res)
        cv2.waitKey(50)
    else:
        break