import cv2
import numpy as np
import statistics
import random
import os
import csv

datafile = open('lookupTable.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
lookupData = np.zeros((0,300))

for row in datareader:
    if len(row)==0:
        continue
    lookupData = np.append(lookupData, [row],0)
lookupData = np.asarray(lookupData, dtype=float)

# global threshold
threshold = 0.001
defaultSigma = [1.2,1.2,1.2]
n = 5

class PixelData:
    def __init__(self, row, column):
        self.sigma = []
        self.pastValues = np.zeros((0,3))
        self.row = row
        self.column = column
        self.diff = np.zeros((0,3))

    def kernelGetProbability(self,currentValue):
        sum=0
        for i in range(0,len(self.pastValues)):
            product = 1
            for index in range(0,3):
                difference = abs(int(currentValue[index]-self.pastValues[i][index]))
                sigIndex = int((self.sigma[index]*100)-1)
                product = product * lookupData[difference][sigIndex]
                # product = product*(1.0*(np.exp((np.power(currentValue[index]-self.pastValues[i][index],2.0))/(-1*2*(np.power(self.sigma[index],2.0)))))/(np.power(2*np.pi,0.5)*self.sigma[index]))
            sum = sum+(1.0*product/len(self.pastValues))
            if sum>threshold:
                return threshold+0.1
            # sum = sum+((np.power(np.power(2*np.pi,0.5)*self.sigma,-1.0)*np.exp(-1*np.power((currentValue-self.pastValues[i]),2.0)/(2*np.power(self.sigma,2.0))))/len(self.pastValues))
        # sum=1.0*sum/(np.power(2*np.pi,0.5)*self.sigma*len(self.pastValues))
        return sum
    
    def updateSigma(self):
        med = np.median(self.diff, axis = 0)
        if 0 in med:
            self.sigma = defaultSigma
        else:
            self.sigma = med/(0.68*np.power(2,0.5))

    def addValue(self, value):
        if self.pastValues.shape[0]>=n:
            self.pastValues = np.delete(self.pastValues,0,0)
            self.diff = np.delete(self.diff, 0,0)

        self.pastValues = np.append(self.pastValues, [value], axis=0)
        if self.pastValues.shape[0]>1:
            difference = [abs(self.pastValues[-1][0] - self.pastValues[-2][0]), abs(self.pastValues[-1][1] - self.pastValues[-2][1]), abs(self.pastValues[-1][2] - self.pastValues[-2][2])]
            self.diff = np.append(self.diff, [difference], axis=0)

#Running on given dataset

pixelDataList = []

init_frame = cv2.imread('col dataset\COL780 Dataset\HighwayI\HighwayI\input\in000000.png', cv2.IMREAD_COLOR)

rows = init_frame.shape[0]
columns = init_frame.shape[1]

for row in range(rows):
    for column in range(columns):
        pixelDataList.append(PixelData(row, column))

for i in range(n):
    for pixel in pixelDataList:
        pixel.addValue(init_frame[pixel.row][pixel.column])
        # cv2.imshow('video',init_frame)
        # cv2.waitKey(50)

for pixel in pixelDataList:
    pixel.updateSigma()

frameNo = 0
while(1):
    if frameNo <10:
        frame_path = 'col dataset\COL780 Dataset\HighwayI\HighwayI\input\in00000' + str(frameNo) +'.png'
    elif frameNo <100:
        frame_path = 'col dataset\COL780 Dataset\HighwayI\HighwayI\input\in0000' + str(frameNo) +'.png'
    else:
        frame_path = 'col dataset\COL780 Dataset\HighwayI\HighwayI\input\in000' + str(frameNo) +'.png'

    current_frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if current_frame is None:
        break

    res = np.zeros((init_frame.shape[0],init_frame.shape[1]))
    for pixel in pixelDataList:
        if pixel.kernelGetProbability(current_frame[pixel.row][pixel.column])<threshold:
            res[pixel.row][pixel.column] = 255
    
    # cv2.imshow('video',current_frame)
    # cv2.imshow('out', res)
    # cv2.waitKey(1)
    outPath = './output/out'+str(frameNo)+'.png'
    cv2.imwrite(outPath,res)
    frameNo = frameNo+1
        
    #     threshold = threshold*0.5
    # defaultSigma = defaultSigma+0.5