import cv2
import numpy as np
import statistics
import random
import os
import csv
import multiprocessing

datafile = open('lookupTable.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
lookupData = np.zeros((0,3000))

for row in datareader:
    if len(row)==0:
        continue
    lookupData = np.append(lookupData, [row],0)
lookupData = np.asarray(lookupData, dtype=float)

global threshold,FDRthresholdm, shortn, longn
threshold = 0.005
FDRthreshold = 0.05
defaultSigma = 1.2
shortn = 20
longn = 20
longTermUpdateRate = 3

class PixelData:
    def __init__(self, row, column):
        self.shortSigma = 0.
        self.longSigma = 0.
        self.shortPastValues = []
        self.longPastValues = []
        self.row = row
        self.column = column
        self.shortDiff = []
        self.longDiff = []

    def kernelGetShortProbability(self,currentValue):
        sum=0
        for i in range(0,len(self.shortPastValues)):
            self.shortSigma = round(self.shortSigma,1)
            difference = abs(int(currentValue-self.shortPastValues[i]))
            sigIndex = int((self.shortSigma*10)-1)
            result = lookupData[difference][sigIndex]
            sum = sum+(1.0*result/len(self.shortPastValues))
            if sum>threshold and sum>FDRthreshold:
                return sum
        return sum
    
    def kernelGetLongProbability(self, currentValue):
        sum=0
        for i in range(0,len(self.longPastValues)):
            self.longSigma = round(self.longSigma,1)
            difference = abs(int(currentValue-self.longPastValues[i]))
            sigIndex = int((self.longSigma*10)-1)
            result = lookupData[difference][sigIndex]
            sum = sum+(1.0*result/len(self.longPastValues))
            if sum>threshold:
                return sum
        return sum
    
    # def updateSigma(self):
    #     med = np.median(self.shortDiff)
    #     self.shortSigma = med/(0.68*np.power(2,0.5))
    #     if self.shortSigma==0:
    #         self.shortSigma = defaultSigma

    def addShortValue(self, value):
        global shortn
        if len(self.shortPastValues)>=shortn:
            self.shortPastValues.pop(0)
            self.shortDiff.pop(0)

        self.shortPastValues.append(value)
        if len(self.shortPastValues)>1:
            difference = abs(self.shortPastValues[-1] - self.shortPastValues[-2])
            self.shortDiff.append(difference)

            med = np.median(self.shortDiff)
            self.shortSigma = med/(0.68*np.power(2,0.5))
            if self.shortSigma==0:
                self.shortSigma = defaultSigma
    
    def addLongValue(self, value):
        global longn
        if len(self.longPastValues)>=longn:
            self.longPastValues.pop(0)
            self.longDiff.pop(0)

        self.longPastValues.append(value)
        if len(self.longPastValues)>1:
            difference = abs(self.longPastValues[-1] - self.longPastValues[-2])
            self.longDiff.append(difference)

            med = np.median(self.longDiff)
            self.longSigma = med/(0.68*np.power(2,0.5))
            if self.longSigma==0:
                self.longSigma = defaultSigma


def getProbabilities(pixels, frame, result):
    global threshold
    for i in range(len(pixels)):
        pixel = pixels[i]
        if pixel.kernelGetShortProbability(int(frame[pixel.row][pixel.column]))<threshold and pixel.kernelGetLongProbability(int(frame[pixel.row][pixel.column]))<threshold:
            result[i] = 255

def getNeighbourIndices(arrShape, i, j):
    n = arrShape[0]
    m = arrShape[1]

    neighbours = []
    for dx in range (-1 if (i > 0) else 0 , 2 if (i < n-1) else 1):
        for dy in range( -1 if (j > 0) else 0,2 if (j < m-1) else 1):
            if (dx != 0 or dy != 0):
                neighbours.append((i + dx,j + dy))
    return neighbours

def removeFalseDetection(pixels, resMatrix, frame, retArray, startIndex, count):
    global FDRthreshold
    for i in range(startIndex, startIndex+count):
        pixel = pixels[i]
        if resMatrix[pixel.row][pixel.column]==255:
            probabilities = []
            neighbours = getNeighbourIndices(resMatrix.shape, pixel.row, pixel.column)
            for neighbour in neighbours:
                probabilities.append(pixels[(neighbour[0]*frame.shape[1])+neighbour[1]].kernelGetShortProbability(int(frame[pixel.row][pixel.column])))
            maxima = max(probabilities)
            if maxima>FDRthreshold:
                retArray[(pixel.row*frame.shape[1])+pixel.column] = 1

#Running on given dataset
if __name__ == '__main__':
    pixelDataList = []

    init_frame = cv2.imread('col dataset\COL780 Dataset\HighwayI\HighwayI\input\in000000.png', cv2.IMREAD_GRAYSCALE)

    rows = init_frame.shape[0]
    columns = init_frame.shape[1]

    for row in range(rows):
        for column in range(columns):
            pixelDataList.append(PixelData(row, column))

    for i in range(2):
        for pixel in pixelDataList:
            pixel.addShortValue(int(init_frame[pixel.row][pixel.column]))

    # for pixel in pixelDataList:
    #     pixel.updateSigma()

    frameNo = 0
    basePath = 'col dataset\COL780 Dataset\HighwayI\HighwayI\input\in000'
    while(1):
        if frameNo <10:
            frame_path = basePath+ '00' + str(frameNo) +'.png'
        elif frameNo <100:
            frame_path = basePath+'0' + str(frameNo) +'.png'
        else:
            frame_path = basePath + str(frameNo) +'.png'

        current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if current_frame is None:
            break
        
        res = np.zeros((init_frame.shape[0],init_frame.shape[1]))
        processnumber = 4
        splitted_lists = np.array_split(pixelDataList,processnumber)
        processRes = []
        for i in range(processnumber):
            processRes.append(multiprocessing.Array('i', int(len(pixelDataList)/processnumber)))

        processes = []
        for i in range(processnumber):
            processes.append(multiprocessing.Process(target = getProbabilities, args=(splitted_lists[i], current_frame, processRes[i])))
       
        for process in processes:
            process.start()
        # for pixel in pixelDataList:
        #     pixel.addShortValue(int(current_frame[pixel.row][pixel.column]))
        for process in processes:
            process.join()

        for i in range(4):
            for j in range(len(processRes[i])):
                if(processRes[i][j]==255):
                    res[splitted_lists[i][j].row][splitted_lists[i][j].column]=255
        
        # removalThreads = 4
        # false_detect = multiprocessing.Array('i', len(pixelDataList))

        # processes = []
        # for i in range(removalThreads):
        #     processes.append(multiprocessing.Process(target = removeFalseDetection, args=(pixelDataList, res, current_frame, false_detect, int((len(pixelDataList)/removalThreads)*i), int(len(pixelDataList)/removalThreads))))
        
        # for process in processes:
        #     process.start()
        
        # if frameNo%longTermUpdateRate == 0:
        #     for pixel in pixelDataList:
        #         pixel.addLongValue(int(current_frame[pixel.row][pixel.column]))
        #     # pixel.updateSigma()
        
        # for process in processes:
        #     process.join()

        # for i in range(len(pixelDataList)):
        #     if false_detect[i]==1:
        #         res[pixelDataList[i].row][pixelDataList[i].column]=0
        
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if res[i][j] == 255:
                    pixelDataList[(i*res.shape[1]) + j].addShortValue(int(current_frame[i][j]))

        outPath = './output/out'+str(frameNo)+'.png'
        cv2.imwrite(outPath,res)
        frameNo = frameNo+1