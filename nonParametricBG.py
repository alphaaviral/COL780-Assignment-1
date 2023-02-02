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

global threshold,FDRthreshold
threshold = 0.005
FDRthreshold = 0.005
defaultSigma = 1.2
n = 20

class PixelData:
    def __init__(self, row, column):
        self.sigma = 0.
        self.pastValues = []
        self.row = row
        self.column = column
        self.diff = []

    def kernelGetProbability(self,currentValue):
        sum=0
        for i in range(0,len(self.pastValues)):
            self.sigma = round(self.sigma,1)
            difference = abs(int(currentValue-self.pastValues[i]))
            sigIndex = int((self.sigma*10)-1)
            result = lookupData[difference][sigIndex]
            sum = sum+(1.0*result/len(self.pastValues))
            if sum>threshold:
                return threshold+0.1
        return sum
    
    def updateSigma(self):
        med = np.median(self.diff)
        self.sigma = med/(0.68*np.power(2,0.5))
        if self.sigma==0:
            self.sigma = defaultSigma

    def addValue(self, value):
        if len(self.pastValues)>=n:
            self.pastValues.pop(0)
            self.diff.pop(0)

        self.pastValues.append(value)
        if len(self.pastValues)>1:
            difference = abs(self.pastValues[-1] - self.pastValues[-2])
            self.diff.append(difference)


def getProbabilities(pixels, frame, result):
    global threshold
    for i in range(len(pixels)):
        pixel = pixels[i]
        if pixel.kernelGetProbability(int(frame[pixel.row][pixel.column]))<threshold:
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
                probabilities.append(pixels[(neighbour[0]*frame.shape[1])+neighbour[1]].kernelGetProbability(int(frame[pixel.row][pixel.column])))
            maxima = max(probabilities)
            if maxima>FDRthreshold:
                retArray[(pixel.row*frame.shape[1])+pixel.column] = 1

#Running on given dataset
if __name__ == '__main__':
    pixelDataList = []

    init_frame = cv2.imread('col dataset\COL780 Dataset\Candela_m1.10\Candela_m1.10\input\Candela_m1.10_000000.png', cv2.IMREAD_GRAYSCALE)

    rows = init_frame.shape[0]
    columns = init_frame.shape[1]

    for row in range(rows):
        for column in range(columns):
            pixelDataList.append(PixelData(row, column))

    for i in range(2):
        for pixel in pixelDataList:
            pixel.addValue(int(init_frame[pixel.row][pixel.column]))

    for pixel in pixelDataList:
        pixel.updateSigma()

    frameNo = 0
    basePath = 'col dataset\COL780 Dataset\Candela_m1.10\Candela_m1.10\input\Candela_m1.10_000'
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
        #     pixel.addValue(int(current_frame[pixel.row][pixel.column]))
        #     pixel.updateSigma()
        for process in processes:
            process.join()

        for i in range(4):
            for j in range(len(processRes[i])):
                if(processRes[i][j]==255):
                    res[splitted_lists[i][j].row][splitted_lists[i][j].column]=255
        
        removalThreads = 3
        false_detect = multiprocessing.Array('i', len(pixelDataList))

        processes = []
        for i in range(removalThreads):
            processes.append(multiprocessing.Process(target = removeFalseDetection, args=(pixelDataList, res, current_frame, false_detect, int((len(pixelDataList)/removalThreads)*i), int(len(pixelDataList)/removalThreads))))
        
        for process in processes:
            process.start()
        
        for pixel in pixelDataList:
            pixel.addValue(int(current_frame[pixel.row][pixel.column]))
            pixel.updateSigma()
        
        for process in processes:
            process.join()

        for i in range(len(pixelDataList)):
            if false_detect[i]==1:
                res[pixelDataList[i].row][pixelDataList[i].column]=0

        outPath = './output/out'+str(frameNo)+'.png'
        cv2.imwrite(outPath,res)
        frameNo = frameNo+1