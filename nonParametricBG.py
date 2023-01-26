import cv2
import numpy as np
import statistics
import random
import os

# global threshold
threshold = 0.01
defaultSigma = 0.2
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
            self.sigma = defaultSigma
        else:
            self.sigma = med/(0.68*np.power(2,0.5))


# #Running on my sample

# pixelDataList = []

# cap = cv2.VideoCapture('sample2.mp4')
# columns = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# for row in range(rows):
#     for column in range(columns):
#         pixelDataList.append(PixelData(row, column))

# for frameNo in range(45):
#     if(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             for pixel in pixelDataList:
#                 pixel.pastValues.append(int(gray[pixel.row][pixel.column]))
#             cv2.imshow('video', gray)
#             cv2.waitKey(50)
#         else:
#             break

# for pixel in pixelDataList:
#     pixel.updateSigma()

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if ret:
#         res = np.zeros((rows,columns))
#         for pixel in pixelDataList:
#             if pixel.kernelGetProbability(int(gray[pixel.row][pixel.column]))<threshold :
#                 res[pixel.row][pixel.column] = 255
#         cv2.imshow('video', gray)
#         cv2.imshow('result', res)
#         cv2.waitKey(50)
#     else:
#         break

#Running on given dataset

for defSig in range(0,3):
    # os.mkdir('./sig'+str(defaultSigma))
    threshold = 0.01
    for th in range(0,3):
        # os.mkdir('./sig'+str(defaultSigma)+'/th'+str(threshold))
        pixelDataList = []
        init_frame = cv2.imread('col dataset\COL780 Dataset\HighwayI\HighwayI\input\in000000.png', cv2.IMREAD_COLOR)
        cv2.imshow('in', init_frame)
        cv2.waitKey(5000)
        rows = init_frame.shape[0]
        columns = init_frame.shape[1]

        for row in range(rows):
            for column in range(columns):
                pixelDataList.append(PixelData(row, column))

        for i in range(100):
            for pixel in pixelDataList:
                pixel.pastValues.append(int(init_frame[pixel.row][pixel.column]))
                # cv2.imshow('video',init_frame)
                # cv2.waitKey(50)

        for pixel in pixelDataList:
            pixel.updateSigma()

        frameNo = 0
        while(frameNo<5):
            if frameNo <10:
                frame_path = 'col dataset\COL780 Dataset\HighwayI\HighwayI\input\in00000' + str(frameNo) +'.png'
            elif frameNo <100:
                frame_path = 'col dataset\COL780 Dataset\HighwayI\HighwayI\input\in0000' + str(frameNo) +'.png'
            else:
                frame_path = 'col dataset\COL780 Dataset\HighwayI\HighwayI\input\in000' + str(frameNo) +'.png'

            current_frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if current_frame is None:
                break

            res = np.zeros(init_frame.shape)
            for pixel in pixelDataList:
                if pixel.kernelGetProbability(int(current_frame[pixel.row][pixel.column]))<threshold:
                    res[pixel.row][pixel.column] = 255
            
            # cv2.imshow('video',current_frame)
            # cv2.imshow('out', res)
            # cv2.waitKey(1)
            outPath = './sig'+str(defaultSigma)+'/th'+str(threshold)+'/out'+str(frameNo)+'.png'
            cv2.imwrite(outPath,res)
            frameNo = frameNo+1
        
        threshold = threshold*0.5
    defaultSigma = defaultSigma+0.5