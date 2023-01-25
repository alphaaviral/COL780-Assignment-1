import cv2
import numpy as np

video = cv2.VideoCapture('sample.mp4')

# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

class Gaussian:
    # sig=0.0
    # mu=0.0
    # weight=0.0
    # pixelCount = 0
    def __init__(self, value):
        self.sig=1.0
        self.mu=value
        self.weight=0.0
        self.pixelCount=1

    def getValue(self, x):
        return np.exp(-np.power(x - self.mu, 2.0) / (2 * np.power(self.sig, 2.0)))
    
    def checkValue(self, x):
        if abs(x-self.mu)<(2*self.sig):
            return True
        else:
            return False

    def addValue(self, value):
        squareSum = (np.power(self.sig,2.0)+np.power(self.mu,2.0))*self.pixelCount
        squareSum = squareSum+np.power(value,2.0)
        self.mu = 1.0*((self.mu*self.pixelCount)+value)/(self.pixelCount+1)
        sig2 = 1.0*squareSum/(self.pixelCount+1) - np.power(self.mu,2.0)
        self.sig = np.power(sig2,0.5)        
        self.pixelCount = self.pixelCount + 1
        # self.weight = 1.0*self.pixelCount/pixelObj.pixelCount

    def deleteValue(self, value):
        squareSum = (np.power(self.sig,2.0)+np.power(self.mu,2.0))*self.pixelCount
        squareSum = squareSum-np.power(value,2.0)
        self.mu = 1.0*((self.mu*self.pixelCount)-value)/(self.pixelCount-1)
        sig2 = 1.0*squareSum/(self.pixelCount-1) - np.power(self.mu,2.0)
        self.sig = np.power(sig2,0.5)
        self.pixelCount = self.pixelCount - 1
        # self.weight = 1.0*self.pixelCount/pixelObj.pixelCount

class PixelData:
    # pixelRow = 0
    # pixelColumn = 0
    # gaussians = []
    # pixelCount = 0
    # pastValues = {}
    threshold=0.1

    def __init__(self,row,column):
        self.pixelRow = row
        self.pixelColumn = column
        self.gaussians = []
        self.pixelCount = 0
        self.pastValues = []

    def addValue(self,value):
        self.pixelCount = self.pixelCount + 1
        flag = 0
        for i in range (len(self.gaussians)):
            if(self.gaussians[i].checkValue(value)):
                # flag = 1
                if(self.gaussians[i].weight>self.threshold):
                    flag = 1                              #background pixel identified
                self.gaussians[i].addValue(value)
                self.pastValues.append([value,self.gaussians[i]]) #try using list of dict
                break
        if(flag == 0):
            self.gaussians.append(Gaussian(value))
            self.pastValues.append([value,self.gaussians[-1]])

        if self.pixelCount>50:
            deletionVal = self.pastValues[0][0]               #list(self.pastValues.keys())[0]
            gauss = self.pastValues[0][1]       #self.pastValues[deletionVal]
            gauss.deleteValue(deletionVal)
            self.pixelCount = self.pixelCount - 1
            self.pastValues.pop(0)
#after deleteing val, also delte it from pastValues
        for i in range (len(self.gaussians)):
            gauss = self.gaussians[i]
            gauss.weight = gauss.pixelCount/self.pixelCount

        return flag
        # updateweights()

pixelDataList = []
# pixelData = np.array

# for i in range(500):
#     video.set(cv2.CAP_PROP_POS_FRAMES, i)
#     ret, frame = video.read()
#     if(ret):
#         cv2.imshow('frame', frame); cv2.waitKey(0)
#         cv2.imwrite('my_video_frame.png', frame)

# for frame in range(noFrames):
#     for row in range(pixelRows):
#         for column in range(pixelColumns):
#             getPixel[row][column]
frames=5
rows=2
columns=1
pixel = np.array([[0],[0]])
print([pixel.shape])
for row in range(rows):
    for column in range(columns):
        pixelDataList.append(PixelData(row, column))

for frame in range(frames):
    x = input()
    pixel[0][0] = int(x)
    x = input()
    pixel[1][0] = int(x)
    for row in range(rows):
        for column in range(columns):
            pixelValue = pixel[row][column]
            pixelDat = pixelDataList[(row*columns)+column]
            status = pixelDat.addValue(pixelValue)