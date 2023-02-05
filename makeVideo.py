import cv2
import os
import numpy as np
def fileNames(loc):
    return os.listdir(loc)
def getFrame(dirLoc,index):
    # files=fileNames(dirLoc)
    frame=cv2.imread(dirLoc+str(index)+'.png', cv2.IMREAD_COLOR)
    return frame
dirLoc="D:\OneDrive - IIT Delhi\Desktop\Course content\Sem 6\COL780\output\Aggregated\IBM\out"
resLoc = "D:\OneDrive - IIT Delhi\Desktop\Course content\Sem 6\COL780\output\Aggregated\IB"
frame=getFrame(dirLoc,1)
result = cv2.VideoWriter(resLoc+'M.mp4',cv2.VideoWriter_fourcc('m','p','4','v'),15.0, (frame.shape[1],frame.shape[0]))
counter=1
finalFrame=len(fileNames("D:\OneDrive - IIT Delhi\Desktop\Course content\Sem 6\COL780\output\Aggregated\IBM"))
while(counter<finalFrame):
    frame = getFrame(dirLoc,counter)
    result.write(frame)
    counter+=1
result.release()