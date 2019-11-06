
import numpy as np
import random
import cv2
import os
import tensorflow
from scipy import spatial
class Mask:
    
    def __init__(self,path):
        self.path=path
        self.image=cv2.imread(path)
        self.H,self.W=self.image.shape[:2]
        self.boxes=[]
        self.masks=[]
        
    def findMask(self):
        weights="/home/anoop/python/frozen_inference_graph.pb"
        path="/home/anoop/python/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(weights, path)
        blob = cv2.dnn.blobFromImage(self.image, swapRB=True, crop=False)
        net.setInput(blob)
        self.boxes, self.masks = net.forward(["detection_out_final", "detection_masks"])
        
    def filterMask(self,confidence=0.5):
        finalmask=[]
        finalbox=[]
        
        for i in range(0,self.boxes.shape[2]):
            classid=int(self.boxes[0,0,i,1])
            conf=self.boxes[0,0,i,2]
            if conf>confidence:
                box = (self.boxes[0, 0, i, 3:7])* np.array([self.W, self.H, self.W, self.H])
                startX,startY,endX,endY=box.astype("int")
                boxW = endX - startX
                boxH = endY - startY
                mask=self.masks[i,classid]
                mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_NEAREST)
                finalmask.append(mask)
                finalbox.append([startX,startY,endX,endY])
        return finalmask,finalbox


def findsim(x,y):
    r=1-spatial.distance.cosine(x,y)
    return r

acbox=np.load("/home/anoop/mask-rcnn/images/box12.npy")
ansmask=np.load("/home/anoop/mask-rcnn/images/mask12.npy")
boxshape=np.load("/home/anoop/mask-rcnn/images/boxshape.npy")

ansbox=[]
for i in acbox:
    ansbox.append(i-50)
    ansbox.append(i+50)
    

ob=Mask("/home/anoop/mask-rcnn/images/india_map(1).jpg")
ob.findMask()
mask,box=ob.filterMask()
mask=np.asarray(mask)
mask1=cv2.resize(mask[0], (boxshape[0],boxshape[1]),interpolation=cv2.INTER_NEAREST)
mask1=mask1.flatten()
ansmask1=ansmask.flatten()
if box[0] in range(ansbox[0],ansbox[1]) and box[1] in range(ansbox[2],ansbox[3]) and box[2] in range(ansbox[4],ansbox[5]) and box[3] in range(ansbox[6],ansbox[7]):
r=findsim(ansmask1,mask1)
print("percentage similarity{}".format(r))

else:
    print("Out of position")
    
    



    



        
        


        
    
    
    
    






