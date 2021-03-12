#imporing libraries
import numpy as np
import cv2
from threading import Thread
from collections import deque

# Create empty points array
pts = deque(maxlen=50)

# define range of blue color in HSV
Lower_blue = np.array([110,50,50])
Upper_blue = np.array([130,255,255])

# Capture webcame frame  
cap=cv2.VideoCapture(0)

def read():
   while True:
       #reading frames
       ret, img=cap.read()
       
       # Threshold the HSV image to get only green colors
       hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
       
       kernel=np.ones((5,5),np.uint8)
       
       #masking
       mask=cv2.inRange(hsv,Lower_blue,Upper_blue)
       
       #applying dilation and erode
       mask = cv2.erode(mask, kernel, iterations=2)
       mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
       mask = cv2.dilate(mask, kernel, iterations=1)
       
       res=cv2.bitwise_and(img,img,mask=mask)
       
       #finding contours
       cnts,heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
       center = None
 
       if len(cnts) > 0:
           # Get the largest contour and its center 
           c = max(cnts, key=cv2.contourArea)
           ((x, y), radius) = cv2.minEnclosingCircle(c)
           M = cv2.moments(c)
           center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
     
           # Allow only countors that have a larger than 5 pixel radius
           if radius > 5:
               
            #creating circle to the contour     
            cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            
       #appending centers 
       pts.appendleft(center)
       for i in range (1,len(pts)):
           if pts[i-1]is None or pts[i] is None:
               continue
           
           #desplaying the traces
           thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
           cv2.line(img, pts[i-1],pts[i],(0,0,225),thick)
           
       #flipping the frame    
       img=cv2.flip(img,1)
       
        #displaying
       cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
       cv2.namedWindow("res",cv2.WINDOW_NORMAL)
       cv2.imshow("Frame", img)
       cv2.imshow("mask",mask)
       cv2.imshow("res",res)
	
	
       if cv2.waitKey(1) ==ord('q'): #q to close the frame
        break
    
#thread creation
Thread(target=read(),args=())   

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
