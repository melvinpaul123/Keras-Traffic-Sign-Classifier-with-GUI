import numpy as np
import cv2
import sys

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
#capturing video through webcam
print("-> Starting Video Stream...")
print("-> Press Q to Exit the Program")
cap = cv2.VideoCapture(0)
# We need to check if camera is opened previously or not 
if (cap.isOpened() == False):  
    print("-> Error reading video file") 
    sys.exit()
    
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

#load the trained model to classify sign
from tensorflow.keras.models import load_model
model = load_model('trafficsign_classifier.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

while True:
    ret, image = cap.read()
    if ret == False:
            print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            cap.release()
            cv2.destroyAllWindows()
            break
        
    img = image.resize((32,32))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    img=img/255
    cv2.putText(image, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    predictions = model.predict(img)
    probabilityValue =np.amax(predictions)
    pred = model.predict_classes([img])[0]
    sign = classes[pred+1]
    
    if probabilityValue > threshold:
        cv2.putText(image,str(pred+1)+" "+str(sign), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Output", image)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        print("--> Ending Video Stream")
        print("--> Closing All Windows")
        break
    if k == ord('p'): #pause stream
        print("-> Pausing Video Stream")
        print("-> Press any key to continue Video Stream")
        cv2.waitKey(-1) #wait until any key is pressed