import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab // Image from the file
path = 'ImagesAttendance' #path of stored images
images = [] #list of all images
classNames = [] #name of images
myList = os.listdir(path)
print(myList)
#the loop is for import the images one by one using their name
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}') #read image one by one
    images.append(curImg) #append the current image
    classNames.append(os.path.splitext(cl)[0]) #image name without extension
print(classNames)

#Encoding Specified Images
def findEncodings(images):
    encodeList = []
    for img in images: #loop through all the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0] #find the encode of image.We are using 0 beacause we are sending a single image
        encodeList.append(encode) #add the encode in encodelist[]
    return encodeList

#for the attendence sheet
#Attendent list input read and write file
def markAttendance(name):
    from datetime import date
    today = date.today()
    filename1 = (today)
    #with open('Attendance-'+str(filename1)+'.csv','w+') as f:
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',') #name and time will be separated based on comma
            nameList.append(entry[0]) #entry 0 means only the names
        if name not in nameList: #check if the current name is present or not if alredady present we will not add it again
            now = datetime.now() #this will give us the date and time
            dtString = now.strftime('%H:%M:%S') #format of time
            f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

#Capture Screen from webcam
#url="http://192.168.0.104:8080/video"
cap = cv2.VideoCapture(0) #for the webcame
while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) #compressing the image to one forth its size to speed up our process
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #convert image from BGR to RGB

    facesCurFrame = face_recognition.face_locations(imgS) #image location of all the faces avaible on webcame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #encode of all the faces from the webcame
    # iterate all the faces we have found in our current frame
    # then matches them with all the encode we have found before named encodeListKnown
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # print(faceDis)
        matchIndex = np.argmin(faceDis) # lowest facedis means best match

        #if face match mark it on attendence List
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)

        #This portion for Detect Unknown Face
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        # we are mul by 4 because before we have compressed the image one forth of its size
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        # green rectangle box around the face with thickness of 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)