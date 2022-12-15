# Importing the libraries and tools needed by our project to detect objects 
import cv2
import numpy as np
import time

# We will be needing a YoloV3 weights file and YoloV3 config file to run the project 
net = cv2.dnn.readNet('Yolo/yolov3.weights', 'Yolo/yolov3.config')
classes= []

# Providing webcam feed to a variable named cap And later using it to detect objects in the given feed 
cap = cv2.VideoCapture(0)
pTime = 0

# We will also be needing a file with coconames in order to recognise the objects with their names 
with open('Yolo/coconames.txt','r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
# print(layer_names)
outputlayes = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# generating an array with colors combination with the help of np.random.uniform method
# np (numpy).random.uniform methods first parameter is used to get result between the given values 
# For example here it is 0 and 255 so our resulted randomly generated values will be between 0 and 255
# second parameter size is used to define the size of array with two paramters 
# Second parameter defines what will be the length of array with randomly generated values 
# And first parameter of size defines what will be the  length of each value in the array 
# Like here the array will be generated of length 80. Which means it will be containing 80 items in it 
# And each item in the array will also be an array of length 3 (the second parameter of size does this)


colors = np.random.uniform(0,255, size=(len(classes),3))
# print(colors,(len(colors)))
# Loading image 
# img = cv2.imread('images/1.jpg')

# Code will run for forever until we close it 
while True:
    # Reading our webcam feed to get it as images 
    success, imgS = cap.read()
    # Changing color of our image from BGR(blue green red) to RGB(red green blue). Because it's easy to our python interpreter to read a RGB image 
    img = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # Resizing our image to scale 1.4 
    img =  cv2.resize(img, None, fx=1.4, fy=1.4)
    height,width,channels = img.shape

# detecting Objects 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)
    for b in blob:
        for n,imgBlob in enumerate(b):
            pass
        # cv2.imshow(str(n), imgBlob)


    net.setInput(blob)
    outs = net.forward(outputlayes)


# Putting text and border on objects 
    class_ids = []
    confidences = []
    boxes = []


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)


                # Rectangle coordinates 
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    # print(len(boxes))
    objects_detected = len(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range (len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            # Drawing rectangle on every object detected in the given video/image 
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            # Putting the object names on the top of the objects detected in the given video/image 
            cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime        
    # Changing the color againg from RGB to BGR to make our image in it's real format of colors 
    new_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)        
    # Putting text on the top of python interpretor window to show frames per second 
    cv2.putText(new_image, f'FPS: {int(fps)}', (20,70), font, 3, colors[int(np.random.uniform()*5)],2)
    # Showing the images converted from our webcam feed 
    cv2.imshow('image',new_image)
    # Setting wait time for every image to be 1 millisecond 
    cv2.waitKey(1)
    # cv2.destroyAllWindows()